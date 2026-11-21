//! Local code-embedding reranker (`--features embed`). It re-orders Lode's top context hits by
//! fusing their lexical rank with the cosine similarity between the query and each hit's
//! source, embedded with `jinaai/jina-embeddings-v2-base-code` on the CPU through candle.
//!
//! Nothing here touches the network except `download_model`, which the user runs
//! explicitly (`lode rerank-model --download`). `--rerank` reads the local Hugging Face cache
//! only and fails with a clear error when the model is missing.

use std::path::{Path, PathBuf};

use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear, linear_no_bias,
};
use rusqlite::{Connection, OptionalExtension};
use tokenizers::{PaddingParams, Tokenizer, TruncationDirection, TruncationParams};

use crate::context::Rerank;
use crate::storage::NodeRow;
use crate::{Error, Result};

pub const MODEL_ID: &str = "jinaai/jina-embeddings-v2-base-code";
pub const MODEL_FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];
/// Token cap for one hit or query; ALiBi has no position limit, this bounds CPU time.
const MAX_TOKENS: usize = 128;
/// Source lines read per hit before tokenizing.
const MAX_LINES: usize = 40;
/// Hits past this lexical rank keep their place.
const RERANK_DEPTH: usize = 10;
/// Reciprocal-rank fusion: `1/(K + lexical rank) + EMBED_WEIGHT/(K + embedding rank)`.
/// Tuned on RepoBench-R (see PORTING.md): heavier embedding weight lifts small repos but
/// costs the hard bucket, so the embedding mostly breaks near-ties in lexical order.
const RRF_K: f64 = 5.0;
const EMBED_WEIGHT: f64 = 0.25;

fn model_err(err: impl std::fmt::Display) -> Error {
    Error::runtime(format!("Rerank model: {err}"))
}

fn hf_cache() -> hf_hub::Cache {
    hf_hub::Cache::from_env()
}

/// Paths of the model files in the local Hugging Face cache, if all are present.
pub fn cached_model_files() -> Option<Vec<PathBuf>> {
    let repo = hf_cache().model(MODEL_ID.to_owned());
    MODEL_FILES.iter().map(|file| repo.get(file)).collect()
}

/// Fetches the model into the Hugging Face cache (the only networked call in Lode).
pub fn download_model() -> Result<Vec<PathBuf>> {
    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .map_err(model_err)?;
    let repo = api.model(MODEL_ID.to_owned());
    MODEL_FILES
        .iter()
        .map(|file| repo.get(file).map_err(model_err))
        .collect()
}

#[derive(Debug, serde::Deserialize)]
struct Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    layer_norm_eps: f64,
    type_vocab_size: usize,
}

#[derive(Debug)]
struct Layer {
    query: Linear,
    key: Linear,
    value: Linear,
    norm_q: LayerNorm,
    norm_k: LayerNorm,
    attn_out: Linear,
    attn_norm: LayerNorm,
    norm_1: LayerNorm,
    up_gated: Linear,
    down: Linear,
    norm_2: LayerNorm,
}

impl Layer {
    fn new(cfg: &Config, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        let (hidden, eps) = (cfg.hidden_size, cfg.layer_norm_eps);
        let attn = vb.pp("attention");
        let this = attn.pp("self");
        Ok(Self {
            query: linear(hidden, hidden, this.pp("query"))?,
            key: linear(hidden, hidden, this.pp("key"))?,
            value: linear(hidden, hidden, this.pp("value"))?,
            norm_q: layer_norm(hidden, eps, this.pp("layer_norm_q"))?,
            norm_k: layer_norm(hidden, eps, this.pp("layer_norm_k"))?,
            attn_out: linear(hidden, hidden, attn.pp("output").pp("dense"))?,
            attn_norm: layer_norm(hidden, eps, attn.pp("output").pp("LayerNorm"))?,
            norm_1: layer_norm(hidden, eps, vb.pp("layer_norm_1"))?,
            up_gated: linear_no_bias(
                hidden,
                2 * cfg.intermediate_size,
                vb.pp("mlp").pp("up_gated_layer"),
            )?,
            down: linear(cfg.intermediate_size, hidden, vb.pp("mlp").pp("down_layer"))?,
            norm_2: layer_norm(hidden, eps, vb.pp("layer_norm_2"))?,
        })
    }

    /// `bias` is the attention mask plus ALiBi, shape (batch, heads, seq, seq).
    fn forward(&self, xs: &Tensor, bias: &Tensor, heads: usize) -> candle_core::Result<Tensor> {
        let (batch, seq, hidden) = xs.dims3()?;
        let head_dim = hidden / heads;
        let split = |t: Tensor| {
            t.reshape((batch, seq, heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.norm_q.forward(&self.query.forward(xs)?)?)?;
        let k = split(self.norm_k.forward(&self.key.forward(xs)?)?)?;
        let v = split(self.value.forward(xs)?)?;
        let scores = (q.matmul(&k.t()?)? / (head_dim as f64).sqrt())?.broadcast_add(bias)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, seq, hidden))?;
        let attended = self
            .attn_norm
            .forward(&(self.attn_out.forward(&context)? + xs)?)?;
        let residual = self.norm_1.forward(&(xs + attended)?)?;
        let up_gated = self.up_gated.forward(&residual)?;
        let inner = up_gated.dim(D::Minus1)? / 2;
        let mlp = (up_gated.narrow(D::Minus1, 0, inner)?
            * up_gated.narrow(D::Minus1, inner, inner)?.gelu_erf()?)?;
        self.norm_2.forward(&(residual + self.down.forward(&mlp)?)?)
    }
}

/// JinaBERT v2 with QK post-norm (the `jina-embeddings-v2-base-code` layout).
#[derive(Debug)]
struct JinaCodeModel {
    word: Embedding,
    token_type: Tensor,
    embed_norm: LayerNorm,
    layers: Vec<Layer>,
    heads: usize,
}

impl JinaCodeModel {
    fn new(cfg: &Config, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        let emb = vb.pp("embeddings");
        let token_types = embedding(
            cfg.type_vocab_size,
            cfg.hidden_size,
            emb.pp("token_type_embeddings"),
        )?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| Layer::new(cfg, vb.pp(format!("encoder.layer.{index}"))))
            .collect::<candle_core::Result<_>>()?;
        Ok(Self {
            word: embedding(cfg.vocab_size, cfg.hidden_size, emb.pp("word_embeddings"))?,
            token_type: token_types.embeddings().get(0)?,
            embed_norm: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, emb.pp("LayerNorm"))?,
            layers,
            heads: cfg.num_attention_heads,
        })
    }

    /// Mean-pooled, L2-normalised sentence embeddings, shape (batch, hidden).
    fn embed(&self, ids: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let (_, seq) = ids.dims2()?;
        let mut xs = self
            .embed_norm
            .forward(&self.word.forward(ids)?.broadcast_add(&self.token_type)?)?;
        let mask_f = mask.to_dtype(DType::F32)?;
        let pad_bias = ((mask_f.clone() - 1.0)? * 1e9)?
            .unsqueeze(1)?
            .unsqueeze(1)?;
        let bias = pad_bias.broadcast_add(&alibi(self.heads, seq, ids.device())?)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &bias, self.heads)?;
        }
        let weights = mask_f.unsqueeze(2)?;
        let summed = xs.broadcast_mul(&weights)?.sum(1)?;
        let pooled = summed.broadcast_div(&weights.sum(1)?.clamp(1e-9, f64::MAX)?)?;
        let norm = pooled
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::MAX)?;
        pooled.broadcast_div(&norm)
    }
}

/// ALiBi slopes as in the reference `rebuild_alibi_tensor`, for any head count.
fn alibi_slopes(heads: usize) -> Vec<f64> {
    let power_of_2 = |n: usize| -> Vec<f64> {
        let start = 2f64.powf(-(2f64.powf(-((n as f64).log2() - 3.0))));
        (0..n).map(|i| start * start.powi(i as i32)).collect()
    };
    if heads.is_power_of_two() {
        return power_of_2(heads);
    }
    let closest = 1usize << heads.ilog2();
    let mut slopes = power_of_2(closest);
    slopes.extend(
        alibi_slopes(2 * closest)
            .into_iter()
            .step_by(2)
            .take(heads - closest),
    );
    slopes
}

fn alibi(heads: usize, seq: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mut values = Vec::with_capacity(heads * seq * seq);
    for slope in alibi_slopes(heads) {
        for i in 0..seq {
            for j in 0..seq {
                values.push((-slope * i.abs_diff(j) as f64) as f32);
            }
        }
    }
    Tensor::from_vec(values, (1, heads, seq, seq), device)
}

/// A loaded model plus tokenizer; build it once per process.
#[derive(Debug)]
pub struct Reranker {
    model: JinaCodeModel,
    tokenizer: Tokenizer,
    /// Same tokenizer cut from the left: a query's last lines are nearest the cursor.
    query_tokenizer: Tokenizer,
    device: Device,
}

impl Reranker {
    /// Loads the model from the local cache; never downloads.
    pub fn load_cached() -> Result<Self> {
        let files = cached_model_files().ok_or_else(|| {
            Error::runtime(format!(
                "Rerank model {MODEL_ID} is not in the local Hugging Face cache. Run `lode rerank-model --download` once."
            ))
        })?;
        Self::load(&files[0], &files[1], &files[2])
    }

    pub fn load(config: &Path, tokenizer: &Path, weights: &Path) -> Result<Self> {
        let cfg: Config = serde_json::from_slice(&std::fs::read(config)?)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer).map_err(model_err)?;
        tokenizer.with_padding(Some(PaddingParams {
            pad_id: 1,
            pad_token: "<pad>".into(),
            ..PaddingParams::default()
        }));
        let mut query_tokenizer = tokenizer.clone();
        for (tok, direction) in [
            (&mut tokenizer, TruncationDirection::Right),
            (&mut query_tokenizer, TruncationDirection::Left),
        ] {
            tok.with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                direction,
                ..TruncationParams::default()
            }))
            .map_err(model_err)?;
        }
        let device = Device::Cpu;
        let vb =
            VarBuilder::from_buffered_safetensors(std::fs::read(weights)?, DType::F32, &device)
                .map_err(model_err)?;
        let model = JinaCodeModel::new(&cfg, vb).map_err(model_err)?;
        Ok(Self {
            model,
            tokenizer,
            query_tokenizer,
            device,
        })
    }

    /// Unit-length embeddings, one row per text; long texts keep their start.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_with(&self.tokenizer, texts)
    }

    fn embed_with(&self, tokenizer: &Tokenizer, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(model_err)?;
        let seq = encodings[0].get_ids().len();
        let flat = |pick: fn(&tokenizers::Encoding) -> &[u32]| -> Vec<u32> {
            encodings
                .iter()
                .flat_map(|e| pick(e).iter().copied())
                .collect()
        };
        let ids = Tensor::from_vec(flat(|e| e.get_ids()), (texts.len(), seq), &self.device)
            .map_err(model_err)?;
        let mask = Tensor::from_vec(
            flat(|e| e.get_attention_mask()),
            (texts.len(), seq),
            &self.device,
        )
        .map_err(model_err)?;
        self.model
            .embed(&ids, &mask)
            .and_then(|t| t.to_vec2())
            .map_err(model_err)
    }
}

impl Rerank for Reranker {
    fn rerank(
        &self,
        conn: &Connection,
        query: &str,
        mut hits: Vec<NodeRow>,
    ) -> Result<Vec<NodeRow>> {
        let depth = hits.len().min(RERANK_DEPTH);
        if depth < 2 {
            return Ok(hits);
        }
        let texts = hits[..depth]
            .iter()
            .map(|hit| hit_text(conn, hit))
            .collect::<Result<Vec<_>>>()?;
        let query_vec = self.embed_with(&self.query_tokenizer, &[query.to_owned()])?;
        let sims: Vec<f64> = self
            .embed(&texts)?
            .iter()
            .map(|v| cosine(&query_vec[0], v))
            .collect();
        let order = fuse(&sims);
        let mut head: Vec<Option<NodeRow>> = hits.drain(..depth).map(Some).collect();
        let mut out: Vec<NodeRow> = order.iter().filter_map(|&i| head[i].take()).collect();
        out.append(&mut hits);
        Ok(out)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

/// New order of the reranked head, given embedding similarities in lexical order.
fn fuse(sims: &[f64]) -> Vec<usize> {
    let mut by_sim: Vec<usize> = (0..sims.len()).collect();
    by_sim.sort_by(|&a, &b| sims[b].total_cmp(&sims[a]).then(a.cmp(&b)));
    let mut embed_rank = vec![0; sims.len()];
    for (rank, &index) in by_sim.iter().enumerate() {
        embed_rank[index] = rank;
    }
    let score = |i: usize| 1.0 / (RRF_K + i as f64) + EMBED_WEIGHT / (RRF_K + embed_rank[i] as f64);
    let mut order: Vec<usize> = (0..sims.len()).collect();
    order.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
    order
}

/// The hit's own source lines, prefixed with its path and qualified name.
fn hit_text(conn: &Connection, hit: &NodeRow) -> Result<String> {
    let root: Option<String> = conn
        .query_row(
            "SELECT root FROM repos WHERE id = ?",
            [&hit.repo_id],
            |row| row.get(0),
        )
        .optional()?;
    let source = root
        .and_then(|root| std::fs::read_to_string(Path::new(&root).join(&hit.path)).ok())
        .unwrap_or_default();
    let start = usize::try_from(hit.start_line.max(1) - 1).unwrap_or(0);
    let span = usize::try_from(hit.end_line - hit.start_line + 1)
        .unwrap_or(1)
        .clamp(1, MAX_LINES);
    let body: Vec<&str> = source.lines().skip(start).take(span).collect();
    Ok(format!("# {} {}\n{}", hit.path, hit.qname, body.join("\n")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alibi_slopes_match_reference_for_twelve_heads() {
        let slopes = alibi_slopes(12);
        let half = 0.5f64;
        let root = 2f64.powf(-0.5);
        let expected: Vec<f64> = (1..=8)
            .map(|i| half.powi(i))
            .chain([1, 3, 5, 7].map(|i| root.powi(i)))
            .collect();
        assert_eq!(slopes.len(), 12);
        for (got, want) in slopes.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-12, "{slopes:?}");
        }
    }

    #[test]
    fn fuse_keeps_order_when_signals_agree_and_lifts_strong_embedding_match() {
        assert_eq!(fuse(&[0.9, 0.8, 0.7]), vec![0, 1, 2]);
        let sims = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.99];
        let order = fuse(&sims);
        let position = |item: usize| order.iter().position(|&i| i == item).unwrap();
        assert_eq!(order[0], 0);
        assert!(position(9) < position(6), "{order:?}");
        assert!(position(5) < position(9), "{order:?}");
    }
}
