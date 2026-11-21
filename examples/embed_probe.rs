//! Prints embeddings of argv texts as JSON; used to compare against the reference model.
fn main() -> lode::Result<()> {
    let texts: Vec<String> = std::env::args().skip(1).collect();
    let reranker = lode::rerank::Reranker::load_cached()?;
    println!("{}", serde_json::to_string(&reranker.embed(&texts)?)?);
    Ok(())
}
