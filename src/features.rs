use crate::{Error, Result};

const FEATURE_DEFAULTS: &[(&str, bool)] = &[("metrics", true)];

/// Feature flag lookup: `LODE_FEATURE_<NAME>` overrides the built-in default.
pub fn enabled(name: &str) -> Result<bool> {
    let default = FEATURE_DEFAULTS
        .iter()
        .find(|(flag, _)| *flag == name)
        .map(|(_, value)| *value)
        .ok_or_else(|| Error::invalid(format!("unknown feature flag: {name}")))?;
    let env_name = format!("LODE_FEATURE_{}", name.to_uppercase().replace('-', "_"));
    Ok(match std::env::var(env_name) {
        Ok(raw) => matches!(
            raw.trim().to_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    })
}
