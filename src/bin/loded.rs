use std::path::PathBuf;

use clap::Parser;

/// Lode local HTTP daemon
#[derive(Debug, Parser)]
#[command(name = "loded")]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 7979)]
    port: u16,
    #[arg(long = "data-dir")]
    data_dir: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    let data_dir = args.data_dir.unwrap_or_else(lode::config::default_data_dir);
    if let Err(err) = lode::daemon::serve(&args.host, args.port, &data_dir) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
