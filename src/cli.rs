use clap::Parser;
use url::Url;

#[derive(Debug, Parser)]
pub struct Cli {
    #[arg(long, short)]
    pub model: String,

    #[arg(long)]
    pub url: Option<Url>,

    #[arg(long)]
    pub gpus: Option<usize>,

    pub prompts: Vec<String>,
}
