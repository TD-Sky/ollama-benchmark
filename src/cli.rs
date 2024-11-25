use clap::Parser;

#[derive(Debug, Parser)]
pub struct Cli {
    #[arg(long, short)]
    pub model: String,

    #[arg(short, default_value_t = 5)]
    pub n: usize,

    #[arg(short)]
    pub prompt: String,
}
