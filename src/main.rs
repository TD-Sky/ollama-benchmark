mod cli;

use clap::Parser;
use cli::Cli;
use futures_util::{stream::FuturesUnordered, StreamExt};
use ollama::Ollama;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut cli = Cli::parse();

    let ollama = cli
        .url
        .take()
        .map(Ollama::new)
        .unwrap_or_else(Ollama::default);

    let mut queue = FuturesUnordered::new();

    for _ in 0..cli.n {
        queue.push(async {
            let mut stream = ollama.generate_stream(&cli.model, &cli.prompt).await?;

            let mut stats = None;

            while let Some(chunk) = stream.next().await {
                let mut chunk = chunk?;
                stats = chunk.stats.take();
            }

            ollama::Result::<_>::Ok(stats.unwrap())
        });
    }

    while let Some(stats) = queue.next().await {
        let stats = stats?;
        let speed = stats.eval_count as f64 / stats.eval_duration.as_secs_f64();
        println!("evaluation speed: {speed} t/s");
    }

    Ok(())
}
