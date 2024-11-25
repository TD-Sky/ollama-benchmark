mod cli;

use clap::Parser;
use cli::Cli;
use futures_util::{stream::FuturesUnordered, StreamExt};
use indoc::printdoc;
use ollama::{GenerateOptions, Ollama, StreamRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut cli = Cli::parse();

    let opts = cli.batch.map(|num_batch| GenerateOptions { num_batch });
    let ollama = cli
        .url
        .take()
        .map(Ollama::new)
        .unwrap_or_else(Ollama::default);

    let mut queue = FuturesUnordered::new();

    for _ in 0..cli.n {
        let req = StreamRequest::builder()
            .model(&cli.model)
            .prompt(&cli.prompt)
            .maybe_options(opts.as_ref())
            .build();

        queue.push(async {
            let mut stream = ollama.generate_stream(req).await?;

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

        let load_time = stats.load_duration.as_secs_f64();
        let load_speed = stats.prompt_eval_count as f64 / load_time;
        let eval_time = stats.eval_duration.as_secs_f64();
        let eval_speed = stats.eval_count as f64 / eval_time;

        printdoc! {"
            [benchmark for `{}`]
            prompt load time: {load_time}s
            prompt load speed: {load_speed} t/s
            evaluation time: {eval_time}s
            evaluation speed: {eval_speed} t/s

            ",
            cli.model
        };
    }

    Ok(())
}
