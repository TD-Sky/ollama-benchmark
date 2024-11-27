mod cli;

use std::sync::Arc;

use clap::Parser;
use cli::Cli;
use futures_util::{stream::FuturesUnordered, StreamExt};
use indoc::printdoc;
use ollama::{GenerateOptions, Ollama, StreamRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut cli = Cli::parse();

    let opts = GenerateOptions::builder().maybe_num_gpu(cli.gpus).build();
    let ollama = Arc::new(
        cli.url
            .take()
            .map(Ollama::new)
            .unwrap_or_else(Ollama::default),
    );

    let mut queue = FuturesUnordered::new();

    for (i, prompt) in cli.prompts.iter().enumerate() {
        let req = StreamRequest::builder()
            .model(&cli.model)
            .prompt(prompt)
            .options(&opts)
            .build();

        let ollama = ollama.clone();

        queue.push(async move {
            let mut stream = ollama.generate_stream(req).await?;

            let mut stats = None;

            let mut answer = String::new();
            while let Some(chunk) = stream.next().await {
                let mut chunk = chunk?;
                answer.push_str(&chunk.response);
                stats = chunk.stats.take();
            }

            ollama::Result::<_>::Ok((i, answer, stats.unwrap()))
        });
    }

    while let Some(res) = queue.next().await {
        let (i, answer, stats) = res?;

        let peval_time = stats.prompt_eval_duration.as_secs_f64();
        let peval_speed = stats.prompt_eval_count as f64 / peval_time;
        let eval_time = stats.eval_duration.as_secs_f64();
        let eval_speed = stats.eval_count as f64 / eval_time;

        printdoc! {"
            [benchmark {i}]
            prompt tokens: {}
            prompt evaluation time: {peval_time}s
            prompt evaluaeval speed: {peval_speed} t/s
            answer tokens: {}
            evaluation time: {eval_time}s
            evaluation speed: {eval_speed} t/s
            ---
            {answer}
            ---

            ",
            stats.prompt_eval_count,
            stats.eval_count,
        };
    }

    Ok(())
}
