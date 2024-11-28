mod cli;

use std::rc::Rc;

use clap::Parser;
use cli::Cli;
use futures_util::{stream::FuturesUnordered, StreamExt};
use indoc::printdoc;
use ollama::{GenerateOptions, Ollama, Stats, StreamRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut cli = Cli::parse();

    let opts = GenerateOptions::builder().maybe_num_gpu(cli.gpus).build();
    let ollama = Rc::new(
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

    let mut stats_group = vec![];
    while let Some(res) = queue.next().await {
        let (i, answer, stats) = res?;

        let peval_time = stats.prompt_eval_duration.as_secs_f64();
        let peval_speed = stats.prompt_eval_count as f64 / peval_time;
        let eval_time = stats.eval_duration.as_secs_f64();
        let eval_speed = stats.eval_count as f64 / eval_time;

        printdoc! {"
            [benchmark {i}]
            prompt tokens: {prompt_tokens}
            prompt evaluation time: {peval_time}s
            prompt evaluaeval speed: {peval_speed} t/s
            answer tokens: {answer_tokens}
            evaluation time: {eval_time}s
            evaluation speed: {eval_speed} t/s
            ---
            {answer}
            ---

            ",
            prompt_tokens = stats.prompt_eval_count,
            answer_tokens = stats.eval_count,
        };

        stats_group.push(stats);
    }

    let MeanStats {
        prompt_eval_count,
        prompt_eval_secs,
        prompt_eval_speed,
        eval_count,
        eval_secs,
        eval_speed,
    } = MeanStats::new(&stats_group);
    printdoc! {"
        === Mean Stats ===
        prompt tokens: {prompt_tokens}
        prompt evaluation time: {prompt_eval_secs}s
        prompt evaluaeval speed: {prompt_eval_speed} t/s
        answer tokens: {answer_tokens}
        evaluation time: {eval_secs}s
        evaluation speed: {eval_speed} t/s
        ",
        prompt_tokens = prompt_eval_count,
        answer_tokens = eval_count,
    };

    Ok(())
}

#[derive(Debug)]
struct MeanStats {
    prompt_eval_count: f64,
    prompt_eval_secs: f64,
    prompt_eval_speed: f64,
    eval_count: f64,
    eval_secs: f64,
    eval_speed: f64,
}

impl MeanStats {
    fn new(stats_group: &[Stats]) -> Self {
        let n = stats_group.len() as f64;

        let total_prompt_eval_count = stats_group
            .iter()
            .map(|stats| stats.prompt_eval_count)
            .sum::<usize>() as f64;
        let total_prompt_eval_secs = stats_group
            .iter()
            .map(|stats| stats.prompt_eval_duration.as_secs_f64())
            .sum::<f64>();
        let total_eval_count = stats_group
            .iter()
            .map(|stats| stats.eval_count)
            .sum::<usize>() as f64;
        let total_eval_secs = stats_group
            .iter()
            .map(|stats| stats.eval_duration.as_secs_f64())
            .sum::<f64>();

        Self {
            prompt_eval_count: total_prompt_eval_count / n,
            prompt_eval_secs: total_prompt_eval_secs / n,
            prompt_eval_speed: total_prompt_eval_count / total_prompt_eval_secs,
            eval_count: total_eval_count / n,
            eval_secs: total_eval_secs / n,
            eval_speed: total_eval_count / total_eval_secs,
        }
    }
}
