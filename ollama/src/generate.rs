//! NOTE: In cuda model, the model won't be loaded into GPU memory when it's not enough

use std::time::Duration;

use bon::Builder;
use chrono::{DateTime, Utc};
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Deserializer, Serialize};
use smol_str::SmolStr;

use crate::{Ollama, Result};

impl Ollama {
    pub async fn generate_stream(
        &self,
        req: StreamRequest<'_>,
    ) -> Result<impl Stream<Item = Result<StreamChunk>>> {
        let url = self.base_url.join("api/generate").unwrap();
        let stream = self
            .client
            .post(url)
            .json(&req)
            .send()
            .await?
            .bytes_stream()
            .map(|res| {
                let bs = res?;
                let chunk: StreamChunk = serde_json::from_slice(&bs)?;
                Ok(chunk)
            });

        Ok(stream)
    }
}

#[derive(Debug, Serialize)]
pub struct GenerateOptions {
    pub num_batch: usize,
}

#[derive(Debug, Deserialize)]
pub struct StreamChunk {
    pub model: SmolStr,
    pub created_at: DateTime<Utc>,
    pub response: SmolStr,
    #[serde(flatten)]
    pub stats: Option<Stats>,
}

#[derive(Debug, Deserialize)]
pub struct Stats {
    pub context: Vec<usize>,
    #[serde(deserialize_with = "duration_from_nanos")]
    pub total_duration: Duration,
    #[serde(deserialize_with = "duration_from_nanos")]
    pub load_duration: Duration,
    pub prompt_eval_count: usize,
    #[serde(deserialize_with = "duration_from_nanos")]
    pub prompt_eval_duration: Duration,
    pub eval_count: usize,
    #[serde(deserialize_with = "duration_from_nanos")]
    pub eval_duration: Duration,
}

fn duration_from_nanos<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: Deserializer<'de>,
{
    let nanos = u64::deserialize(deserializer)?;
    Ok(Duration::from_nanos(nanos))
}

#[derive(Debug, Serialize, Builder)]
pub struct StreamRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    options: Option<&'a GenerateOptions>,
}
