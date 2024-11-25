mod error;
mod generate;

use std::sync::Arc;

pub use error::{Error, Result};

use reqwest::Client;
use url::Url;

#[derive(Debug, Clone)]
pub struct Ollama {
    base_url: Arc<Url>,
    client: Client,
}

impl Default for Ollama {
    fn default() -> Self {
        Self {
            base_url: Url::parse("http://localhost:11434/").unwrap().into(),
            client: Client::new(),
        }
    }
}

impl Ollama {
    pub fn new(base_url: Url) -> Self {
        assert!(
            base_url.as_str().ends_with('/'),
            "ollama url should ends with '/'"
        );

        Self {
            base_url: base_url.into(),
            client: Client::new(),
        }
    }
}
