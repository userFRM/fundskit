//! Stateful `Fundskit` client — async 13F-holdings endpoints with blocking
//! wrappers.
//!
//! Fetches per-quarter parquet shards from GitHub raw (or a configurable
//! origin) with ETag-aware caching, SHA-256 manifest verification, and CDN
//! mirror fallback. Falls back to stale cache on transient network failures.
//!
//! # Quick start — free functions
//!
//! ```no_run
//! use fundskit::holders_of;
//!
//! #[tokio::main]
//! async fn main() -> fundskit::Result<()> {
//!     // Which managers hold Apple (by CUSIP), largest position first.
//!     for h in holders_of("037833100").await?.iter().take(5) {
//!         println!("{} holds {} shares (${})", h.manager_name, h.shares, h.value_usd);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! # Client pattern (reuse across calls)
//!
//! ```no_run
//! use fundskit::Fundskit;
//!
//! #[tokio::main]
//! async fn main() -> fundskit::Result<()> {
//!     let client = Fundskit::new();
//!     let holdings = client.holdings_by_manager("BERKSHIRE HATHAWAY").await?;
//!     println!("{} positions", holdings.len());
//!     Ok(())
//! }
//! ```

use std::path::PathBuf;

use crate::error::{Error, Result};
use crate::fetcher::{default_cache_dir, resolved_base_url, CachedFetcher};
use crate::parquet_io::read_holdings;
use crate::record::Holding;

/// Stateful fundskit client.
///
/// Wraps an ETag-aware cached fetcher and exposes flat async query methods.
/// Create once and reuse; the internal reqwest client is kept alive for
/// connection pooling.
///
/// ```no_run
/// use fundskit::Fundskit;
/// use std::path::PathBuf;
///
/// let client = Fundskit::new()
///     .with_base_url("https://my-mirror.example.com/fundskit")
///     .with_cache_dir(PathBuf::from("/tmp/fundskit-test"));
/// ```
#[derive(Clone)]
pub struct Fundskit {
    fetcher: CachedFetcher,
}

impl Fundskit {
    /// Create a client with the default GitHub raw backend and XDG cache.
    ///
    /// Reads `FUNDSKIT_BASE_URL` and `FUNDSKIT_CACHE_DIR` from the environment
    /// if set. **This function never fails.** Errors are deferred to the first
    /// fetch.
    pub fn new() -> Self {
        let http = reqwest::Client::builder()
            .user_agent("fundskit/0.1 (+https://github.com/userFRM/fundskit)")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            fetcher: CachedFetcher::new(http, resolved_base_url(), default_cache_dir()),
        }
    }

    /// Override the origin URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.fetcher.set_base_url(url.into());
        self
    }

    /// Override the on-disk cache directory.
    pub fn with_cache_dir(mut self, dir: PathBuf) -> Self {
        self.fetcher.set_cache_dir(dir);
        self
    }

    /// Override the CDN mirror URL. `None` disables mirror fallback.
    pub fn with_mirror_url(mut self, url: Option<String>) -> Self {
        self.fetcher.set_mirror_url(url);
        self
    }

    // ── Async query endpoints ───────────────────────────────────────────────

    /// All holdings filed by a manager, matched by exact filer CIK if
    /// `name_or_cik` parses as an integer, otherwise by case-insensitive
    /// substring of the manager name. Most recent report period first, then
    /// largest position first within a period.
    pub async fn holdings_by_manager(&self, name_or_cik: &str) -> Result<Vec<Holding>> {
        let rows = self.load_all_rows().await?;
        let matched: Vec<Holding> = if let Ok(cik) = name_or_cik.parse::<u32>() {
            rows.into_iter().filter(|r| r.manager_cik == cik).collect()
        } else {
            let needle = name_or_cik.to_lowercase();
            rows.into_iter()
                .filter(|r| r.manager_name.to_lowercase().contains(&needle))
                .collect()
        };
        Ok(sort_period_value(matched))
    }

    /// Every manager holding a security, matched by exact CUSIP (9 chars) or by
    /// case-insensitive substring of the issuer name. Most recent report period
    /// first, then largest position first.
    pub async fn holders_of(&self, cusip_or_issuer: &str) -> Result<Vec<Holding>> {
        let rows = self.load_all_rows().await?;
        let q = cusip_or_issuer.trim();
        let matched: Vec<Holding> = if is_cusip(q) {
            let cu = q.to_uppercase();
            rows.into_iter().filter(|r| r.cusip == cu).collect()
        } else {
            let needle = q.to_lowercase();
            rows.into_iter()
                .filter(|r| r.issuer_name.to_lowercase().contains(&needle))
                .collect()
        };
        Ok(sort_period_value(matched))
    }

    /// The most recent report period present in the data (a `YYYYMMDD` integer),
    /// or `None` if there are no holdings.
    pub async fn latest_period(&self) -> Result<Option<i32>> {
        Ok(self
            .load_all_rows()
            .await?
            .iter()
            .map(|r| r.report_period)
            .max())
    }

    /// New positions a manager reported in `period` that were absent from its
    /// prior 13F. The comparison is by CUSIP: a holding is "new" when its CUSIP
    /// appears in `period` but in none of the manager's earlier periods. Largest
    /// new position first.
    pub async fn new_positions(&self, manager: &str, period: i32) -> Result<Vec<Holding>> {
        use std::collections::HashSet;
        let all = self.holdings_by_manager(manager).await?;
        let prior_cusips: HashSet<&str> = all
            .iter()
            .filter(|r| r.report_period < period)
            .map(|r| r.cusip.as_str())
            .collect();
        let mut out: Vec<Holding> = all
            .iter()
            .filter(|r| r.report_period == period && !prior_cusips.contains(r.cusip.as_str()))
            .cloned()
            .collect();
        out.sort_by_key(|r| std::cmp::Reverse(r.value_usd));
        Ok(out)
    }

    // ── Blocking wrappers ───────────────────────────────────────────────────

    /// Blocking variant of [`holdings_by_manager`](Self::holdings_by_manager).
    pub fn holdings_by_manager_blocking(&self, name_or_cik: &str) -> Result<Vec<Holding>> {
        let c = self.clone();
        let q = name_or_cik.to_owned();
        block(async move { c.holdings_by_manager(&q).await })
    }

    /// Blocking variant of [`holders_of`](Self::holders_of).
    pub fn holders_of_blocking(&self, cusip_or_issuer: &str) -> Result<Vec<Holding>> {
        let c = self.clone();
        let q = cusip_or_issuer.to_owned();
        block(async move { c.holders_of(&q).await })
    }

    /// Blocking variant of [`latest_period`](Self::latest_period).
    pub fn latest_period_blocking(&self) -> Result<Option<i32>> {
        let c = self.clone();
        block(async move { c.latest_period().await })
    }

    /// Blocking variant of [`new_positions`](Self::new_positions).
    pub fn new_positions_blocking(&self, manager: &str, period: i32) -> Result<Vec<Holding>> {
        let c = self.clone();
        let m = manager.to_owned();
        block(async move { c.new_positions(&m, period).await })
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    /// Fetch every quarter shard listed in the manifest and flat-concatenate
    /// the rows.
    pub(crate) async fn load_all_rows(&self) -> Result<Vec<Holding>> {
        let keys = self.discover_shards().await?;
        let mut all = Vec::new();
        for key in keys {
            let bytes = self.fetcher.fetch(&key).await?;
            all.extend(read_holdings(&bytes)?);
        }
        Ok(all)
    }

    /// Fetch `manifest.json` and return sorted shard keys (relative paths
    /// without the `.parquet` suffix, e.g. `period=2024Q1/fund13f-2024Q1`).
    async fn discover_shards(&self) -> Result<Vec<String>> {
        let url = format!("{}/manifest.json", self.fetcher.base_url);
        let resp = self
            .fetcher
            .http
            .get(&url)
            .send()
            .await
            .map_err(Error::Http)?;
        if !resp.status().is_success() {
            return Err(Error::Other(format!(
                "manifest.json: HTTP {} {}",
                resp.status().as_u16(),
                resp.status().canonical_reason().unwrap_or("")
            )));
        }
        let manifest: serde_json::Value = resp.json().await.map_err(Error::Http)?;
        let obj = manifest
            .as_object()
            .ok_or_else(|| Error::Other("manifest.json is not a JSON object".into()))?;
        let mut keys: Vec<String> = obj
            .keys()
            .filter(|k| k.ends_with(".parquet"))
            .map(|k| k.trim_end_matches(".parquet").to_string())
            .collect();
        keys.sort();
        Ok(keys)
    }
}

impl Default for Fundskit {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A CUSIP is 9 alphanumeric characters. Treat such a query as an exact CUSIP
/// match; anything else is an issuer-name substring.
fn is_cusip(s: &str) -> bool {
    s.len() == 9 && s.bytes().all(|b| b.is_ascii_alphanumeric())
}

/// Sort by report period descending, then value descending — newest filing
/// first, biggest position first.
fn sort_period_value(mut rows: Vec<Holding>) -> Vec<Holding> {
    rows.sort_by_key(|r| {
        (
            std::cmp::Reverse(r.report_period),
            std::cmp::Reverse(r.value_usd),
        )
    });
    rows
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// All holdings filed by a manager (name or CIK), one-shot client.
pub async fn holdings_by_manager(name_or_cik: &str) -> Result<Vec<Holding>> {
    Fundskit::new().holdings_by_manager(name_or_cik).await
}

/// Every manager holding a security (CUSIP or issuer name), one-shot client.
pub async fn holders_of(cusip_or_issuer: &str) -> Result<Vec<Holding>> {
    Fundskit::new().holders_of(cusip_or_issuer).await
}

/// The most recent report period in the data, one-shot client.
pub async fn latest_period() -> Result<Option<i32>> {
    Fundskit::new().latest_period().await
}

// ---------------------------------------------------------------------------
// Blocking helper
// ---------------------------------------------------------------------------

/// Drive a future to completion from any context (sync or async).
///
/// - Inside a tokio **multi-thread** runtime: `block_in_place` + `block_on`.
/// - Inside a **current-thread** runtime or no runtime: the future is driven on
///   a dedicated OS thread with its own runtime so the caller is not re-entered.
pub(crate) fn block<F, T>(fut: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| handle.block_on(fut))
        }
        _ => std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(Error::Io)
                .and_then(|rt| rt.block_on(fut))
        })
        .join()
        .expect("blocking thread panicked"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(manager: &str, cik: u32, cusip: &str, issuer: &str, period: i32, value: i64) -> Holding {
        Holding {
            report_period: period,
            filed_date: period + 44, // ~45 days after quarter end
            accession: format!("{cik}-{period}"),
            manager_cik: cik,
            manager_name: manager.into(),
            issuer_name: issuer.into(),
            cusip: cusip.into(),
            ticker: String::new(),
            title_of_class: "COM".into(),
            value_usd: value,
            shares: value / 100,
            share_type: "SH".into(),
            put_call: String::new(),
            discretion: "SOLE".into(),
        }
    }

    #[test]
    fn cusip_detection() {
        assert!(is_cusip("037833100"));
        assert!(is_cusip("92343V104"));
        assert!(!is_cusip("AAPL"));
        assert!(!is_cusip("03783310")); // 8 chars
        assert!(!is_cusip("Apple Inc"));
    }

    #[test]
    fn new_positions_excludes_carryover() {
        // Same client logic without I/O: feed rows through the filter directly.
        let rows = [
            // Q1: holds A and B.
            h("FUND", 1, "AAA111111", "Issuer A", 20240331, 1000),
            h("FUND", 1, "BBB222222", "Issuer B", 20240331, 2000),
            // Q2: still A, drops B, adds C (new) and D (new).
            h("FUND", 1, "AAA111111", "Issuer A", 20240630, 1100),
            h("FUND", 1, "CCC333333", "Issuer C", 20240630, 5000),
            h("FUND", 1, "DDD444444", "Issuer D", 20240630, 3000),
        ];
        use std::collections::HashSet;
        let period = 20240630;
        let prior: HashSet<&str> = rows
            .iter()
            .filter(|r| r.report_period < period)
            .map(|r| r.cusip.as_str())
            .collect();
        let mut newp: Vec<&Holding> = rows
            .iter()
            .filter(|r| r.report_period == period && !prior.contains(r.cusip.as_str()))
            .collect();
        newp.sort_by_key(|r| std::cmp::Reverse(r.value_usd));
        assert_eq!(newp.len(), 2, "C and D are new; A carried over");
        assert_eq!(newp[0].cusip, "CCC333333"); // biggest first
        assert_eq!(newp[1].cusip, "DDD444444");
    }

    #[test]
    fn sort_newest_then_largest() {
        let sorted = sort_period_value(vec![
            h("F", 1, "AAA111111", "A", 20240331, 100),
            h("F", 1, "BBB222222", "B", 20240630, 50),
            h("F", 1, "CCC333333", "C", 20240630, 200),
        ]);
        assert_eq!(sorted[0].report_period, 20240630);
        assert_eq!(sorted[0].value_usd, 200); // largest in newest period
        assert_eq!(sorted[2].report_period, 20240331);
    }
}
