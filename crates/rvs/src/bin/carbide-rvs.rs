/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Rack Validation Service (RVS)
//!
//! External validation orchestrator for NICC. Bridges NICC with test
//! frameworks (Benchpress, MPI-based, SLURM-based, etc.) to perform
//! partition-aware rack validation.

use std::path::PathBuf;

use carbide_rvs::artifact;
use carbide_rvs::client;
use carbide_rvs::config::Config;
use carbide_rvs::ctx::RvsCtx;
use carbide_rvs::error::RvsError;
use carbide_rvs::partitions::Partitions;
use carbide_rvs::rack;
use carbide_rvs::scenario;
use carbide_rvs::validation;
use clap::Parser;
use forge_tls::client_config::ClientCert;
use rpc::forge_tls_client::{ApiConfig, ForgeClientConfig};
use tokio::io::AsyncWriteExt;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Parser)]
#[command(about = "Rack Validation Service")]
struct Cli {
    /// Path to TOML config file. Defaults and CARBIDE_RVS__* env vars apply if omitted.
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), RvsError> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    tracing_subscriber::registry()
        .with(logfmt::layer())
        .with(env_filter)
        .init();

    tracing::info!("carbide-rvs: Rack Validation Service starting");

    let cli = Cli::parse();

    // Load config: defaults -> optional TOML -> CARBIDE_RVS__* env vars
    let cfg = Config::load(cli.config.as_deref())?;
    tracing::info!(config = ?cfg, "config loaded");

    // Load all scenarios -- soft fail per file so a single bad config doesn't block others.
    let scenarios: Vec<scenario::Scenario> = cfg
        .scenario_config_paths
        .iter()
        .filter_map(|path| {
            match scenario::Scenario::load(std::path::Path::new(path)) {
                Ok(s) => {
                    tracing::info!(path, model = %s.rack.model, sot_release = %s.rack.sot_release, "scenario loaded");
                    Some(s)
                }
                Err(e) => {
                    tracing::warn!(path, error = %e, "scenario not loaded, skipping");
                    None
                }
            }
        })
        .collect();

    // Build NICC client from config
    let client_cert = ClientCert {
        cert_path: cfg.tls.identity_pemfile_path.clone(),
        key_path: cfg.tls.identity_keyfile_path.clone(),
    };
    let client_config = ForgeClientConfig::new(cfg.tls.root_cafile_path.clone(), Some(client_cert));
    let api_config = ApiConfig::new(&cfg.nicc.url, &client_config);
    let nicc = client::NiccClient::new(&api_config);

    let ctx = RvsCtx { nicc, scenarios, cfg, sot_override_path: None };

    // Liveness probe server
    let listen_addr = ctx.cfg.metrics_endpoint.to_string();
    tracing::info!(addr = %listen_addr, "starting liveness HTTP server");

    let listener = tokio::net::TcpListener::bind(ctx.cfg.metrics_endpoint).await?;

    // Run validation and liveness concurrently; a hard error from validation
    // exits the process.
    tokio::select! {
        result = run_validation(&ctx) => result?,
        () = serve_liveness(listener) => {},
    }

    Ok(())
}

async fn serve_liveness(listener: tokio::net::TcpListener) {
    loop {
        match listener.accept().await {
            Ok((mut stream, _addr)) => {
                tokio::spawn(async move {
                    // TODO[#416]: proper responses instead of this
                    let mut buf = [0u8; 1024];
                    let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await;
                    let body = "carbide-rvs: alive\n";
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
            Err(e) => {
                tracing::warn!(error = %e, "liveness: accept failed, retrying");
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }
}

// Rack validation high-level flow
async fn run_validation(ctx: &RvsCtx) -> Result<(), RvsError> {
    artifact::start_cache_server(ctx).await?;
    let interval = std::time::Duration::from_secs(ctx.cfg.poll_interval_secs);
    loop {
        let racks = rack::fetch_racks(&ctx.nicc).await?;
        artifact::process_artifacts(&racks, ctx).await?;
        let os_uri = ctx.scenarios.first().map(|s| s.os.uri.as_str()).unwrap_or("");
        for job in validation::plan(Partitions::try_from(racks)?, &ctx.nicc, os_uri).await? {
            let report = validation::validate_partition(job).await?;
            validation::submit_report(report).await?;
        }
        tracing::info!(ctx.cfg.poll_interval_secs, "validation: cycle complete, sleeping");
        tokio::time::sleep(interval).await;
    }
}
