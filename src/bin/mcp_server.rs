//! MCP (Model Context Protocol) server for the twopoint + pt libraries.
//!
//! Exposes both kNN measurement tools and perturbation theory predictions
//! as MCP tools over JSON-RPC/stdio, suitable for use by LLMs.
//!
//! Run: cargo run --bin twopoint-mcp --features theory

use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ── Measurement imports ─────────────────────────────────────────────────
use twopoint::estimator::{
    cdf_k_values, cdf_r_grid, exclude_self_pairs, linear_bins,
    LandySzalayKnn,
};
use twopoint::estimator::cumulants::jacobian_cumulants_multi_k;
use twopoint::diagnostics::{erlang_cdf, erlang_pdf};
use twopoint::ladder::DilutionLadder;
use twopoint::mock::{CoxMock, CoxMockParams};
use twopoint::tree::PointTree;
use twopoint::validation::{run_single_mock, aggregate_mocks, ValidationConfig};

// ── Theory imports ──────────────────────────────────────────────────────
use twopoint::theory::{self, *};

// ═══════════════════════════════════════════════════════════════════════════
// JSON-RPC / MCP protocol types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

// ═══════════════════════════════════════════════════════════════════════════
// Server state
// ═══════════════════════════════════════════════════════════════════════════

struct ServerState {
    workspace: Option<Workspace>,
    cosmology: Option<Cosmology>,
}

impl ServerState {
    fn new() -> Self {
        ServerState {
            workspace: None,
            cosmology: None,
        }
    }

    #[allow(dead_code)]
    fn ensure_workspace(&mut self, cosmo: &Cosmology) -> &Workspace {
        if self.workspace.is_none() {
            let mut ws = Workspace::new(2000);
            ws.update_cosmology(cosmo);
            self.workspace = Some(ws);
        }
        self.workspace.as_ref().unwrap()
    }

    fn set_cosmology(&mut self, cosmo: Cosmology) {
        if let Some(ref mut ws) = self.workspace {
            ws.update_cosmology(&cosmo);
        } else {
            let mut ws = Workspace::new(2000);
            ws.update_cosmology(&cosmo);
            self.workspace = Some(ws);
        }
        self.cosmology = Some(cosmo);
    }

    fn get_cosmology(&self) -> Result<&Cosmology, String> {
        self.cosmology.as_ref().ok_or_else(|| {
            "No cosmology set. Call set_cosmology first.".to_string()
        })
    }

    fn get_workspace(&self) -> Result<&Workspace, String> {
        self.workspace.as_ref().ok_or_else(|| {
            "No workspace initialized. Call set_cosmology first.".to_string()
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tool definitions
// ═══════════════════════════════════════════════════════════════════════════

fn tool_definitions() -> Value {
    json!([
        // ── Cosmology setup ─────────────────────────────────────────
        {
            "name": "set_cosmology",
            "description": "Set cosmological parameters and initialize the power spectrum workspace. Must be called before any theory computation. Use preset='planck2018' for default Planck 2018 parameters, or provide custom values.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "preset": {"type": "string", "description": "Preset name: 'planck2018'", "enum": ["planck2018"]},
                    "omega_m": {"type": "number", "description": "Matter density parameter Omega_m"},
                    "omega_b": {"type": "number", "description": "Baryon density parameter Omega_b"},
                    "h": {"type": "number", "description": "Hubble parameter h = H0/100"},
                    "n_s": {"type": "number", "description": "Scalar spectral index"},
                    "sigma8": {"type": "number", "description": "sigma_8 normalization"},
                    "m_nu": {"type": "number", "description": "Sum of neutrino masses [eV] (default: 0.06)"},
                    "w0": {"type": "number", "description": "Dark energy EOS w0 (default: -1.0)"},
                    "wa": {"type": "number", "description": "Dark energy EOS wa (default: 0.0)"}
                }
            }
        },
        // ── Theory: perturbation theory predictions ─────────────────
        {
            "name": "sigma2j_detailed",
            "description": "Compute full perturbation theory predictions at a smoothing radius R. Returns sigma2_lin, sigma2_zel, sigma2_j, xi_bar, s3_matter, s3_jacobian, and loop decomposition. Requires set_cosmology first.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "r": {"type": "number", "description": "Smoothing radius R [Mpc/h]"},
                    "n_lpt": {"type": "integer", "description": "LPT order (0-3, default: 2)", "default": 2},
                    "c_j2": {"type": "number", "description": "EFT counterterm c_J^2 (default: 0)", "default": 0},
                    "c_j4": {"type": "number", "description": "EFT counterterm c_J^4 (default: 0)", "default": 0},
                    "compute_bispec": {"type": "boolean", "description": "Compute bias bispectrum integrals (slower)", "default": false}
                },
                "required": ["r"]
            }
        },
        {
            "name": "sigma2j_at_radii",
            "description": "Batch compute sigma2_J at multiple radii. Fast path using pre-built workspace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "radii": {"type": "array", "items": {"type": "number"}, "description": "Radii R [Mpc/h]"},
                    "n_lpt": {"type": "integer", "default": 2}
                },
                "required": ["radii"]
            }
        },
        {
            "name": "sigma2j_plot",
            "description": "Compute sigma2_J on a fine grid of 50 radii (3-200 Mpc/h) for plotting. By default returns only the fast fields (sigma2_j, sigma2_zel, sigma2_lin, xi_bar, d1-d3, counterterm). Set diagnostics=true to also compute and return p22, two_p13, s3_matter, s3_jacobian (~15 ms/R extra for 3D integrals; sigma2_j itself is unaffected).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_lpt": {"type": "integer", "default": 2},
                    "diagnostics": {"type": "boolean", "default": false, "description": "If true, compute the 3D P22/P13/S3 diagnostic integrals (~10x slower). Only enable if you will consume those fields."}
                }
            }
        },
        {
            "name": "linear_power_spectrum",
            "description": "Evaluate the linear matter power spectrum P_L(k) at given wavenumbers.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "k_values": {"type": "array", "items": {"type": "number"}, "description": "Wavenumbers k [h/Mpc]"}
                },
                "required": ["k_values"]
            }
        },
        {
            "name": "xi_bar",
            "description": "Compute the volume-averaged correlation function xi_bar(R) at given radii.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "radii": {"type": "array", "items": {"type": "number"}, "description": "Smoothing radii R [Mpc/h]"}
                },
                "required": ["radii"]
            }
        },
        {
            "name": "sigma2_linear",
            "description": "Compute tree-level sigma^2_lin(R) at given radii.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "radii": {"type": "array", "items": {"type": "number"}, "description": "Smoothing radii R [Mpc/h]"}
                },
                "required": ["radii"]
            }
        },
        {
            "name": "skewness",
            "description": "Compute tree-level skewness S3(R) for matter and Jacobian at given radii.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "radii": {"type": "array", "items": {"type": "number"}, "description": "Smoothing radii R [Mpc/h]"}
                },
                "required": ["radii"]
            }
        },
        {
            "name": "bias_integrals",
            "description": "Compute all 6 bias-operator bispectrum integrals at a radius: I_F2, I_F2J, I_delta2, I_s2, I_nabla, I_cs2.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "r": {"type": "number", "description": "Smoothing radius R [Mpc/h]"}
                },
                "required": ["r"]
            }
        },
        {
            "name": "doroshkevich_moments",
            "description": "Compute exact Doroshkevich (Zel'dovich Jacobian) moments at given sigma: mean_J, Var(J), s3, s4.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sigma": {"type": "number", "description": "Linear sigma = sqrt(sigma2_lin)"},
                    "n_gauss": {"type": "integer", "description": "Quadrature points per dim (default: 48)", "default": 48}
                },
                "required": ["sigma"]
            }
        },
        {
            "name": "doroshkevich_cdf",
            "description": "Compute the CDF of the Zel'dovich Jacobian P(J < j0 | sigma) at given thresholds. Optionally with linear bias b1 for biased tracers.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sigma": {"type": "number", "description": "Linear sigma"},
                    "j_thresholds": {"type": "array", "items": {"type": "number"}, "description": "Jacobian thresholds (sorted ascending)"},
                    "b1": {"type": "number", "description": "Linear bias (default: 0, unbiased)", "default": 0},
                    "n_gauss": {"type": "integer", "default": 48}
                },
                "required": ["sigma", "j_thresholds"]
            }
        },
        // ── Theory: kNN CDF predictions ─────────────────────────────
        {
            "name": "rknn_cdf_predict",
            "description": "Predict R-kNN CDF (random query points) from perturbation theory at given k values and radii. Zero free parameters — all from P_L(k).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "k_values": {"type": "array", "items": {"type": "integer"}, "description": "Neighbour ranks k (e.g. [1, 2, 4, 8])"},
                    "nbar": {"type": "number", "description": "Reference number density [h^3/Mpc^3]"},
                    "r_values": {"type": "array", "items": {"type": "number"}, "description": "Query radii [Mpc/h]"},
                    "n_lpt": {"type": "integer", "default": 2},
                    "with_poisson": {"type": "boolean", "description": "Include Poisson discreteness correction (important for k<10)", "default": false}
                },
                "required": ["k_values", "nbar", "r_values"]
            }
        },
        {
            "name": "dknn_cdf_predict",
            "description": "Predict D-kNN CDF for biased tracers (galaxy query points). Includes bias shift in the Jacobian distribution.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "k_values": {"type": "array", "items": {"type": "integer"}, "description": "Neighbour ranks k"},
                    "nbar_ref": {"type": "number", "description": "Reference sample density [h^3/Mpc^3]"},
                    "b1": {"type": "number", "description": "Linear bias of tracers"},
                    "r_values": {"type": "array", "items": {"type": "number"}, "description": "Query radii [Mpc/h]"},
                    "n_lpt": {"type": "integer", "default": 2}
                },
                "required": ["k_values", "nbar_ref", "b1", "r_values"]
            }
        },
        // ── Geometry / conversions ──────────────────────────────────
        {
            "name": "geometry",
            "description": "Convert between mass M [Msun/h], Lagrangian radius R [Mpc/h], effective wavenumber k_eff [h/Mpc], and kNN neighbour rank k at density nbar.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "from": {"type": "string", "enum": ["mass", "radius", "k_eff", "knn"], "description": "Input coordinate"},
                    "values": {"type": "array", "items": {"type": "number"}, "description": "Input values"},
                    "nbar": {"type": "number", "description": "Number density (required for knn conversion)"},
                    "omega_m": {"type": "number", "description": "Omega_m (required for mass<->radius, uses current cosmology if not set)"}
                },
                "required": ["from", "values"]
            }
        },
        // ── Measurement: mock generation ────────────────────────────
        {
            "name": "generate_mock",
            "description": "Generate a CoxMock (line-point Cox process) catalog for testing. Returns positions and catalog info. Use preset='validation' for standard test, or specify parameters.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "preset": {"type": "string", "enum": ["euclid_small", "tiny", "validation", "poisson"]},
                    "box_size": {"type": "number", "description": "Box side length [Mpc/h]"},
                    "n_lines": {"type": "integer", "description": "Number of lines"},
                    "line_length": {"type": "number", "description": "Line length [Mpc/h]"},
                    "n_points": {"type": "integer", "description": "Number of points"},
                    "seed": {"type": "integer", "description": "RNG seed (default: 42)", "default": 42},
                    "n_randoms": {"type": "integer", "description": "Number of random points to generate (default: 5x data)"}
                }
            }
        },
        // ── Measurement: xi estimation ──────────────────────────────
        {
            "name": "estimate_xi",
            "description": "Estimate the two-point correlation function xi(r) from a point catalog in a periodic box using kNN pair-count densities (Landy-Szalay estimator).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Data point positions [[x,y,z], ...]"},
                    "random_points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Random point positions (optional; auto-generated if omitted)"},
                    "n_randoms": {"type": "integer", "description": "Number of randoms to generate if random_points not provided", "default": 50000},
                    "box_size": {"type": "number", "description": "Periodic box side length [Mpc/h]"},
                    "k_max": {"type": "integer", "description": "Maximum neighbors per query (default: 64)", "default": 64},
                    "n_bins": {"type": "integer", "description": "Number of radial bins", "default": 30},
                    "r_min": {"type": "number", "description": "Minimum radius [Mpc/h]", "default": 5.0},
                    "r_max": {"type": "number", "description": "Maximum radius [Mpc/h]", "default": 200.0}
                },
                "required": ["data_points", "box_size"]
            }
        },
        // ── Measurement: kNN CDFs ───────────────────────────────────
        {
            "name": "empirical_cdfs",
            "description": "Compute empirical kNN CDFs from a point catalog. Returns CDF(r) for each k value.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    "box_size": {"type": "number"},
                    "k_values": {"type": "array", "items": {"type": "integer"}, "description": "k values (default: powers of 2 up to k_max)"},
                    "k_max": {"type": "integer", "default": 64},
                    "n_r": {"type": "integer", "description": "Number of r-grid points", "default": 150},
                    "r_min": {"type": "number", "default": 1.0},
                    "r_max": {"type": "number", "default": 200.0}
                },
                "required": ["data_points", "box_size"]
            }
        },
        // ── Measurement: cumulants ──────────────────────────────────
        {
            "name": "count_cumulants",
            "description": "Measure cell-count cumulants xi_bar(R), sigma2(R), S3(R), S4(R) from a catalog using per-point kNN profiles. These are the measurement counterparts of sigma2_J and S3 from perturbation theory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    "box_size": {"type": "number"},
                    "k_max": {"type": "integer", "default": 64},
                    "max_dilution_level": {"type": "integer", "default": 2},
                    "n_r": {"type": "integer", "description": "Number of radial grid points", "default": 30},
                    "r_min": {"type": "number", "default": 5.0},
                    "r_max": {"type": "number", "default": 200.0}
                },
                "required": ["data_points", "box_size"]
            }
        },
        {
            "name": "jacobian_cumulants",
            "description": "Measure Jacobian J = (r_k/R_L)^3 cumulants from kNN distances at multiple k values. Returns mean_J, Var(J), s3, s4 as functions of smoothing scale R(k).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data_points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    "box_size": {"type": "number"},
                    "k_values": {"type": "array", "items": {"type": "integer"}, "description": "Neighbour ranks (default: [1,2,4,8,16,32,64])"},
                    "k_max": {"type": "integer", "default": 64}
                },
                "required": ["data_points", "box_size"]
            }
        },
        // ── Measurement: validation pipeline ────────────────────────
        {
            "name": "run_validation",
            "description": "Run the full CoxMock validation pipeline: generate mocks, estimate xi(r) and kNN-CDFs, compare to analytic truth. Returns chi2, mean/std of xi, and CDF summaries.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_mocks": {"type": "integer", "default": 10},
                    "preset": {"type": "string", "enum": ["euclid_small", "tiny", "validation", "poisson"], "default": "validation"},
                    "k_max": {"type": "integer", "default": 8},
                    "n_bins": {"type": "integer", "default": 40},
                    "r_min": {"type": "number", "default": 5.0},
                    "r_max": {"type": "number", "default": 250.0},
                    "random_ratio": {"type": "integer", "default": 5},
                    "max_dilution_level": {"type": "integer", "default": 2}
                }
            }
        },
        // ── Diagnostics ─────────────────────────────────────────────
        {
            "name": "erlang_cdf",
            "description": "Compute the Poisson (Erlang) CDF for the k-th nearest neighbor at distance r given number density nbar. This is the null hypothesis (no clustering) baseline.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "k": {"type": "integer"},
                    "r_values": {"type": "array", "items": {"type": "number"}},
                    "nbar": {"type": "number"}
                },
                "required": ["k", "r_values", "nbar"]
            }
        },
        {
            "name": "xibar_j_full",
            "description": "Compute the full three-layer ξ̄_J prediction at given radii for a tracer with linear bias b1. Returns tree-level, exact Zel'dovich, one-loop, and full prediction at each radius. Supports optional redshift-space distortions via growth rate f: σ²_s = K₂ σ²_L drives Doroshkevich and ε, K₁ multiplies the tree-level ξ̄ and one-loop P₁₃.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "radii": {"type": "array", "items": {"type": "number"}, "description": "Smoothing radii R [Mpc/h]"},
                    "b1": {"type": "number", "description": "Linear bias of tracer population"},
                    "n_corrections": {"type": "integer", "default": 3, "description": "Number of geometric series terms (0=Zel only, 1=+1loop, 2+=higher)"},
                    "f_growth": {"type": "number", "default": 0.0, "description": "Growth rate f (0 = real space; z=0 Planck has f≈0.525). Enables RSD when >0."}
                },
                "required": ["radii", "b1"]
            }
        },
        {
            "name": "theory_at_knn_scale",
            "description": "Look up all PT predictions at the smoothing scale corresponding to k-th nearest neighbor at density nbar. Returns sigma2_lin, sigma2_zel, sigma2_j, xi_bar, S3 at R(k,nbar).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "k_values": {"type": "array", "items": {"type": "integer"}, "description": "Neighbour ranks"},
                    "nbar": {"type": "number", "description": "Number density [h^3/Mpc^3]"}
                },
                "required": ["k_values", "nbar"]
            }
        }
    ])
}

// ═══════════════════════════════════════════════════════════════════════════
// Tool dispatch
// ═══════════════════════════════════════════════════════════════════════════

fn handle_tool_call(state: &mut ServerState, name: &str, args: &Value) -> Result<Value, String> {
    match name {
        "set_cosmology" => handle_set_cosmology(state, args),
        "sigma2j_detailed" => handle_sigma2j_detailed(state, args),
        "sigma2j_at_radii" => handle_sigma2j_at_radii(state, args),
        "sigma2j_plot" => handle_sigma2j_plot(state, args),
        "linear_power_spectrum" => handle_linear_power_spectrum(state, args),
        "xi_bar" => handle_xi_bar(state, args),
        "sigma2_linear" => handle_sigma2_linear(state, args),
        "skewness" => handle_skewness(state, args),
        "bias_integrals" => handle_bias_integrals(state, args),
        "doroshkevich_moments" => handle_doroshkevich_moments(args),
        "doroshkevich_cdf" => handle_doroshkevich_cdf(args),
        "rknn_cdf_predict" => handle_rknn_cdf_predict(state, args),
        "dknn_cdf_predict" => handle_dknn_cdf_predict(state, args),
        "geometry" => handle_geometry(state, args),
        "generate_mock" => handle_generate_mock(args),
        "estimate_xi" => handle_estimate_xi(args),
        "empirical_cdfs" => handle_empirical_cdfs(args),
        "count_cumulants" => handle_count_cumulants(args),
        "jacobian_cumulants" => handle_jacobian_cumulants(args),
        "run_validation" => handle_run_validation(args),
        "erlang_cdf" => handle_erlang_cdf(args),
        "xibar_j_full" => handle_xibar_j_full(state, args),
        "theory_at_knn_scale" => handle_theory_at_knn_scale(state, args),
        _ => Err(format!("Unknown tool: {}", name)),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tool handlers
// ═══════════════════════════════════════════════════════════════════════════

fn handle_set_cosmology(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let cosmo = if args.get("preset").and_then(|v| v.as_str()) == Some("planck2018") {
        Cosmology::planck2018()
    } else {
        let om = args["omega_m"].as_f64().ok_or("omega_m required")?;
        let ob = args["omega_b"].as_f64().ok_or("omega_b required")?;
        let h = args["h"].as_f64().ok_or("h required")?;
        let ns = args["n_s"].as_f64().ok_or("n_s required")?;
        let s8 = args["sigma8"].as_f64().ok_or("sigma8 required")?;
        let mnu = args.get("m_nu").and_then(|v| v.as_f64()).unwrap_or(0.06);
        let w0 = args.get("w0").and_then(|v| v.as_f64()).unwrap_or(-1.0);
        let wa = args.get("wa").and_then(|v| v.as_f64()).unwrap_or(0.0);
        Cosmology::with_extensions(om, ob, h, ns, s8, mnu, w0, wa)
    };
    state.set_cosmology(cosmo);
    let c = state.get_cosmology().unwrap();
    Ok(json!({
        "status": "ok",
        "omega_m": c.omega_m,
        "omega_b": c.omega_b,
        "h": c.h,
        "n_s": c.n_s,
        "sigma8": c.sigma8,
        "m_nu": c.m_nu,
        "w0": c.w0,
        "wa": c.wa
    }))
}

fn handle_sigma2j_detailed(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let cosmo = state.get_cosmology()?.clone();
    let r = args["r"].as_f64().ok_or("r required")?;
    let n_lpt = args.get("n_lpt").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
    let c_j2 = args.get("c_j2").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let c_j4 = args.get("c_j4").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let compute_bispec = args.get("compute_bispec").and_then(|v| v.as_bool()).unwrap_or(false);

    // sigma2j_detailed: user explicitly asked for "detailed" → diagnostics ON.
    let result = pt::sigma2_j_full(&cosmo, r, n_lpt, c_j2, c_j4, true, compute_bispec);
    let mut out = json!({
        "r": result.r,
        "mass": result.mass,
        "k_eff": result.k_eff,
        "sigma2_lin": result.sigma2_lin,
        "sigma2_zel": result.sigma2_zel,
        "sigma2_j": result.sigma2_j,
        "xi_bar": result.xi_bar,
        "d1": result.d1,
        "d2": result.d2,
        "d3": result.d3,
        "p22": result.p22,
        "two_p13": result.two_p13,
        "counterterm": result.counterterm,
        "s3_matter": result.s3_matter,
        "s3_jacobian": result.s3_jacobian,
        "truncation_error": result.truncation_error,
        "elapsed_us": result.elapsed_ns / 1000
    });
    if let Some(bi) = &result.bispec {
        out["bispec"] = json!({
            "i_f2": bi.i_f2, "i_f2j": bi.i_f2j, "i_delta2": bi.i_delta2,
            "i_s2": bi.i_s2, "i_nabla": bi.i_nabla, "i_cs2": bi.i_cs2
        });
    }
    Ok(out)
}

fn handle_sigma2j_at_radii(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let cosmo = state.get_cosmology()?.clone();
    let ws = state.get_workspace()?;
    let radii: Vec<f64> = serde_json::from_value(args["radii"].clone())
        .map_err(|e| format!("radii: {}", e))?;
    let n_lpt = args.get("n_lpt").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
    let mut out = vec![0.0; radii.len()];
    pt::sigma2_j_at_radii(&cosmo, &radii, n_lpt, 0.0, 0.0, ws, &mut out);
    Ok(json!({ "radii": radii, "sigma2_j": out }))
}

fn handle_sigma2j_plot(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let cosmo = state.get_cosmology()?.clone();
    let n_lpt = args.get("n_lpt").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
    // Caller opts in to the expensive 3D diagnostics. Default is off — the
    // plot path is ~10× faster without them and σ²_J itself is identical.
    let diagnostics = args.get("diagnostics").and_then(|v| v.as_bool()).unwrap_or(false);
    let results = pt::sigma2_j_plot(&cosmo, n_lpt, 0.0, 0.0, diagnostics);
    let out: Vec<Value> = results.iter().map(|r| {
        let mut obj = json!({
            "r": r.r, "mass": r.mass, "k_eff": r.k_eff,
            "sigma2_lin": r.sigma2_lin, "sigma2_zel": r.sigma2_zel, "sigma2_j": r.sigma2_j,
            "d1": r.d1, "d2": r.d2, "d3": r.d3,
            "counterterm": r.counterterm,
            "xi_bar": r.xi_bar
        });
        if diagnostics {
            obj["p22"] = json!(r.p22);
            obj["two_p13"] = json!(r.two_p13);
            obj["s3_matter"] = json!(r.s3_matter);
            obj["s3_jacobian"] = json!(r.s3_jacobian);
        }
        obj
    }).collect();
    Ok(json!(out))
}

fn handle_linear_power_spectrum(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let cosmo = state.get_cosmology()?;
    let k_values: Vec<f64> = serde_json::from_value(args["k_values"].clone())
        .map_err(|e| format!("k_values: {}", e))?;
    let pk: Vec<f64> = k_values.iter().map(|&k| cosmo.p_lin(k)).collect();
    Ok(json!({ "k": k_values, "pk": pk }))
}

fn handle_xi_bar(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let ws = state.get_workspace()?;
    let ip = IntegrationParams::fast();
    let radii: Vec<f64> = serde_json::from_value(args["radii"].clone())
        .map_err(|e| format!("radii: {}", e))?;
    let xb: Vec<f64> = radii.iter().map(|&r| pt::integrals::xi_bar_ws(r, ws, &ip)).collect();
    Ok(json!({ "radii": radii, "xi_bar": xb }))
}

fn handle_sigma2_linear(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let ws = state.get_workspace()?;
    let ip = IntegrationParams::fast();
    let radii: Vec<f64> = serde_json::from_value(args["radii"].clone())
        .map_err(|e| format!("radii: {}", e))?;
    let s2: Vec<f64> = radii.iter().map(|&r| pt::integrals::sigma2_tree_ws(r, ws, &ip)).collect();
    Ok(json!({ "radii": radii, "sigma2_lin": s2 }))
}

fn handle_skewness(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let ws = state.get_workspace()?;
    let ip = IntegrationParams::default();
    let radii: Vec<f64> = serde_json::from_value(args["radii"].clone())
        .map_err(|e| format!("radii: {}", e))?;
    let s3_m: Vec<f64> = radii.iter().map(|&r| pt::integrals::s3_tree_matter(r, ws, &ip)).collect();
    let s3_j: Vec<f64> = radii.iter().map(|&r| pt::integrals::s3_tree_jacobian(r, ws, &ip)).collect();
    Ok(json!({ "radii": radii, "s3_matter": s3_m, "s3_jacobian": s3_j }))
}

fn handle_bias_integrals(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let ws = state.get_workspace()?;
    let ip = IntegrationParams::default();
    let r = args["r"].as_f64().ok_or("r required")?;
    let bi = pt::integrals::s3_bias_integrals(r, ws, &ip);
    Ok(json!({
        "r": r,
        "i_f2": bi.i_f2, "i_f2j": bi.i_f2j, "i_delta2": bi.i_delta2,
        "i_s2": bi.i_s2, "i_nabla": bi.i_nabla, "i_cs2": bi.i_cs2,
        "s3_matter": bi.s3_matter(), "s3_jacobian": bi.s3_jacobian()
    }))
}

fn handle_doroshkevich_moments(args: &Value) -> Result<Value, String> {
    let sigma = args["sigma"].as_f64().ok_or("sigma required")?;
    let ng = args.get("n_gauss").and_then(|v| v.as_u64()).unwrap_or(48) as usize;
    let m = pt::doroshkevich::doroshkevich_moments(sigma, ng, 6.0);
    Ok(json!({
        "sigma": m.sigma, "mean_j": m.mean_j, "variance": m.variance,
        "mu3": m.mu3, "mu4": m.mu4, "s3": m.s3, "s4": m.s4
    }))
}

fn handle_doroshkevich_cdf(args: &Value) -> Result<Value, String> {
    let sigma = args["sigma"].as_f64().ok_or("sigma required")?;
    let jt: Vec<f64> = serde_json::from_value(args["j_thresholds"].clone())
        .map_err(|e| format!("j_thresholds: {}", e))?;
    let b1 = args.get("b1").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let ng = args.get("n_gauss").and_then(|v| v.as_u64()).unwrap_or(48) as usize;
    let cdf = pt::doroshkevich::doroshkevich_cdf_biased(sigma, b1, &jt, ng, 6.0);
    Ok(json!({ "j_thresholds": jt, "cdf": cdf, "sigma": sigma, "b1": b1 }))
}

fn handle_rknn_cdf_predict(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let ws = state.get_workspace()?;
    let ip = IntegrationParams::fast();
    let k_values: Vec<usize> = serde_json::from_value(args["k_values"].clone())
        .map_err(|e| format!("k_values: {}", e))?;
    let nbar = args["nbar"].as_f64().ok_or("nbar required")?;
    let r_values: Vec<f64> = serde_json::from_value(args["r_values"].clone())
        .map_err(|e| format!("r_values: {}", e))?;
    let n_lpt = args.get("n_lpt").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
    let with_poisson = args.get("with_poisson").and_then(|v| v.as_bool()).unwrap_or(false);

    let params = KnnCdfParams { n_lpt, ..KnnCdfParams::default() };

    let results: Vec<Value> = k_values.iter().map(|&k| {
        if with_poisson {
            let pred = pt::knn_cdf::rknn_cdf_with_poisson(k, nbar, &r_values, ws, &ip, &params);
            json!({
                "k": k, "r_lag": pred.base.r_lag,
                "sigma_lin": pred.base.sigma_lin, "sigma_eff": pred.base.sigma_eff,
                "cdf": pred.base.cdf, "cdf_poisson_corrected": pred.cdf_poisson
            })
        } else {
            let pred = pt::knn_cdf::rknn_cdf_physical(k, nbar, &r_values, ws, &ip, &params);
            json!({
                "k": k, "r_lag": pred.r_lag,
                "sigma_lin": pred.sigma_lin, "sigma_eff": pred.sigma_eff,
                "cdf": pred.cdf
            })
        }
    }).collect();

    Ok(json!({ "r_values": r_values, "nbar": nbar, "predictions": results }))
}

fn handle_dknn_cdf_predict(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let ws = state.get_workspace()?;
    let ip = IntegrationParams::fast();
    let k_values: Vec<usize> = serde_json::from_value(args["k_values"].clone())
        .map_err(|e| format!("k_values: {}", e))?;
    let nbar_ref = args["nbar_ref"].as_f64().ok_or("nbar_ref required")?;
    let b1 = args["b1"].as_f64().ok_or("b1 required")?;
    let r_values: Vec<f64> = serde_json::from_value(args["r_values"].clone())
        .map_err(|e| format!("r_values: {}", e))?;
    let n_lpt = args.get("n_lpt").and_then(|v| v.as_u64()).unwrap_or(2) as usize;

    let params = KnnCdfParams { n_lpt, ..KnnCdfParams::default() };
    let results: Vec<Value> = k_values.iter().map(|&k| {
        let pred = pt::knn_cdf::dknn_cdf_biased(k, nbar_ref, b1, &r_values, ws, &ip, &params);
        json!({ "k": k, "r_lag": pred.r_lag, "sigma_lin": pred.sigma_lin, "sigma_eff": pred.sigma_eff, "cdf": pred.cdf })
    }).collect();

    Ok(json!({ "r_values": r_values, "nbar_ref": nbar_ref, "b1": b1, "predictions": results }))
}

fn handle_geometry(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let from = args["from"].as_str().ok_or("from required")?;
    let values: Vec<f64> = serde_json::from_value(args["values"].clone())
        .map_err(|e| format!("values: {}", e))?;
    let omega_m = args.get("omega_m").and_then(|v| v.as_f64())
        .or_else(|| state.cosmology.as_ref().map(|c| c.omega_m))
        .unwrap_or(0.3153);
    let nbar = args.get("nbar").and_then(|v| v.as_f64());

    let results: Vec<Value> = values.iter().map(|&v| {
        match from {
            "mass" => {
                let r = pt::mass_to_radius(v, omega_m);
                json!({"mass": v, "radius": r, "k_eff": pt::radius_to_k_eff(r)})
            }
            "radius" => {
                let m = pt::radius_to_mass(v, omega_m);
                json!({"radius": v, "mass": m, "k_eff": pt::radius_to_k_eff(v)})
            }
            "k_eff" => {
                let r = pt::k_eff_to_radius(v);
                json!({"k_eff": v, "radius": r, "mass": pt::radius_to_mass(r, omega_m)})
            }
            "knn" => {
                let nb = nbar.unwrap_or(1e-3);
                let r = pt::knn_to_radius(v as usize, nb);
                json!({"k": v as usize, "radius": r, "mass": pt::radius_to_mass(r, omega_m), "k_eff": pt::radius_to_k_eff(r), "nbar": nb})
            }
            _ => json!({"error": "unknown from type"})
        }
    }).collect();

    Ok(json!(results))
}

fn handle_generate_mock(args: &Value) -> Result<Value, String> {
    let params = if let Some(preset) = args.get("preset").and_then(|v| v.as_str()) {
        match preset {
            "euclid_small" => CoxMockParams::euclid_small(),
            "tiny" => CoxMockParams::tiny(),
            "validation" => CoxMockParams::validation(),
            "poisson" => CoxMockParams::poisson(),
            _ => return Err(format!("Unknown preset: {}", preset)),
        }
    } else {
        CoxMockParams {
            box_size: args["box_size"].as_f64().ok_or("box_size required")?,
            n_lines: args["n_lines"].as_u64().ok_or("n_lines required")? as usize,
            line_length: args["line_length"].as_f64().ok_or("line_length required")?,
            n_points: args["n_points"].as_u64().ok_or("n_points required")? as usize,
        }
    };

    let seed = args.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);
    let mock = CoxMock::generate(&params, seed);
    let n_rand = args.get("n_randoms").and_then(|v| v.as_u64())
        .unwrap_or((params.n_points * 5) as u64) as usize;
    let randoms = CoxMock::generate_randoms(n_rand, params.box_size, seed + 1000);

    Ok(json!({
        "n_data": mock.positions.len(),
        "n_randoms": randoms.len(),
        "box_size": params.box_size,
        "nbar": params.nbar(),
        "data_points": mock.positions,
        "random_points": randoms,
        "params": params,
        "xi_analytic_at_10": params.xi_analytic(10.0),
        "xi_analytic_at_50": params.xi_analytic(50.0)
    }))
}

fn parse_points(v: &Value) -> Result<Vec<[f64; 3]>, String> {
    let arr: Vec<Vec<f64>> = serde_json::from_value(v.clone())
        .map_err(|e| format!("Invalid points: {}", e))?;
    Ok(arr.into_iter().map(|p| [p[0], p[1], p[2]]).collect())
}

fn handle_estimate_xi(args: &Value) -> Result<Value, String> {
    let data = parse_points(&args["data_points"])?;
    let box_size = args["box_size"].as_f64().ok_or("box_size required")?;
    let k_max = args.get("k_max").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
    let n_bins = args.get("n_bins").and_then(|v| v.as_u64()).unwrap_or(30) as usize;
    let r_min = args.get("r_min").and_then(|v| v.as_f64()).unwrap_or(5.0);
    let r_max = args.get("r_max").and_then(|v| v.as_f64()).unwrap_or(200.0);

    let randoms = if args.get("random_points").is_some() && !args["random_points"].is_null() {
        parse_points(&args["random_points"])?
    } else {
        let n_rand = args.get("n_randoms").and_then(|v| v.as_u64()).unwrap_or(50000) as usize;
        CoxMock::generate_randoms(n_rand, box_size, 99999)
    };

    let data_tree = PointTree::build(data.clone());
    let random_tree = PointTree::build(randoms.clone());
    let estimator = LandySzalayKnn::new(k_max);

    let dd_dists = estimator.query_distances_periodic(&data_tree, &data, box_size);
    let dd_dists = exclude_self_pairs(dd_dists, k_max);
    let dr_dists = estimator.query_distances_periodic(&random_tree, &data, box_size);

    let r_edges = linear_bins(r_min, r_max, n_bins);
    let dd = LandySzalayKnn::pair_count_density(&dd_dists, &r_edges);
    let dr = LandySzalayKnn::pair_count_density(&dr_dists, &r_edges);
    let xi = LandySzalayKnn::estimate_xi_dp(&dd, &dr);

    Ok(json!({
        "r": xi.r,
        "xi": xi.xi,
        "r2_xi": xi.r2_xi(),
        "n_data": data.len(),
        "n_randoms": randoms.len(),
        "k_max": k_max
    }))
}

fn handle_empirical_cdfs(args: &Value) -> Result<Value, String> {
    let data = parse_points(&args["data_points"])?;
    let box_size = args["box_size"].as_f64().ok_or("box_size required")?;
    let k_max = args.get("k_max").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
    let n_r = args.get("n_r").and_then(|v| v.as_u64()).unwrap_or(150) as usize;
    let r_min = args.get("r_min").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let r_max = args.get("r_max").and_then(|v| v.as_f64()).unwrap_or(200.0);

    let k_values: Vec<usize> = if let Some(kv) = args.get("k_values") {
        serde_json::from_value(kv.clone()).map_err(|e| format!("k_values: {}", e))?
    } else {
        cdf_k_values(k_max)
    };

    let r_grid = cdf_r_grid(r_min, r_max, n_r);

    let data_tree = PointTree::build(data.clone());
    let estimator = LandySzalayKnn::new(k_max);
    let dists = estimator.query_distances_periodic(&data_tree, &data, box_size);
    let dists = exclude_self_pairs(dists, k_max);
    let cdfs = LandySzalayKnn::empirical_cdfs(&dists, &k_values, &r_grid);

    Ok(json!({
        "r_values": cdfs.r_values,
        "k_values": cdfs.k_values,
        "cdf_values": cdfs.cdf_values,
        "n_queries": cdfs.n_queries
    }))
}

fn handle_count_cumulants(args: &Value) -> Result<Value, String> {
    let data = parse_points(&args["data_points"])?;
    let box_size = args["box_size"].as_f64().ok_or("box_size required")?;
    let k_max = args.get("k_max").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
    let max_level = args.get("max_dilution_level").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
    let n_r = args.get("n_r").and_then(|v| v.as_u64()).unwrap_or(30) as usize;
    let r_min = args.get("r_min").and_then(|v| v.as_f64()).unwrap_or(5.0);
    let r_max = args.get("r_max").and_then(|v| v.as_f64()).unwrap_or(200.0);

    let randoms = CoxMock::generate_randoms(data.len() * 5, box_size, 77777);
    let ladder = DilutionLadder::build(data.len(), max_level, 12345);
    let profiles = twopoint::PerPointProfiles::compute(&data, &randoms, &ladder, k_max, box_size);
    let r_grid = cdf_r_grid(r_min, r_max, n_r);
    let cum = profiles.count_cumulants(&r_grid, &twopoint::RrMode::Analytic);

    Ok(json!({
        "r": cum.r,
        "xi_bar": cum.xi_bar,
        "sigma2": cum.sigma2,
        "s3": cum.s3,
        "s4": cum.s4,
        "sigma2_err": cum.sigma2_err,
        "s3_err": cum.s3_err,
        "n_points": cum.n_points
    }))
}

fn handle_jacobian_cumulants(args: &Value) -> Result<Value, String> {
    let data = parse_points(&args["data_points"])?;
    let box_size = args["box_size"].as_f64().ok_or("box_size required")?;
    let k_max = args.get("k_max").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
    let k_values: Vec<usize> = if let Some(kv) = args.get("k_values") {
        serde_json::from_value(kv.clone()).map_err(|e| format!("k_values: {}", e))?
    } else {
        vec![1, 2, 4, 8, 16, 32, 64].into_iter().filter(|&k| k <= k_max).collect()
    };

    let vol = box_size * box_size * box_size;
    let nbar = data.len() as f64 / vol;
    let data_tree = PointTree::build(data.clone());
    let estimator = LandySzalayKnn::new(k_max);
    let dists = estimator.query_distances_periodic(&data_tree, &data, box_size);
    let dists = exclude_self_pairs(dists, k_max);

    let cums = jacobian_cumulants_multi_k(&dists, &k_values, nbar);
    let results: Vec<Value> = cums.iter().map(|c| json!({
        "k": c.k, "r_lag": c.r_lag, "mean_j": c.mean_j,
        "variance": c.variance, "s3": c.s3, "s4": c.s4,
        "n_points": c.n_points
    })).collect();

    Ok(json!({ "nbar": nbar, "results": results }))
}

fn handle_run_validation(args: &Value) -> Result<Value, String> {
    let preset = args.get("preset").and_then(|v| v.as_str()).unwrap_or("validation");
    let params = match preset {
        "euclid_small" => CoxMockParams::euclid_small(),
        "tiny" => CoxMockParams::tiny(),
        "validation" => CoxMockParams::validation(),
        "poisson" => CoxMockParams::poisson(),
        _ => return Err(format!("Unknown preset: {}", preset)),
    };
    let config = ValidationConfig {
        n_mocks: args.get("n_mocks").and_then(|v| v.as_u64()).unwrap_or(10) as usize,
        k_max: args.get("k_max").and_then(|v| v.as_u64()).unwrap_or(8) as usize,
        n_bins: args.get("n_bins").and_then(|v| v.as_u64()).unwrap_or(40) as usize,
        r_min: args.get("r_min").and_then(|v| v.as_f64()).unwrap_or(5.0),
        r_max: args.get("r_max").and_then(|v| v.as_f64()).unwrap_or(250.0),
        random_ratio: args.get("random_ratio").and_then(|v| v.as_u64()).unwrap_or(5) as usize,
        params: params.clone(),
        max_dilution_level: args.get("max_dilution_level").and_then(|v| v.as_u64()).unwrap_or(2) as usize,
        box_size: Some(params.box_size),
    };

    let mocks: Vec<_> = (0..config.n_mocks).map(|i| run_single_mock(&config, i)).collect();
    let result = aggregate_mocks(&config, &mocks);

    Ok(serde_json::to_value(&result).map_err(|e| format!("serialize: {}", e))?)
}

fn handle_erlang_cdf(args: &Value) -> Result<Value, String> {
    let k = args["k"].as_u64().ok_or("k required")? as usize;
    let nbar = args["nbar"].as_f64().ok_or("nbar required")?;
    let r_values: Vec<f64> = serde_json::from_value(args["r_values"].clone())
        .map_err(|e| format!("r_values: {}", e))?;
    let cdf: Vec<f64> = r_values.iter().map(|&r| erlang_cdf(k, r, nbar)).collect();
    let pdf: Vec<f64> = r_values.iter().map(|&r| erlang_pdf(k, r, nbar)).collect();
    Ok(json!({ "k": k, "nbar": nbar, "r_values": r_values, "cdf": cdf, "pdf": pdf }))
}

fn handle_xibar_j_full(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let cosmo = state.get_cosmology()?.clone();
    let radii: Vec<f64> = serde_json::from_value(args["radii"].clone())
        .map_err(|e| format!("radii: {}", e))?;
    let b1 = args["b1"].as_f64().ok_or("b1 required")?;
    let n_corr = args.get("n_corrections").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
    let f_growth = args.get("f_growth").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let results = if f_growth > 0.0 {
        let rsd = pt::integrals::RsdParams { f: f_growth, n_los: 12 };
        theory::xibar_j_plot_rsd(&cosmo, &radii, b1, &rsd, n_corr)
    } else {
        theory::xibar_j_plot(&cosmo, &radii, b1, n_corr)
    };
    let out: Vec<Value> = results.iter().map(|r| json!({
        "r": r.r,
        "b1": r.b1,
        "f_growth": r.f_growth,
        "xibar_tree": r.xibar_tree,
        "xibar_zel": r.xibar_zel,
        "xibar_1loop": r.xibar_1loop,
        "xibar_full": r.xibar_full,
        "sigma2_s": r.sigma2_lin,
        "epsilon": r.epsilon
    })).collect();
    Ok(json!(out))
}

fn handle_theory_at_knn_scale(state: &mut ServerState, args: &Value) -> Result<Value, String> {
    let cosmo = state.get_cosmology()?.clone();
    let k_values: Vec<usize> = serde_json::from_value(args["k_values"].clone())
        .map_err(|e| format!("k_values: {}", e))?;
    let nbar = args["nbar"].as_f64().ok_or("nbar required")?;

    let results: Vec<Value> = k_values.iter().map(|&k| {
        let r = theory::theory_at_knn_scale(k, nbar, &cosmo);
        json!({
            "k": k, "r": r.r, "mass": r.mass, "k_eff": r.k_eff,
            "sigma2_lin": r.sigma2_lin, "sigma2_zel": r.sigma2_zel, "sigma2_j": r.sigma2_j,
            "xi_bar": r.xi_bar, "s3_matter": r.s3_matter, "s3_jacobian": r.s3_jacobian
        })
    }).collect();

    Ok(json!({ "nbar": nbar, "predictions": results }))
}

// ═══════════════════════════════════════════════════════════════════════════
// MCP protocol handling
// ═══════════════════════════════════════════════════════════════════════════

fn handle_request(state: &mut ServerState, req: &JsonRpcRequest) -> JsonRpcResponse {
    let id = req.id.clone().unwrap_or(Value::Null);

    let result = match req.method.as_str() {
        "initialize" => Ok(json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "twopoint-mcp",
                "version": "0.1.0"
            }
        })),
        // Notifications get no response per JSON-RPC 2.0 spec
        "notifications/initialized" | "notifications/cancelled" => return JsonRpcResponse {
            jsonrpc: "2.0".into(), id, result: None, error: None,
        },
        "tools/list" => Ok(json!({ "tools": tool_definitions() })),
        "tools/call" => {
            let tool_name = req.params.get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let tool_args = req.params.get("arguments")
                .cloned()
                .unwrap_or(json!({}));

            match handle_tool_call(state, tool_name, &tool_args) {
                Ok(result) => Ok(json!({
                    "content": [{
                        "type": "text",
                        "text": serde_json::to_string_pretty(&result).unwrap_or_default()
                    }]
                })),
                Err(e) => Ok(json!({
                    "content": [{
                        "type": "text",
                        "text": format!("Error: {}", e)
                    }],
                    "isError": true
                })),
            }
        }
        _ => Err(format!("Unknown method: {}", req.method)),
    };

    match result {
        Ok(r) => JsonRpcResponse {
            jsonrpc: "2.0".into(), id, result: Some(r), error: None,
        },
        Err(e) => JsonRpcResponse {
            jsonrpc: "2.0".into(), id, result: None,
            error: Some(JsonRpcError { code: -32601, message: e }),
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main: stdio JSON-RPC loop
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    // Debug logging to file
    use std::fs::OpenOptions;
    let mut dbg = OpenOptions::new().create(true).append(true)
        .open("/tmp/twopoint-mcp-rust.log").ok();
    macro_rules! dbglog {
        ($($arg:tt)*) => {
            if let Some(ref mut f) = dbg {
                let _ = writeln!(f, "{}", format!($($arg)*));
                let _ = f.flush();
            }
        }
    }
    dbglog!("=== START pid={} ===", std::process::id());

    let stdin = io::stdin();
    let mut state = ServerState::new();
    let mut line_buf = String::new();
    // If we see EOF repeatedly after having already processed messages, exit.
    // This supports both piped stdin (exits after input) and Claude Code's
    // long-lived connection (stdin closed briefly but reopened).
    let mut eof_count: u32 = 0;
    let mut any_message_seen = false;

    dbglog!("Entering main loop");

    loop {
        line_buf.clear();
        match stdin.lock().read_line(&mut line_buf) {
            Ok(0) => {
                eof_count += 1;
                // If EOF persists for >2s after we've received at least one message,
                // the parent is gone — exit cleanly.
                if any_message_seen && eof_count > 40 {
                    dbglog!("EOF persisted after messages; exiting");
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
                continue;
            }
            Ok(_) => { eof_count = 0; }
            Err(e) => {
                dbglog!("Read error: {}, sleeping...", e);
                std::thread::sleep(std::time::Duration::from_millis(50));
                continue;
            }
        }

        let line = line_buf.trim();
        if line.is_empty() {
            continue;
        }

        // Skip Content-Length headers (we parse bare JSON lines)
        if line.starts_with("Content-Length:") {
            continue;
        }

        if !line.starts_with('{') {
            dbglog!("Skipping non-JSON line: {:?}", &line[..line.len().min(60)]);
            continue;
        }

        any_message_seen = true;
        dbglog!("Parsing JSON, len={}", line.len());
        let req: JsonRpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                dbglog!("Parse error: {}", e);
                continue;
            }
        };

        let is_notification = req.id.is_none();
        dbglog!("method={} id={:?} notification={}", req.method, req.id, is_notification);

        let resp = handle_request(&mut state, &req);

        if !is_notification {
            let body = serde_json::to_string(&resp).unwrap();
            dbglog!("Sending response: {} bytes", body.len());
            let out = io::stdout();
            let mut out = out.lock();
            let _ = writeln!(out, "{}", body);
            let _ = out.flush();
            dbglog!("Response sent");
        }
    }
}
