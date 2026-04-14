//! Loop integrals for sigma^2_J perturbation theory.
//!
//! Tree: sigma^2_lin = int k^3/(2pi^2) |W(kR)|^2 P_L(k) d ln k
//! 1-loop: P22 + 2*P13 corrections
//! 2-loop: P33 + 2*P24 + 2*P15 (factorised)
//! Counterterms: c_J^2 * sigma^2_{J,2}(R) + c_J^4 * sigma^2_{J,4}(R)

use std::f64::consts::PI;
use crate::cosmology::{Cosmology, top_hat};

/// Integration grid parameters.
#[derive(Clone, Copy)]
pub struct IntegrationParams {
    pub n_k: usize,   // outer k-integral points
    pub n_p: usize,   // inner p-integral points
    pub n_mu: usize,  // angular integral points
    pub ln_k_min: f64,
    pub ln_k_max: f64,
}

impl Default for IntegrationParams {
    fn default() -> Self {
        IntegrationParams {
            n_k: 250,
            n_p: 200,
            n_mu: 48,
            ln_k_min: (1e-5_f64).ln(),
            ln_k_max: (100.0_f64).ln(),
        }
    }
}

impl IntegrationParams {
    pub fn fast() -> Self {
        IntegrationParams { n_k: 150, n_p: 120, n_mu: 32,
            ln_k_min: (1e-5_f64).ln(), ln_k_max: (100.0_f64).ln() }
    }
}

// ── Workspace-based integral functions ───────────────────────────────────
use crate::Workspace;

pub fn sigma2_tree_ws(r: f64, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    let n = ip.n_k; let dlk = (ip.ln_k_max - ip.ln_k_min) / n as f64;
    let mut s = 0.0;
    for i in 0..=n {
        let k = (ip.ln_k_min + i as f64 * dlk).exp();
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        let wt = top_hat(k * r);
        s += w * k.powi(3) / (2.0 * PI * PI) * wt * wt * ws.p_lin(k);
    }
    s * dlk
}

pub fn sigma2_jn_ws(r: f64, power: i32, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    let n = ip.n_k; let dlk = (ip.ln_k_max - ip.ln_k_min) / n as f64;
    let mut s = 0.0;
    for i in 0..=n {
        let k = (ip.ln_k_min + i as f64 * dlk).exp();
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        let wt = top_hat(k * r);
        s += w * k.powi(3 + power) / (2.0 * PI * PI) * wt * wt * ws.p_lin(k);
    }
    s * dlk
}

pub fn sigma2_p22_raw_ws(r: f64, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    use rayon::prelude::*;
    let (nk, np, nmu) = (ip.n_k, ip.n_p, ip.n_mu);
    let lke = (20.0/r+1.0).ln().min(ip.ln_k_max);
    let dlk = (lke - ip.ln_k_min)/nk as f64;
    let dlp = (ip.ln_k_max - ip.ln_k_min)/np as f64;
    let dmu = 2.0/nmu as f64;
    let c1 = 1.0/(4.0*PI*PI); let c2 = 1.0/(2.0*PI*PI);
    let tot: f64 = (0..=nk).into_par_iter().map(|ik| {
        let k=(ip.ln_k_min+ik as f64*dlk).exp();
        let wk=if ik==0||ik==nk{0.5}else{1.0};
        let wt=top_hat(k*r); if wt.abs()<1e-15 { return 0.0; } let k2=k*k;
        let mut pk=0.0;
        for ip2 in 0..=np { let p=(ip.ln_k_min+ip2 as f64*dlp).exp();
            let wp=if ip2==0||ip2==np{0.5}else{1.0};
            let plp=ws.p_lin(p); if plp<1e-30{continue;} let p2=p*p;
            let mut ms=0.0;
            for jm in 0..=nmu { let mu=-1.0+jm as f64*dmu;
                let wm=if jm==0||jm==nmu{0.5}else{1.0};
                let q2=k2+p2-2.0*k*p*mu; if q2<1e-20{continue;}
                ms+=wm*(k2*(1.0-mu*mu)/q2).powi(2)*plp*ws.p_lin(q2.sqrt());
            } pk+=wp*p*p2*ms*dmu;
        }
        wk*k*k2*c2*wt*wt*pk*c1*dlp
    }).sum();
    tot*dlk
}

pub fn sigma2_p13_raw_ws(r: f64, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    use rayon::prelude::*;
    let (nk, np, nmu) = (ip.n_k, ip.n_p, ip.n_mu);
    let lke = (20.0/r+1.0).ln().min(ip.ln_k_max);
    let dlk = (lke - ip.ln_k_min)/nk as f64;
    let dlp = (ip.ln_k_max - ip.ln_k_min)/np as f64;
    let dmu = 2.0/nmu as f64;
    let c1 = 1.0/(4.0*PI*PI); let c2 = 1.0/(2.0*PI*PI);
    let tot: f64 = (0..=nk).into_par_iter().map(|ik| {
        let k=(ip.ln_k_min+ik as f64*dlk).exp();
        let wk=if ik==0||ik==nk{0.5}else{1.0};
        let wt=top_hat(k*r); if wt.abs()<1e-15 { return 0.0; }
        let plk=ws.p_lin(k); if plk<1e-30 { return 0.0; } let k2=k*k;
        let mut ig=0.0;
        for ip2 in 0..=np { let p=(ip.ln_k_min+ip2 as f64*dlp).exp();
            let wp=if ip2==0||ip2==np{0.5}else{1.0};
            let plp=ws.p_lin(p); if plp<1e-30{continue;} let p2=p*p;
            let mut ms=0.0;
            for jm in 0..=nmu { let mu=-1.0+jm as f64*dmu;
                let wm=if jm==0||jm==nmu{0.5}else{1.0};
                let q2=k2+p2-2.0*k*p*mu; if q2<1e-20{continue;}
                let pd=k*p*mu-p2;
                ms+=wm*(1.0-mu*mu)*(1.0-pd*pd/(p2*q2));
            } ig+=wp*p*p2*plp*ms*dmu;
        }
        wk*k*k2*c2*wt*wt*plk*c1*ig*dlp
    }).sum();
    tot*dlk
}

/// R-independent inner P₁₃ kernel at a given k.
///
/// Returns the "inner" 3D integral of the P₁₃ structure, stripped of the
/// outer W(kR) (or W²(kR)) smoothing and the leading P_L(k). That is,
/// if the full P₁₃-smoothed integral is
///   ∫ d ln k (k³/2π²) W^n(kR) P_L(k) × I_P13(k),
/// this returns I_P13(k). R-independent, allowing outer FFTLog smoothing.
pub fn p13_inner_kernel(k: f64, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    use rayon::prelude::*;
    let (np, nmu) = (ip.n_p, ip.n_mu);
    let dlp = (ip.ln_k_max - ip.ln_k_min) / np as f64;
    let dmu = 2.0 / nmu as f64;
    let c1 = 1.0 / (4.0 * PI * PI);
    let k2 = k * k;
    let inner: f64 = (0..=np).into_par_iter().map(|ip2| {
        let p = (ip.ln_k_min + ip2 as f64 * dlp).exp();
        let wp = if ip2 == 0 || ip2 == np { 0.5 } else { 1.0 };
        let plp = ws.p_lin(p);
        if plp < 1e-30 { return 0.0; }
        let p2 = p * p;
        let mut ms = 0.0;
        for jm in 0..=nmu {
            let mu = -1.0 + jm as f64 * dmu;
            let wm = if jm == 0 || jm == nmu { 0.5 } else { 1.0 };
            let q2 = k2 + p2 - 2.0 * k * p * mu;
            if q2 < 1e-20 { continue; }
            let pd = k * p * mu - p2;
            ms += wm * (1.0 - mu * mu) * (1.0 - pd * pd / (p2 * q2));
        }
        wp * p * p2 * plp * ms * dmu
    }).sum();
    inner * c1 * dlp
}

/// Tabulate I_P13(k) on a log-k grid (FFTLog input grid).
/// One-time cost per cosmology; expensive 3D quadrature but parallelized.
pub fn p13_effective_pk_table(
    ws: &Workspace, ip: &IntegrationParams,
    ln_k_min: f64, ln_k_max: f64, n: usize,
) -> Vec<f64> {
    use rayon::prelude::*;
    let dlnk = (ln_k_max - ln_k_min) / (n - 1) as f64;
    (0..n).into_par_iter().map(|i| {
        let k = (ln_k_min + i as f64 * dlnk).exp();
        let plk = ws.p_lin(k);
        if plk < 1e-30 { return 0.0; }
        plk * p13_inner_kernel(k, ws, ip)
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// P₂₂-cross kernels for the Lagrangian bias expansion (b₂ and b_{s²})
//
// The one-loop correction to the biased Jacobian cross-spectrum ξ̄_J picks up
// a P₂₂-type term. In the Lagrangian/Jacobian framework used throughout this
// crate, the relevant F₂ is the Jacobian kernel
//     F₂^{(J)}(p, q) = (2/7)(1 − ν²),  ν = p̂·q̂
// NOT the full SPT F₂ (which contains (p/q + q/p) terms absorbed elsewhere
// into the Zel'dovich flow / polynomial-LPT structure, and whose IR
// divergences cancel only after adding the corresponding P₁₃ piece). The
// Jacobian F₂^{(J)} is IR-safe and matches the tidal structure already used
// by `sigma2_p22_raw_ws` in this crate. With this choice,
//
//   F₂,bias(p, q) = b₂/2 + b_{s²} S₂(p, q),   S₂ = ν² − 1/3
//   P₂₂,cross(k) = 2 ∫ F₂^{(J)} · F₂,bias · P_L(p) P_L(|k−p|) d³p/(2π)³
//                = b₂ · I_{b₂}(k) + b_{s²} · I_{bs²}(k)
//
// where (after substituting F₂^{(J)} = (2/7)(1−ν²))
//   I_{b₂}(k)  =     ∫ (2/7)(1−ν²) · P_L(p) P_L(q) d³p/(2π)³
//   I_{bs²}(k) = 2 ∫ (2/7)(1−ν²) · (ν² − 1/3) · P_L(p) P_L(q) d³p/(2π)³
// with q = k − p, q² = k² + p² − 2kpμ, ν = (kμ − p)/q, and the measure
// d³p/(2π)³ → p²/(4π²) dp dμ for axially-symmetric integrands.
// ═══════════════════════════════════════════════════════════════════════════

/// Inner (p, μ) integral for P₂₂,cross,b₂ at fixed k (Jacobian F₂^{(J)}).
pub fn p22_cross_b2_inner_kernel(
    k: f64, ws: &Workspace, ip: &IntegrationParams,
) -> f64 {
    use rayon::prelude::*;
    let (np, nmu) = (ip.n_p, ip.n_mu);
    let dlp = (ip.ln_k_max - ip.ln_k_min) / np as f64;
    let dmu = 2.0 / nmu as f64;
    let c_outer = 1.0 / (4.0 * PI * PI);
    let k2 = k * k;
    let inner: f64 = (0..=np).into_par_iter().map(|ip2| {
        let p = (ip.ln_k_min + ip2 as f64 * dlp).exp();
        let wp = if ip2 == 0 || ip2 == np { 0.5 } else { 1.0 };
        let plp = ws.p_lin(p);
        if plp < 1e-30 { return 0.0; }
        let p2 = p * p;
        let mut ms = 0.0;
        for jm in 0..=nmu {
            let mu = -1.0 + jm as f64 * dmu;
            let wm = if jm == 0 || jm == nmu { 0.5 } else { 1.0 };
            let q2 = k2 + p2 - 2.0 * k * p * mu;
            if q2 < 1e-20 { continue; }
            let q = q2.sqrt();
            let plq = ws.p_lin(q);
            if plq < 1e-30 { continue; }
            let nu = (k * mu - p) / q;
            let f2j = (2.0 / 7.0) * (1.0 - nu * nu);
            ms += wm * f2j * plq;
        }
        wp * p * p2 * plp * ms * dmu
    }).sum();
    inner * c_outer * dlp
}

/// Inner (p, μ) integral for P₂₂,cross,b_{s²} at fixed k (Jacobian F₂^{(J)}).
pub fn p22_cross_bs2_inner_kernel(
    k: f64, ws: &Workspace, ip: &IntegrationParams,
) -> f64 {
    use rayon::prelude::*;
    let (np, nmu) = (ip.n_p, ip.n_mu);
    let dlp = (ip.ln_k_max - ip.ln_k_min) / np as f64;
    let dmu = 2.0 / nmu as f64;
    let c_outer = 1.0 / (4.0 * PI * PI);
    let k2 = k * k;
    let inner: f64 = (0..=np).into_par_iter().map(|ip2| {
        let p = (ip.ln_k_min + ip2 as f64 * dlp).exp();
        let wp = if ip2 == 0 || ip2 == np { 0.5 } else { 1.0 };
        let plp = ws.p_lin(p);
        if plp < 1e-30 { return 0.0; }
        let p2 = p * p;
        let mut ms = 0.0;
        for jm in 0..=nmu {
            let mu = -1.0 + jm as f64 * dmu;
            let wm = if jm == 0 || jm == nmu { 0.5 } else { 1.0 };
            let q2 = k2 + p2 - 2.0 * k * p * mu;
            if q2 < 1e-20 { continue; }
            let q = q2.sqrt();
            let plq = ws.p_lin(q);
            if plq < 1e-30 { continue; }
            let nu = (k * mu - p) / q;
            let f2j = (2.0 / 7.0) * (1.0 - nu * nu);
            let s2 = nu * nu - 1.0 / 3.0;
            ms += wm * f2j * s2 * plq;
        }
        wp * p * p2 * plp * ms * dmu
    }).sum();
    2.0 * inner * c_outer * dlp
}

/// Tabulate I_{b₂}(k) on a log-k grid (FFTLog input).
pub fn p22_cross_b2_effective_pk_table(
    ws: &Workspace, ip: &IntegrationParams,
    ln_k_min: f64, ln_k_max: f64, n: usize,
) -> Vec<f64> {
    use rayon::prelude::*;
    let dlnk = (ln_k_max - ln_k_min) / (n - 1) as f64;
    (0..n).into_par_iter().map(|i| {
        let k = (ln_k_min + i as f64 * dlnk).exp();
        p22_cross_b2_inner_kernel(k, ws, ip)
    }).collect()
}

/// Tabulate I_{bs²}(k) on a log-k grid (FFTLog input).
pub fn p22_cross_bs2_effective_pk_table(
    ws: &Workspace, ip: &IntegrationParams,
    ln_k_min: f64, ln_k_max: f64, n: usize,
) -> Vec<f64> {
    use rayon::prelude::*;
    let dlnk = (ln_k_max - ln_k_min) / (n - 1) as f64;
    (0..n).into_par_iter().map(|i| {
        let k = (ln_k_min + i as f64 * dlnk).exp();
        p22_cross_bs2_inner_kernel(k, ws, ip)
    }).collect()
}

/// P₁₃ kernel integral smoothed with single W(kR) — for ξ̄ cross-spectrum.
/// Same as sigma2_p13_raw_ws but W(kR) instead of W²(kR).
///
/// Uses full k-range (no R-dependent truncation) to avoid discontinuities
/// in R-dependence. The top-hat W(kR) naturally suppresses high-k.
pub fn xibar_p13_raw_ws(r: f64, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    use rayon::prelude::*;
    let (nk, np, nmu) = (ip.n_k, ip.n_p, ip.n_mu);
    let lke = ip.ln_k_max;
    let dlk = (lke - ip.ln_k_min)/nk as f64;
    let dlp = (ip.ln_k_max - ip.ln_k_min)/np as f64;
    let dmu = 2.0/nmu as f64;
    let c1 = 1.0/(4.0*PI*PI); let c2 = 1.0/(2.0*PI*PI);
    let tot: f64 = (0..=nk).into_par_iter().map(|ik| {
        let k=(ip.ln_k_min+ik as f64*dlk).exp();
        let wk=if ik==0||ik==nk{0.5}else{1.0};
        let wt=top_hat(k*r); if wt.abs()<1e-15 { return 0.0; }
        let plk=ws.p_lin(k); if plk<1e-30 { return 0.0; } let k2=k*k;
        let mut ig=0.0;
        for ip2 in 0..=np { let p=(ip.ln_k_min+ip2 as f64*dlp).exp();
            let wp=if ip2==0||ip2==np{0.5}else{1.0};
            let plp=ws.p_lin(p); if plp<1e-30{continue;} let p2=p*p;
            let mut ms=0.0;
            for jm in 0..=nmu { let mu=-1.0+jm as f64*dmu;
                let wm=if jm==0||jm==nmu{0.5}else{1.0};
                let q2=k2+p2-2.0*k*p*mu; if q2<1e-20{continue;}
                let pd=k*p*mu-p2;
                ms+=wm*(1.0-mu*mu)*(1.0-pd*pd/(p2*q2));
            } ig+=wp*p*p2*plp*ms*dmu;
        }
        wk*k*k2*c2*wt*plk*c1*ig*dlp  // single wt, not wt*wt
    }).sum();
    tot*dlk
}

pub fn sigma2_p15_ws(r: f64, b2p13: f64, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    use rayon::prelude::*;
    let (nk, np, nmu) = (ip.n_k, ip.n_p, ip.n_mu);
    let lke = (20.0/r+1.0).ln().min(ip.ln_k_max);
    let dlk = (lke - ip.ln_k_min)/nk as f64;
    let dlp = (ip.ln_k_max - ip.ln_k_min)/np as f64;
    let dmu = 2.0/nmu as f64;
    let c1 = 1.0/(4.0*PI*PI); let c2 = 1.0/(2.0*PI*PI); let bh=b2p13/2.0;
    let tot: f64 = (0..=nk).into_par_iter().map(|ik| {
        let k=(ip.ln_k_min+ik as f64*dlk).exp();
        let wk=if ik==0||ik==nk{0.5}else{1.0};
        let wt=top_hat(k*r); if wt.abs()<1e-15 { return 0.0; }
        let plk=ws.p_lin(k); if plk<1e-30 { return 0.0; } let k2=k*k;
        let mut ig=0.0;
        for ip2 in 0..=np { let p=(ip.ln_k_min+ip2 as f64*dlp).exp();
            let wp=if ip2==0||ip2==np{0.5}else{1.0};
            let plp=ws.p_lin(p); if plp<1e-30{continue;} let p2=p*p;
            let mut ms=0.0;
            for jm in 0..=nmu { let mu=-1.0+jm as f64*dmu;
                let wm=if jm==0||jm==nmu{0.5}else{1.0};
                let q2=k2+p2-2.0*k*p*mu; if q2<1e-20{continue;}
                let pd=k*p*mu-p2;
                ms+=wm*(1.0-mu*mu)*(1.0-pd*pd/(p2*q2));
            } ig+=wp*p*p2*plp*ms*dmu;
        }
        let s13=bh*ig*dlp*c1;
        wk*k*k2*c2*wt*wt*plk*s13*s13
    }).sum();
    2.0*tot*dlk
}

/// Compute sigma^2_lin(R) = tree-level variance.
pub fn sigma2_tree(cosmo: &Cosmology, r: f64, params: &IntegrationParams) -> f64 {
    let n = params.n_k;
    let dlk = (params.ln_k_max - params.ln_k_min) / n as f64;
    let mut s = 0.0;
    for i in 0..=n {
        let k = (params.ln_k_min + i as f64 * dlk).exp();
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        let wth = top_hat(k * r);
        s += w * k.powi(3) / (2.0 * PI * PI) * wth * wth * cosmo.p_lin(k);
    }
    s * dlk
}

/// Compute sigma^2_{J,2}(R) = int k^5/(2pi^2) |W|^2 P_L dk (counterterm kernel).
pub fn sigma2_j2(cosmo: &Cosmology, r: f64, params: &IntegrationParams) -> f64 {
    let n = params.n_k;
    let dlk = (params.ln_k_max - params.ln_k_min) / n as f64;
    let mut s = 0.0;
    for i in 0..=n {
        let k = (params.ln_k_min + i as f64 * dlk).exp();
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        let wth = top_hat(k * r);
        s += w * k.powi(3) / (2.0 * PI * PI) * k * k * wth * wth * cosmo.p_lin(k);
    }
    s * dlk
}

/// Compute sigma^2_{J,4}(R) = int k^7/(2pi^2) |W|^2 P_L dk (3-loop counterterm).
pub fn sigma2_j4(cosmo: &Cosmology, r: f64, params: &IntegrationParams) -> f64 {
    let n = params.n_k;
    let dlk = (params.ln_k_max - params.ln_k_min) / n as f64;
    let mut s = 0.0;
    for i in 0..=n {
        let k = (params.ln_k_min + i as f64 * dlk).exp();
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        let wth = top_hat(k * r);
        s += w * k.powi(3) / (2.0 * PI * PI) * k.powi(4) * wth * wth * cosmo.p_lin(k);
    }
    s * dlk
}

/// P22 raw kernel integral (coefficient = 1).
/// P22_raw = int |W(kR)|^2 * [int (1-mu12^2)^2 P_L(p) P_L(|k-p|) d^3p/(4pi^2)] dk
pub fn sigma2_p22_raw(cosmo: &Cosmology, r: f64, params: &IntegrationParams) -> f64 {
    let n_k = params.n_k;
    let n_p = params.n_p;
    let n_mu = params.n_mu;
    let lk_max_eff = (20.0 / r + 1.0).ln().min(params.ln_k_max);
    let dlk = (lk_max_eff - params.ln_k_min) / n_k as f64;
    let dlp = (params.ln_k_max - params.ln_k_min) / n_p as f64;
    let dmu = 2.0 / n_mu as f64;

    let mut total = 0.0;
    for ik in 0..=n_k {
        let k = (params.ln_k_min + ik as f64 * dlk).exp();
        let wk = if ik == 0 || ik == n_k { 0.5 } else { 1.0 };
        let wth_k = top_hat(k * r);
        if wth_k.abs() < 1e-15 { continue; }

        let mut p22_k = 0.0;
        for ip in 0..=n_p {
            let p = (params.ln_k_min + ip as f64 * dlp).exp();
            let wp = if ip == 0 || ip == n_p { 0.5 } else { 1.0 };
            let pl_p = cosmo.p_lin(p);
            if pl_p < 1e-30 { continue; }

            let mut mu_sum = 0.0;
            for jm in 0..=n_mu {
                let mu = -1.0 + jm as f64 * dmu;
                let wm = if jm == 0 || jm == n_mu { 0.5 } else { 1.0 };
                let q2 = k * k + p * p - 2.0 * k * p * mu;
                if q2 < 1e-20 { continue; }
                let tidal_sq = (k * k * (1.0 - mu * mu) / q2).powi(2);
                mu_sum += wm * tidal_sq * pl_p * cosmo.p_lin(q2.sqrt());
            }
            p22_k += wp * p.powi(3) * mu_sum * dmu;
        }
        p22_k /= 4.0 * PI * PI;
        p22_k *= dlp;
        total += wk * k.powi(3) / (2.0 * PI * PI) * wth_k * wth_k * p22_k;
    }
    total * dlk
}

/// P13 raw kernel integral (coefficient = 1).
/// Uses the tidal kernel: int (1-mu^2)*tidal(p,k-p) * P_L(p) dp
pub fn sigma2_p13_raw(cosmo: &Cosmology, r: f64, params: &IntegrationParams) -> f64 {
    let n_k = params.n_k;
    let n_p = params.n_p;
    let n_mu = params.n_mu;
    let lk_max_eff = (20.0 / r + 1.0).ln().min(params.ln_k_max);
    let dlk = (lk_max_eff - params.ln_k_min) / n_k as f64;
    let dlp = (params.ln_k_max - params.ln_k_min) / n_p as f64;
    let dmu = 2.0 / n_mu as f64;

    let mut total = 0.0;
    for ik in 0..=n_k {
        let k = (params.ln_k_min + ik as f64 * dlk).exp();
        let wk = if ik == 0 || ik == n_k { 0.5 } else { 1.0 };
        let wth_k = top_hat(k * r);
        if wth_k.abs() < 1e-15 { continue; }
        let pl_k = cosmo.p_lin(k);
        if pl_k < 1e-30 { continue; }

        let mut integral = 0.0;
        for ip in 0..=n_p {
            let p = (params.ln_k_min + ip as f64 * dlp).exp();
            let wp = if ip == 0 || ip == n_p { 0.5 } else { 1.0 };
            let pl_p = cosmo.p_lin(p);
            if pl_p < 1e-30 { continue; }

            let mut mu_sum = 0.0;
            for jm in 0..=n_mu {
                let mu = -1.0 + jm as f64 * dmu;
                let wm = if jm == 0 || jm == n_mu { 0.5 } else { 1.0 };
                let kmp2 = k * k + p * p - 2.0 * k * p * mu;
                if kmp2 < 1e-20 { continue; }
                let one_minus_mu2 = 1.0 - mu * mu;
                let p_dot_kmp = k * p * mu - p * p;
                let tidal = 1.0 - p_dot_kmp * p_dot_kmp / (p * p * kmp2);
                mu_sum += wm * one_minus_mu2 * tidal;
            }
            integral += wp * p.powi(3) * pl_p * mu_sum * dmu;
        }
        integral *= dlp;

        let p13_k = pl_k / (4.0 * PI * PI) * integral;
        total += wk * k.powi(3) / (2.0 * PI * PI) * wth_k * wth_k * p13_k;
    }
    total * dlk
}

/// P15 factorised: 2*P15 ~ C_15 * int |W|^2 P_L * Sigma13^2 dk.
/// Sigma13(k) = P13(k) / P_L(k) = raw kernel integral at each k.
pub fn sigma2_p15_factorised(
    cosmo: &Cosmology, r: f64, b_2p13: f64,
    params: &IntegrationParams,
) -> f64 {
    let n_k = params.n_k;
    let n_p = params.n_p;
    let n_mu = params.n_mu;
    let lk_max_eff = (20.0 / r + 1.0).ln().min(params.ln_k_max);
    let dlk = (lk_max_eff - params.ln_k_min) / n_k as f64;
    let dlp = (params.ln_k_max - params.ln_k_min) / n_p as f64;
    let dmu = 2.0 / n_mu as f64;

    let b_half = b_2p13 / 2.0;
    let c_15 = 2.0; // factorisation coefficient

    let mut total = 0.0;
    for ik in 0..=n_k {
        let k = (params.ln_k_min + ik as f64 * dlk).exp();
        let wk = if ik == 0 || ik == n_k { 0.5 } else { 1.0 };
        let wth_k = top_hat(k * r);
        if wth_k.abs() < 1e-15 { continue; }
        let pl_k = cosmo.p_lin(k);
        if pl_k < 1e-30 { continue; }

        // Compute Sigma13_raw(k)
        let mut integral = 0.0;
        for ip in 0..=n_p {
            let p = (params.ln_k_min + ip as f64 * dlp).exp();
            let wp = if ip == 0 || ip == n_p { 0.5 } else { 1.0 };
            let pl_p = cosmo.p_lin(p);
            if pl_p < 1e-30 { continue; }
            let mut mu_sum = 0.0;
            for jm in 0..=n_mu {
                let mu = -1.0 + jm as f64 * dmu;
                let wm = if jm == 0 || jm == n_mu { 0.5 } else { 1.0 };
                let kmp2 = k * k + p * p - 2.0 * k * p * mu;
                if kmp2 < 1e-20 { continue; }
                let one_minus_mu2 = 1.0 - mu * mu;
                let p_dot_kmp = k * p * mu - p * p;
                let tidal = 1.0 - p_dot_kmp * p_dot_kmp / (p * p * kmp2);
                mu_sum += wm * one_minus_mu2 * tidal;
            }
            integral += wp * p.powi(3) * pl_p * mu_sum * dmu;
        }
        let sigma13 = b_half * integral * dlp / (4.0 * PI * PI);

        total += wk * k.powi(3) / (2.0 * PI * PI) * wth_k * wth_k
            * pl_k * sigma13 * sigma13;
    }
    c_15 * total * dlk
}

// ═══════════════════════════════════════════════════════════════════════════
// Generic triple-W bispectrum integrals for the full kNN programme
// ═══════════════════════════════════════════════════════════════════════════

/// Volume-averaged correlation function: ξ̄(R) = ∫ k³/(2π²) W(kR) P_L(k) d ln k.
/// Single power of W (not W²). This is the D-kNN first cumulant (mean excess).
pub fn xi_bar_ws(r: f64, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    let n = ip.n_k;
    let dlk = (ip.ln_k_max - ip.ln_k_min) / n as f64;
    let mut s = 0.0;
    for i in 0..=n {
        let k = (ip.ln_k_min + i as f64 * dlk).exp();
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        let wt = top_hat(k * r);
        // Note: single W, not W²
        s += w * k.powi(3) / (2.0 * PI * PI) * wt * ws.p_lin(k);
    }
    s * dlk
}

/// Bispectrum kernel identifiers for the generic triple-W integral.
#[derive(Clone, Copy, Debug)]
pub enum BispecKernel {
    /// F2(k1,k2) — gravitational nonlinearity (tree-level bispectrum)
    F2,
    /// F2^(J) = (2/7)(1-μ²) — Jacobian-space kernel
    F2J,
    /// 1 — the b₂ (local quadratic bias) piece: B^(δ²) = PL(k1)PL(k2)
    Delta2,
    /// S2(k1,k2) = (k̂1·k̂2)² - 1/3 — tidal bias bs² piece
    S2,
    /// (k1²+k2²) F2(k1,k2) — derivative bias b_{∇²δ}
    Nabla2Delta,
    /// k12² = |k1+k2|² — EFT counterterm cs²
    Cs2,
}

/// Standard F2 kernel: F2(k1,k2) = 5/7 + μ/2(k1/k2 + k2/k1) + 2μ²/7.
#[inline(always)]
fn f2_kernel(k1: f64, k2: f64, mu: f64) -> f64 {
    5.0/7.0 + mu/2.0 * (k1/k2 + k2/k1) + 2.0/7.0 * mu * mu
}

/// S2 tidal kernel: S2(k1,k2) = (k̂1·k̂2)² - 1/3 = μ² - 1/3.
#[inline(always)]
fn s2_kernel(mu: f64) -> f64 {
    mu * mu - 1.0/3.0
}

/// Evaluate the kernel K(k1, k2, μ) for a given BispecKernel variant.
#[inline(always)]
fn eval_bispec_kernel(kernel: BispecKernel, k1: f64, k2: f64, mu: f64, k12_sq: f64) -> f64 {
    match kernel {
        BispecKernel::F2 => f2_kernel(k1, k2, mu),
        BispecKernel::F2J => 2.0/7.0 * (1.0 - mu * mu),
        BispecKernel::Delta2 => 1.0,
        BispecKernel::S2 => s2_kernel(mu),
        BispecKernel::Nabla2Delta => (k1*k1 + k2*k2) * f2_kernel(k1, k2, mu),
        BispecKernel::Cs2 => k12_sq,
    }
}

/// Generic triple-W bispectrum integral:
///
///   I(R; K) = ∫∫ K(k1,k2,μ) P_L(k1) P_L(k2) W(k1·R) W(k2·R) W(k12·R)
///             × k1² k2² dk1 dk2 dμ / (2π)⁴
///
/// where k12 = |k1+k2| = sqrt(k1²+k2²+2k1k2μ).
///
/// This is the building block for all tree-level bispectrum observables:
///   S3^(R) = b1³ I(R; F2) + b1² b2 I(R; δ²) + b1² bs² I(R; S2) + ...
///   S3^(R)_J = I(R; F2J)   [matter field in J-space]
///
/// The factor of 3 (for cyclic permutations) is NOT included — caller adds it.
pub fn triple_w_integral(
    r: f64, kernel: BispecKernel, ws: &Workspace, ip: &IntegrationParams,
) -> f64 {
    use rayon::prelude::*;
    let (nk, np, nmu) = (ip.n_k, ip.n_p, ip.n_mu);
    let lke = (20.0 / r + 1.0).ln().min(ip.ln_k_max);
    let dlk1 = (lke - ip.ln_k_min) / nk as f64;
    let dlk2 = (lke - ip.ln_k_min) / np as f64;
    let dmu = 2.0 / nmu as f64;

    let total: f64 = (0..=nk).into_par_iter().map(|ik| {
        let k1 = (ip.ln_k_min + ik as f64 * dlk1).exp();
        let wk1 = if ik == 0 || ik == nk { 0.5 } else { 1.0 };
        let pl1 = ws.p_lin(k1);
        if pl1 < 1e-30 { return 0.0; }
        let w1 = top_hat(k1 * r);
        let k1_sq = k1 * k1;

        let mut acc = 0.0;
        for ip2 in 0..=np {
            let k2 = (ip.ln_k_min + ip2 as f64 * dlk2).exp();
            let wk2 = if ip2 == 0 || ip2 == np { 0.5 } else { 1.0 };
            let pl2 = ws.p_lin(k2);
            if pl2 < 1e-30 { continue; }
            let w2 = top_hat(k2 * r);
            let k2_sq = k2 * k2;
            let pp = pl1 * pl2;

            let mut mu_sum = 0.0;
            for jm in 0..=nmu {
                let mu = -1.0 + jm as f64 * dmu;
                let wm = if jm == 0 || jm == nmu { 0.5 } else { 1.0 };

                let k12_sq = k1_sq + k2_sq + 2.0 * k1 * k2 * mu;
                if k12_sq < 1e-20 { continue; }
                let k12 = k12_sq.sqrt();
                let w12 = top_hat(k12 * r);

                let kern = eval_bispec_kernel(kernel, k1, k2, mu, k12_sq);
                mu_sum += wm * kern * w1 * w2 * w12 * pp;
            }
            acc += wk1 * wk2 * k1 * k1_sq * k2 * k2_sq * mu_sum * dmu;
        }
        acc
    }).sum();
    total * dlk1 * dlk2 / (4.0 * PI.powi(4))
}

/// Tree-level skewness S₃(R) of the matter field (no bias).
/// Uses the standard F2 kernel with cyclic symmetry factor × 3.
pub fn s3_tree_matter(r: f64, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    // B_tree = 2 F2(k1,k2) PL(k1) PL(k2) + cyclic
    // S3 = ∫ B_tree W1 W2 W12 d³k1 d³k2 / (2π)⁶
    // = 2 × 3 × I(R; F2)  [factor 2 from B = 2 F2 PL PL, factor 3 from cyclic]
    6.0 * triple_w_integral(r, BispecKernel::F2, ws, ip)
}

/// Tree-level skewness of the Jacobian field S₃^(J)(R).
/// Uses F2^(J) = (2/7)(1-μ²).
pub fn s3_tree_jacobian(r: f64, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    6.0 * triple_w_integral(r, BispecKernel::F2J, ws, ip)
}

/// Compute all bias-operator bispectrum integrals at once for a given R.
/// Returns (I_F2, I_delta2, I_s2, I_nabla, I_cs2) — the five independent
/// building blocks from which the full tracer S₃ is assembled:
///
///   S₃ = b₁³ (6·I_F2) + b₁²b₂ (6·I_δ²) + b₁²bs² (6·I_s²)
///       + b₁²b_{∇²δ} (6·I_∇) + b₁²cs² (6·I_cs) + ...
pub fn s3_bias_integrals(
    r: f64, ws: &Workspace, ip: &IntegrationParams,
) -> BispecIntegrals {
    BispecIntegrals {
        i_f2:    triple_w_integral(r, BispecKernel::F2, ws, ip),
        i_f2j:   triple_w_integral(r, BispecKernel::F2J, ws, ip),
        i_delta2: triple_w_integral(r, BispecKernel::Delta2, ws, ip),
        i_s2:    triple_w_integral(r, BispecKernel::S2, ws, ip),
        i_nabla: triple_w_integral(r, BispecKernel::Nabla2Delta, ws, ip),
        i_cs2:   triple_w_integral(r, BispecKernel::Cs2, ws, ip),
    }
}

/// All bispectrum building blocks at a single scale R.
#[derive(Clone, Debug)]
pub struct BispecIntegrals {
    /// Gravitational F2 kernel integral (matter bispectrum)
    pub i_f2: f64,
    /// Jacobian F2^(J) kernel integral
    pub i_f2j: f64,
    /// b₂ (local quadratic bias) integral
    pub i_delta2: f64,
    /// bs² (tidal bias) integral  
    pub i_s2: f64,
    /// b_{∇²δ} (derivative bias) integral
    pub i_nabla: f64,
    /// cs² (EFT counterterm) integral
    pub i_cs2: f64,
}

impl BispecIntegrals {
    /// Assemble the tree-level tracer skewness from bias parameters.
    /// S₃ = 6 × [b₁³ I_F2 + b₁²b₂ I_δ² + b₁²bs² I_s² + b₁²b_∇ I_∇ + b₁²cs² I_cs]
    pub fn s3_tracer(&self, b1: f64, b2: f64, bs2: f64, b_nabla: f64, cs2: f64) -> f64 {
        6.0 * (b1.powi(3) * self.i_f2
             + b1 * b1 * b2 * self.i_delta2
             + b1 * b1 * bs2 * self.i_s2
             + b1 * b1 * b_nabla * self.i_nabla
             + b1 * b1 * cs2 * self.i_cs2)
    }

    /// S₃ for the matter field (b₁=1, all other bias=0).
    pub fn s3_matter(&self) -> f64 {
        6.0 * self.i_f2
    }

    /// S₃ for the Jacobian (J-space).
    pub fn s3_jacobian(&self) -> f64 {
        6.0 * self.i_f2j
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Redshift-space distortions (RSD)
// ═══════════════════════════════════════════════════════════════════════════

/// RSD parameters. When present, all kernels acquire Kaiser factors.
#[derive(Clone, Copy, Debug)]
pub struct RsdParams {
    /// Growth rate f = d ln D / d ln a ≈ Ωm^0.55.
    pub f: f64,
    /// Number of Gauss-Legendre points for ẑ-averaging (monopole).
    /// 12 is sufficient for polynomial μ_z dependence up to degree 8.
    pub n_los: usize,
}

impl RsdParams {
    pub fn new(f: f64) -> Self { RsdParams { f, n_los: 12 } }

    /// Compute f from Omega_m at a given redshift (LCDM approximation).
    pub fn from_cosmology(omega_m: f64, z: f64) -> Self {
        let a = 1.0 / (1.0 + z);
        let om_z = omega_m / (omega_m + (1.0 - omega_m) * a.powi(3));
        RsdParams::new(om_z.powf(0.55))
    }
}

/// Tree-level Kaiser enhancement factor for σ²_lin:
///   σ²_s = (1 + 2f/3 + f²/5) × σ²_real
#[inline]
pub fn kaiser_sigma2(f: f64) -> f64 {
    1.0 + 2.0 * f / 3.0 + f * f / 5.0
}

/// Tree-level Kaiser enhancement for ξ̄:
///   ξ̄_s = (1 + f/3) × ξ̄_real
/// (only the monopole component of the Kaiser factor integrated with single W)
#[inline]
pub fn kaiser_xibar(f: f64) -> f64 {
    1.0 + f / 3.0
}

/// Redshift-space F2 kernel (Zel'dovich piece):
///   F2_Zel,s(k1,k2,μ12; μz1,μz2, f)
///     = ½[(1+fμz1²)(1+fμz2²) - (μ12 + f μz1 μz2)²]
#[inline]
fn f2_zel_rsd(mu12: f64, mu_z1: f64, mu_z2: f64, f: f64) -> f64 {
    let a = (1.0 + f * mu_z1 * mu_z1) * (1.0 + f * mu_z2 * mu_z2);
    let b = mu12 + f * mu_z1 * mu_z2;
    0.5 * (a - b * b)
}

/// Redshift-space F2 full kernel (Zel'dovich + 2LPT):
///   F2_full,s = F2_Zel,s + (1+f μz12²) × (-3/7) × (1-μ12²)
///
/// mu_z12 = ẑ · k̂12 where k12 = k1 + k2.
#[inline]
fn f2_full_rsd(mu12: f64, mu_z1: f64, mu_z2: f64, mu_z12: f64, f: f64) -> f64 {
    f2_zel_rsd(mu12, mu_z1, mu_z2, f)
        + (1.0 + f * mu_z12 * mu_z12) * (-3.0 / 7.0) * (1.0 - mu12 * mu12)
}

/// Compute μ_z for each wavevector given the integration geometry.
///
/// We work in coordinates where k1 is along the z-axis and k2 is in the
/// x-z plane. The line-of-sight ẑ has direction (sin θ_z cos φ_z, sin θ_z sin φ_z, cos θ_z).
///
/// Returns (mu_z1, mu_z2, mu_z12) for given (k1, k2, mu12, cos_theta_z, phi_z).
#[inline]
fn compute_mu_z(
    k1: f64, k2: f64, mu12: f64,
    cos_theta_z: f64, phi_z: f64,
) -> (f64, f64, f64) {
    let sin_theta_z = (1.0 - cos_theta_z * cos_theta_z).max(0.0).sqrt();
    // k1 along z: k̂1 = (0, 0, 1)
    let mu_z1 = cos_theta_z;
    // k2 in x-z plane: k̂2 = (sin α, 0, cos α) where cos α = mu12
    let sin_alpha = (1.0 - mu12 * mu12).max(0.0).sqrt();
    let mu_z2 = sin_alpha * sin_theta_z * phi_z.cos() + mu12 * cos_theta_z;
    // k12 = k1 + k2
    let k12x = k2 * sin_alpha;
    let k12y = 0.0;
    let k12z = k1 + k2 * mu12;
    let k12 = (k12x * k12x + k12y * k12y + k12z * k12z).sqrt();
    let mu_z12 = if k12 > 1e-20 {
        (k12x * sin_theta_z * phi_z.cos() + k12z * cos_theta_z) / k12
    } else { 0.0 };
    (mu_z1, mu_z2, mu_z12)
}

/// Gauss-Legendre nodes and weights for n points on [-1, 1].
/// Pre-computed for n = 12 (sufficient for degree-8 polynomials).
fn gauss_legendre_12() -> ([f64; 12], [f64; 12]) {
    let x = [
        -0.981560634246719, -0.904117256370475, -0.769902674194305,
        -0.587317954286617, -0.367831498998180, -0.125233408511469,
         0.125233408511469,  0.367831498998180,  0.587317954286617,
         0.769902674194305,  0.904117256370475,  0.981560634246719,
    ];
    let w = [
        0.047175336386512, 0.106939325995318, 0.160078328543346,
        0.203167426723066, 0.233492536538355, 0.249147045813403,
        0.249147045813403, 0.233492536538355, 0.203167426723066,
        0.160078328543346, 0.106939325995318, 0.047175336386512,
    ];
    (x, w)
}

/// Redshift-space σ²_tree (Kaiser-enhanced, monopole-averaged).
/// This is the exact angular integral, not just the (1+2f/3+f²/5) approximation.
pub fn sigma2_tree_rsd(r: f64, rsd: &RsdParams, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    // For the tree level, the Kaiser formula is exact:
    // σ²_s = (1 + 2f/3 + f²/5) σ²_real
    kaiser_sigma2(rsd.f) * sigma2_tree_ws(r, ws, ip)
}

/// Redshift-space ξ̄ (monopole).
pub fn xi_bar_rsd(r: f64, rsd: &RsdParams, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    kaiser_xibar(rsd.f) * xi_bar_ws(r, ws, ip)
}

/// Redshift-space Doroshkevich baseline σ²_Zel,s(R).
///
/// In redshift space, the Zel'dovich Jacobian is J_s = det(I + G^s_1)
/// where G^s = (I + f ẑẑᵀ) · G. The variance of J_s - 1 smoothed
/// at scale R depends on f and is averaged over ẑ for the monopole.
///
/// At tree level: σ²_Zel,s = S_s × (1 + 2 S_s / 15)²
/// where S_s = (1 + 2f/3 + f²/5) × S_real.
///
/// This is exact: the Doroshkevich formula uses only the eigenvalue
/// variances of the deformation tensor, and the RSD modification
/// changes these variances by the Kaiser factor.
pub fn sigma2_zel_rsd(s_real: f64, f: f64) -> f64 {
    let s_s = kaiser_sigma2(f) * s_real;
    s_s * (1.0 + 2.0 * s_s / 15.0).powi(2)
}

/// Redshift-space triple-W bispectrum integral (monopole-averaged).
///
/// The kernel K(k1,k2,μ12) is replaced by its RSD version which depends
/// on the line-of-sight direction ẑ. The monopole is obtained by averaging
/// over ẑ orientations using Gauss-Legendre quadrature.
///
/// This adds two angular dimensions (θ_z, φ_z) to the integral.
pub fn triple_w_integral_rsd(
    r: f64, kernel: BispecKernel, rsd: &RsdParams,
    ws: &Workspace, ip: &IntegrationParams,
) -> f64 {
    let (nk, np, nmu) = (ip.n_k, ip.n_p, ip.n_mu);
    let lke = (20.0 / r + 1.0).ln().min(ip.ln_k_max);
    let dlk1 = (lke - ip.ln_k_min) / nk as f64;
    let dlk2 = (lke - ip.ln_k_min) / np as f64;
    let dmu = 2.0 / nmu as f64;
    let f = rsd.f;

    // Gauss-Legendre for cos(θ_z)
    let (gl_x, gl_w) = gauss_legendre_12();
    // Uniform grid for φ_z (periodic, trapezoid)
    let n_phi = 16_usize;
    let dphi = 2.0 * PI / n_phi as f64;

    let mut total = 0.0;

    for ik in 0..=nk {
        let k1 = (ip.ln_k_min + ik as f64 * dlk1).exp();
        let wk1 = if ik == 0 || ik == nk { 0.5 } else { 1.0 };
        let pl1 = ws.p_lin(k1);
        if pl1 < 1e-30 { continue; }
        let w1 = top_hat(k1 * r);
        let k1_sq = k1 * k1;

        for ip2 in 0..=np {
            let k2 = (ip.ln_k_min + ip2 as f64 * dlk2).exp();
            let wk2 = if ip2 == 0 || ip2 == np { 0.5 } else { 1.0 };
            let pl2 = ws.p_lin(k2);
            if pl2 < 1e-30 { continue; }
            let w2 = top_hat(k2 * r);
            let k2_sq = k2 * k2;
            let pp = pl1 * pl2;

            for jm in 0..=nmu {
                let mu12 = -1.0 + jm as f64 * dmu;
                let wm = if jm == 0 || jm == nmu { 0.5 } else { 1.0 };

                let k12_sq = k1_sq + k2_sq + 2.0 * k1 * k2 * mu12;
                if k12_sq < 1e-20 { continue; }
                let k12 = k12_sq.sqrt();
                let w12 = top_hat(k12 * r);

                // Average over ẑ for the monopole
                let mut los_avg = 0.0;
                for i_th in 0..12 {
                    let cos_tz = gl_x[i_th];
                    let w_th = gl_w[i_th];
                    for i_phi in 0..n_phi {
                        let phi_z = i_phi as f64 * dphi;
                        let (mu_z1, mu_z2, mu_z12) = compute_mu_z(k1, k2, mu12, cos_tz, phi_z);

                        // Evaluate the RSD kernel
                        let kern = match kernel {
                            BispecKernel::F2 => f2_full_rsd(mu12, mu_z1, mu_z2, mu_z12, f),
                            BispecKernel::F2J => {
                                // J-space RSD: F2_Zel,s (no 2LPT piece)
                                f2_zel_rsd(mu12, mu_z1, mu_z2, f)
                            },
                            BispecKernel::Delta2 => {
                                // b₂ piece: each PL gets Kaiser factor
                                (1.0 + f * mu_z1 * mu_z1) * (1.0 + f * mu_z2 * mu_z2)
                            },
                            BispecKernel::S2 => {
                                (1.0 + f * mu_z1 * mu_z1) * (1.0 + f * mu_z2 * mu_z2)
                                    * s2_kernel(mu12)
                            },
                            BispecKernel::Nabla2Delta => {
                                (k1_sq + k2_sq) * f2_full_rsd(mu12, mu_z1, mu_z2, mu_z12, f)
                            },
                            BispecKernel::Cs2 => {
                                k12_sq * (1.0 + f * mu_z1 * mu_z1) * (1.0 + f * mu_z2 * mu_z2)
                            },
                        };
                        // Weight: GL weight × uniform φ weight × (1/4π for sphere average)
                        los_avg += w_th * kern / (4.0 * PI) * dphi;
                    }
                }
                // los_avg now has the monopole-averaged kernel value

                total += wk1 * wk2 * k1 * k1_sq * k2 * k2_sq * wm * los_avg * w1 * w2 * w12 * pp;
            }
        }
    }
    total * dlk1 * dlk2 * dmu / (4.0 * PI.powi(4))
}

/// Redshift-space tree-level S₃ (matter, monopole-averaged).
pub fn s3_tree_matter_rsd(r: f64, rsd: &RsdParams, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    6.0 * triple_w_integral_rsd(r, BispecKernel::F2, rsd, ws, ip)
}

/// Redshift-space tree-level S₃ (Jacobian, monopole-averaged).
pub fn s3_tree_jacobian_rsd(r: f64, rsd: &RsdParams, ws: &Workspace, ip: &IntegrationParams) -> f64 {
    6.0 * triple_w_integral_rsd(r, BispecKernel::F2J, rsd, ws, ip)
}

/// All RSD bispectrum integrals at once.
pub fn s3_bias_integrals_rsd(
    r: f64, rsd: &RsdParams, ws: &Workspace, ip: &IntegrationParams,
) -> BispecIntegrals {
    BispecIntegrals {
        i_f2:     triple_w_integral_rsd(r, BispecKernel::F2, rsd, ws, ip),
        i_f2j:    triple_w_integral_rsd(r, BispecKernel::F2J, rsd, ws, ip),
        i_delta2: triple_w_integral_rsd(r, BispecKernel::Delta2, rsd, ws, ip),
        i_s2:     triple_w_integral_rsd(r, BispecKernel::S2, rsd, ws, ip),
        i_nabla:  triple_w_integral_rsd(r, BispecKernel::Nabla2Delta, rsd, ws, ip),
        i_cs2:    triple_w_integral_rsd(r, BispecKernel::Cs2, rsd, ws, ip),
    }
}
