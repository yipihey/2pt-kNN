//! Cosmology module: linear power spectrum from syren-new (Bartlett et al. 2023).
//!
//! P_L(k,a) = EH_nw(k) * D(k,a)² * F(k) * R(a) * S(k)
//! All factors are closed-form algebraic expressions.

use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Cosmology {
    pub omega_m: f64,
    pub omega_b: f64,
    pub h: f64,
    pub n_s: f64,
    pub sigma8: f64,
    pub m_nu: f64,
    pub w0: f64,
    pub wa: f64,
    pub a_s: f64,
}

impl Cosmology {
    pub fn new(omega_m: f64, omega_b: f64, h: f64, n_s: f64, sigma8: f64) -> Self {
        Self::with_extensions(omega_m, omega_b, h, n_s, sigma8, 0.06, -1.0, 0.0)
    }

    pub fn with_extensions(
        omega_m: f64, omega_b: f64, h: f64, n_s: f64, sigma8: f64,
        m_nu: f64, w0: f64, wa: f64,
    ) -> Self {
        let a_s = sigma8_to_as(sigma8, omega_m, omega_b, h, n_s, m_nu, w0, wa);
        Cosmology { omega_m, omega_b, h, n_s, sigma8, m_nu, w0, wa, a_s }
    }

    pub fn planck2018() -> Self { Self::new(0.3153, 0.0493, 0.6736, 0.9649, 0.8111) }

    #[inline]
    pub fn p_lin_at(&self, k: f64, a: f64) -> f64 {
        if k <= 0.0 { return 0.0; }
        eisenstein_hu_nw(k, self)
            * growth_factor(k, self, a).powi(2)
            * log_f_fiducial(k, self).exp()
            * growth_correction_r(self, a)
            * 10.0_f64.powf(log10_s(k, self))
    }

    #[inline]
    pub fn p_lin(&self, k: f64) -> f64 { self.p_lin_at(k, 1.0) }
}

// ── σ8 → As (algebraic fit from Bartlett et al.) ────────────────────────

fn sigma8_to_as(s8: f64, om: f64, ob: f64, h: f64, ns: f64, mnu: f64, w0: f64, wa: f64) -> f64 {
    let c = [0.0187, 2.4891, 12.9495, 0.7527, 2.3685, 1.5062, 1.3057, 0.0885,
             0.1471, 3.4982, 0.006, 19.2779, 11.1463, 1.5433, 7.0578, 2.0564];
    let t1 = c[0] * (-ob*c[1] + om*c[2] + (-c[3]*w0 + (-c[4]*w0 - c[5]*wa).ln()).ln());
    let t2 = om*c[6] + c[7]*mnu + c[8]*ns - (om*c[9] - c[10]*wa).ln();
    let t3 = ob*c[11] - om*c[12] - ns;
    let t4 = -om*c[13] - c[14]*h + c[15]*mnu + ns;
    (s8 / (t1*t2*t3*t4)).powi(2)
}

// ── Eisenstein & Hu no-wiggle ───────────────────────────────────────────

fn eisenstein_hu_nw(k: f64, c: &Cosmology) -> f64 {
    let fb = c.omega_b / c.omega_m;
    let omh2 = c.omega_m * c.h * c.h;
    let obh2 = c.omega_b * c.h * c.h;
    let th = 2.7255 / 2.7;
    let s = 44.5 * (9.83/omh2).ln() / (1.0 + 10.0*obh2.powf(0.75)).sqrt();
    let ag = 1.0 - 0.328*(431.0*omh2).ln()*fb + 0.38*(22.3*omh2).ln()*fb*fb;
    let gam = c.omega_m * c.h * (ag + (1.0-ag)/(1.0 + (0.43*k*c.h*s).powi(4)));
    let q = k * th*th / gam;
    let c0 = 14.2 + 731.0/(1.0+62.5*q);
    let l0 = (2.0*std::f64::consts::E + 1.8*q).ln();
    let tk = l0 / (l0 + c0*q*q);
    2.0*PI*PI/(k*k*k) * (c.a_s*1e-9) * (k*c.h/0.05).powf(c.n_s-1.0)
        * (2.0*k*k*2998.0*2998.0/(5.0*c.omega_m)).powi(2) * tk*tk
}

// ── Growth factor D(k,a) — Carroll/Lahav + E&H neutrino suppression ────

fn growth_factor(k: f64, c: &Cosmology, a: f64) -> f64 {
    let mnu = c.m_nu + 1e-10;
    let z = 1.0/a - 1.0;
    let th4 = (2.7255/2.7_f64).powi(4);
    let zeq = 2.5e4 * c.omega_m * c.h * c.h / th4;

    // Ω(a) and Ω_Λ(a) with w0-wa
    let om_a = c.omega_m * a.powi(-3);
    let ol_a = (1.0 - c.omega_m) * a.powf(-3.0*(1.0 + c.w0 + c.wa))
        * (-3.0*c.wa*(1.0-a)).exp();
    let g2 = om_a + ol_a;
    let omn = om_a / g2;
    let oln = ol_a / g2;

    let d1 = (1.0+zeq)/(1.0+z) * 2.5*omn
        / (omn.powf(4.0/7.0) - oln + (1.0+omn/2.0)*(1.0+oln/70.0));

    // Neutrino free-streaming (Bond+1980, E&H 1997)
    let onu = mnu / (93.14 * c.h * c.h);
    let fc = (c.omega_m - c.omega_b - onu) / c.omega_m;
    let fb = c.omega_b / c.omega_m;
    let fnu = onu / c.omega_m;
    let fcb = fc + fb;
    let pcb = 0.25 * (5.0 - (1.0 + 24.0*fcb).sqrt());
    let nnu = if mnu > 1e-8 { 3.0 } else { 0.0 };
    let th2 = 2.7255/2.7;
    let qnu = k * c.h * th2*th2 / (c.omega_m * c.h * c.h);
    let yfs = 17.2 * fnu * (1.0 + 0.488*fnu.powf(-7.0/6.0)) * (nnu*qnu/fnu).powi(2);

    let dcbnu = (fcb.powf(0.7/pcb) + (d1/(1.0+yfs)).powf(0.7)).powf(pcb/0.7)
        * d1.powf(1.0-pcb);
    dcbnu / (1.0+zeq)
}

// ── Growth correction R(a) ──────────────────────────────────────────────

fn growth_correction_r(c: &Cosmology, a: f64) -> f64 {
    let d = [0.8545, 0.394, 0.7294, 0.5347, 0.4662, 4.6669,
             0.4136, 1.4769, 0.5959, 0.4553, 0.0799, 5.8311,
             5.8014, 6.7085, 0.3445, 1.2498, 0.3756, 0.2136];
    let (om, w0, wa) = (c.omega_m, c.w0, c.wa);
    let p1 = d[0];
    let p2 = -1.0 / (a*d[1] + d[2] + (om*d[3] - a*d[4]) * (-d[5]*w0 - d[6]*wa).ln());
    let n3 = om*d[7] - a*d[8] + (-d[9]*w0 - d[10]*wa).ln();
    let d3 = -a*d[11] + d[12] + d[13]*(om*d[14] + a*d[15] - 1.0)*(d[16]*w0 + d[17]*wa + 1.0);
    let p3 = -n3/d3;
    1.0 + (1.0-a) * (p1 + p2 + p3)
}

// ── BAO correction logF ─────────────────────────────────────────────────

fn log_f_fiducial(k: f64, c: &Cosmology) -> f64 {
    let b = [0.05448654, 0.00379, 0.0396711937097927, 0.127733431568858, 1.35,
        4.053543862744234, 0.0008084539054750851, 1.8852431049189666,
        0.11418372931475675, 3.798, 14.909, 5.56, 15.8274343004709, 0.0230755621512691,
        0.86531976, 0.8425442636372944, 4.553956000000005, 5.116999999999995,
        70.0234239999998, 0.01107, 5.35, 6.421, 134.309, 5.324, 21.532,
        4.741999999999985, 16.68722499999999, 3.078, 16.987, 0.05881491,
        0.0006864690561825617, 195.498, 0.0038454457516892, 0.276696018851544,
        7.385, 12.3960625361899, 0.0134114370723638];
    let (om, ob, h) = (c.omega_m, c.omega_b, c.h);
    let l1 = b[0]*h - b[1];
    let l2 = ((ob*b[2])/(h*h+b[3]).sqrt()).powf(b[4]*om)
        * ((b[5]*k-ob)/(b[6]+(ob-b[7]*k).powi(2)).sqrt()
           * b[8] * (b[9]*k).powf(-b[10]*k)
           * (om*b[11] - (b[12]*k)/(b[13]+ob*ob).sqrt()).cos()
           - b[14]*((b[15]*k)/(1.0+b[16]*k*k).sqrt()-om)
             * (b[17]*h/(1.0+b[18]*k*k).sqrt()).cos());
    let l3 = b[19]*(b[20]*om+b[21]*h-(b[22]*k).ln()+(b[23]*k).powf(-b[24]*k))
        * (b[25]/(1.0+b[26]*k*k).sqrt()).cos();
    let l4 = (b[27]*k).powf(-b[28]*k)
        * (b[29]*k - b[30]*(b[31]*k).ln()/(b[32]+(om-b[33]*h).powi(2)).sqrt())
        * (om*b[34] - (b[35]*k)/(ob*ob+b[36]).sqrt()).cos();
    l1 + l2 + l3 + l4
}

// ── Extended cosmology correction log10(S) ──────────────────────────────

fn log10_s(k: f64, c: &Cosmology) -> f64 {
    let e = [0.2841, 0.1679, 0.0534, 0.0024, 0.1183, 0.3971,
             0.0985, 0.0009, 0.1258, 0.2476, 0.1841, 0.0316,
             0.1385, 0.2825, 0.8098, 0.019, 0.1376, 0.3733];
    let (om,ob,h,mnu,w0,wa) = (c.omega_m,c.omega_b,c.h,c.m_nu,c.w0,c.wa);
    let s = -e[0]*h - e[1]*w0 - e[2]*mnu/(e[3]+k*k).sqrt()
        - (e[4]*h)/(e[5]*h+mnu)
        + e[6]*mnu/(h*(e[7]+(om*e[8]+k).powi(2)).sqrt())
        + (e[9]*ob - e[10]*w0 - e[11]*wa + (e[12]*w0+e[13])/(e[14]*wa+w0))
          / (e[15] + (om + e[16]*(-e[17]*w0).ln()).powi(2)).sqrt();
    s / 10.0  // syren-new convention: log10(S) = raw_sum / 10
}

// ── Utility ─────────────────────────────────────────────────────────────

#[inline]
pub fn top_hat(x: f64) -> f64 {
    if x.abs() < 1e-6 { 1.0 - x*x/10.0 + x.powi(4)/280.0 }
    else { 3.0*(x.sin() - x*x.cos()) / (x*x*x) }
}
