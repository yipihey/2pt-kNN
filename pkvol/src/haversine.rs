//! Haversine angular distance and unit-vector helpers.
//!
//! All angles are in radians. RA can be any real number; the unit-vector
//! mapping makes RA periodic automatically.

#[inline]
pub fn radec_to_unit(ra: f64, dec: f64) -> [f64; 3] {
    let cd = dec.cos();
    [cd * ra.cos(), cd * ra.sin(), dec.sin()]
}

/// Great-circle (haversine) distance in radians between two RA/Dec pairs.
#[inline]
pub fn haversine(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let dra = ra2 - ra1;
    let ddec = dec2 - dec1;
    let s_dec = (ddec * 0.5).sin();
    let s_ra = (dra * 0.5).sin();
    let h = s_dec * s_dec + dec1.cos() * dec2.cos() * s_ra * s_ra;
    // clamp for fp safety
    2.0 * h.sqrt().min(1.0).asin()
}

/// Angular distance from a chord distance between two unit vectors.
#[inline]
pub fn chord_to_theta(chord: f64) -> f64 {
    2.0 * (0.5 * chord).min(1.0).asin()
}

/// Chord distance (Euclidean L2 between unit vectors) for a given angle.
#[inline]
pub fn theta_to_chord(theta: f64) -> f64 {
    2.0 * (0.5 * theta).sin()
}

/// Angular distance between two unit vectors using their dot product.
/// Slightly more stable than haversine for very small / very large theta.
#[inline]
pub fn theta_from_dot(u: &[f64; 3], v: &[f64; 3]) -> f64 {
    let dot = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0);
    dot.acos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    #[test]
    fn haversine_matches_dot_for_random_pairs() {
        let cases = [
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, PI / 2.0, 0.0, PI / 2.0),
            (0.0, 0.0, PI, 0.0, PI),
            (0.0, -PI / 2.0, 0.0, PI / 2.0, PI),
            (0.0, 0.0, 0.0, 1e-9, 1e-9),
        ];
        for (ra1, dec1, ra2, dec2, expected) in cases {
            let h = haversine(ra1, dec1, ra2, dec2);
            assert_abs_diff_eq!(h, expected, epsilon = 1e-9);
            let u = radec_to_unit(ra1, dec1);
            let v = radec_to_unit(ra2, dec2);
            let td = theta_from_dot(&u, &v);
            assert_abs_diff_eq!(h, td, epsilon = 1e-7);
        }
    }

    #[test]
    fn chord_round_trip() {
        for &theta in &[0.0, 1e-6, 0.001, 0.1, 1.0, 2.5] {
            let c = theta_to_chord(theta);
            assert_abs_diff_eq!(chord_to_theta(c), theta, epsilon = 1e-12);
        }
    }

    #[test]
    fn ra_periodicity_in_unit_vectors() {
        // RA wrap: theta should be small for RA = 0 vs RA = 2*pi.
        let u = radec_to_unit(0.0, 0.3);
        let v = radec_to_unit(2.0 * PI, 0.3);
        let theta = theta_from_dot(&u, &v);
        assert_abs_diff_eq!(theta, 0.0, epsilon = 1e-9);
    }
}
