//! Multipole moments for LCA tree nodes.
//!
//! Stores monopole (total weight), with stubs for future dipole and
//! quadrupole extensions.

/// Axis-aligned bounding box in 3D.
#[derive(Debug, Clone, Copy)]
pub struct BBox3 {
    pub lo: [f64; 3],
    pub hi: [f64; 3],
}

/// Monopole moment: just the total weight.
#[derive(Debug, Clone, Copy, Default)]
pub struct Monopole {
    /// Total weight (sum of particle weights in this subtree).
    pub w: f64,
}

impl Monopole {
    #[inline]
    pub fn accumulate(&mut self, weight: f64) {
        self.w += weight;
    }

    #[inline]
    pub fn merge(&mut self, other: &Monopole) {
        self.w += other.w;
    }
}

impl BBox3 {
    /// Create an empty bounding box (will be expanded by `expand`).
    pub fn empty() -> Self {
        Self {
            lo: [f64::MAX; 3],
            hi: [f64::MIN; 3],
        }
    }

    /// Expand the bounding box to include a point.
    #[inline]
    pub fn expand(&mut self, point: &[f64; 3]) {
        for k in 0..3 {
            if point[k] < self.lo[k] {
                self.lo[k] = point[k];
            }
            if point[k] > self.hi[k] {
                self.hi[k] = point[k];
            }
        }
    }

    /// Center of the bounding box.
    pub fn center(&self) -> [f64; 3] {
        [
            0.5 * (self.lo[0] + self.hi[0]),
            0.5 * (self.lo[1] + self.hi[1]),
            0.5 * (self.lo[2] + self.hi[2]),
        ]
    }

    /// Side lengths along each axis.
    pub fn side_lengths(&self) -> [f64; 3] {
        [
            self.hi[0] - self.lo[0],
            self.hi[1] - self.lo[1],
            self.hi[2] - self.lo[2],
        ]
    }

    /// Volume of the bounding box.
    pub fn volume(&self) -> f64 {
        let s = self.side_lengths();
        s[0] * s[1] * s[2]
    }

    /// Minimum squared distance between any point in this box and any point
    /// in another box.
    pub fn min_dist_sq(&self, other: &BBox3) -> f64 {
        let mut d2 = 0.0;
        for k in 0..3 {
            let gap = (other.lo[k] - self.hi[k]).max(self.lo[k] - other.hi[k]).max(0.0);
            d2 += gap * gap;
        }
        d2
    }

    /// Maximum squared distance between any point in this box and any point
    /// in another box.
    pub fn max_dist_sq(&self, other: &BBox3) -> f64 {
        let mut d2 = 0.0;
        for k in 0..3 {
            let d = (self.hi[k] - other.lo[k]).abs().max((self.lo[k] - other.hi[k]).abs());
            d2 += d * d;
        }
        d2
    }
}

/// Compute tight bounding box and monopole for a particle span.
pub fn compute_bbox_and_monopole(
    positions: &[[f64; 3]],
    weights: &[f64],
    start: usize,
    end: usize,
) -> (BBox3, Monopole) {
    let mut bbox = BBox3::empty();
    let mut mono = Monopole::default();
    for i in start..end {
        bbox.expand(&positions[i]);
        mono.accumulate(weights[i]);
    }
    (bbox, mono)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_expand_and_center() {
        let mut bbox = BBox3::empty();
        bbox.expand(&[1.0, 2.0, 3.0]);
        bbox.expand(&[5.0, 4.0, 1.0]);
        assert_eq!(bbox.lo, [1.0, 2.0, 1.0]);
        assert_eq!(bbox.hi, [5.0, 4.0, 3.0]);
        let c = bbox.center();
        assert!((c[0] - 3.0).abs() < 1e-12);
        assert!((c[1] - 3.0).abs() < 1e-12);
        assert!((c[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn bbox_volume() {
        let bbox = BBox3 {
            lo: [0.0, 0.0, 0.0],
            hi: [2.0, 3.0, 4.0],
        };
        assert!((bbox.volume() - 24.0).abs() < 1e-12);
    }

    #[test]
    fn bbox_min_max_dist() {
        let a = BBox3 {
            lo: [0.0, 0.0, 0.0],
            hi: [1.0, 1.0, 1.0],
        };
        let b = BBox3 {
            lo: [3.0, 0.0, 0.0],
            hi: [4.0, 1.0, 1.0],
        };
        // Min distance: gap along x = 3.0 - 1.0 = 2.0
        assert!((a.min_dist_sq(&b) - 4.0).abs() < 1e-12);
        // Max distance: corners (0,0,0) and (4,1,1) → 16+1+1 = 18
        assert!((a.max_dist_sq(&b) - 18.0).abs() < 1e-12);
    }

    #[test]
    fn monopole_accumulate() {
        let mut m = Monopole::default();
        m.accumulate(1.5);
        m.accumulate(2.5);
        assert!((m.w - 4.0).abs() < 1e-12);
    }

    #[test]
    fn compute_bbox_and_mono() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.5, 1.0, 1.5]];
        let weights = [1.0, 2.0, 3.0];
        let (bbox, mono) = compute_bbox_and_monopole(&positions, &weights, 0, 3);
        assert_eq!(bbox.lo, [0.0, 0.0, 0.0]);
        assert_eq!(bbox.hi, [1.0, 2.0, 3.0]);
        assert!((mono.w - 6.0).abs() < 1e-12);
    }
}
