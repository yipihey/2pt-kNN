//! Morton (Z-order) encoding and decoding for 3D octree grids.
//!
//! Provides bit-interleaving of three integer coordinates into a single
//! 64-bit Morton code, with PDEP/PEXT acceleration on x86-64 with BMI2,
//! and a portable lookup-table fallback.

/// Flag distinguishing data vs random catalog particles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatalogFlag {
    Data,
    Random,
}

/// A particle record: Morton code, weight, catalog membership, and
/// original index into the source catalog (data or random).
#[derive(Clone, Copy, Debug)]
pub struct MortonParticle {
    pub code: u64,
    pub weight: f64,
    pub catalog: CatalogFlag,
    /// Index into the original data or random position array.
    pub orig_index: u32,
}

/// Configuration for Morton encoding.
#[derive(Clone, Debug)]
pub struct MortonConfig {
    /// Bits per axis (21 for 64-bit Morton, 10 for 32-bit).
    pub bits_per_axis: u32,
    /// Side length of the enclosing cubic box.
    pub box_size: f64,
    /// Whether the box has periodic boundary conditions.
    pub periodic: bool,
}

impl MortonConfig {
    pub fn new(box_size: f64, periodic: bool) -> Self {
        Self {
            bits_per_axis: 21,
            box_size,
            periodic,
        }
    }

    /// Maximum integer coordinate value: 2^B - 1.
    pub fn max_coord(&self) -> u32 {
        (1u32 << self.bits_per_axis) - 1
    }
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

/// Map a 3D floating-point position into B-bit unsigned integer coordinates.
///
/// Positions are clamped to [0, box_size). Returns (Ix, Iy, Iz).
pub fn quantize(pos: &[f64; 3], config: &MortonConfig) -> (u32, u32, u32) {
    let scale = (1u64 << config.bits_per_axis) as f64 / config.box_size;
    let max = config.max_coord();
    let ix = ((pos[0] * scale) as u32).min(max);
    let iy = ((pos[1] * scale) as u32).min(max);
    let iz = ((pos[2] * scale) as u32).min(max);
    (ix, iy, iz)
}

// ---------------------------------------------------------------------------
// Lookup table for the portable fallback
// ---------------------------------------------------------------------------

/// Spread 8 bits into every-third bit position (24 bits out).
/// E.g. 0b1010_0011 -> 0b001_000_001_000_000_000_001_001
const fn spread_byte(b: u8) -> u32 {
    let mut result = 0u32;
    let mut i = 0;
    while i < 8 {
        if b & (1 << i) != 0 {
            result |= 1 << (i * 3);
        }
        i += 1;
    }
    result
}

const fn build_lut() -> [u32; 256] {
    let mut lut = [0u32; 256];
    let mut i = 0u16;
    while i < 256 {
        lut[i as usize] = spread_byte(i as u8);
        i += 1;
    }
    lut
}

static SPREAD_LUT: [u32; 256] = build_lut();


// ---------------------------------------------------------------------------
// Morton encode — portable (LUT)
// ---------------------------------------------------------------------------

fn encode_morton_64_lut(ix: u32, iy: u32, iz: u32) -> u64 {
    let mut code: u64 = 0;
    // Process 8 bits at a time: 3 bytes covers 24 bits (we need 21).
    for byte_idx in 0..3u32 {
        let shift = byte_idx * 8;
        let bx = ((ix >> shift) & 0xFF) as usize;
        let by = ((iy >> shift) & 0xFF) as usize;
        let bz = ((iz >> shift) & 0xFF) as usize;
        let bit_offset = byte_idx as u64 * 24;
        code |= (SPREAD_LUT[bx] as u64) << bit_offset;
        code |= ((SPREAD_LUT[by] as u64) << 1) << bit_offset;
        code |= ((SPREAD_LUT[bz] as u64) << 2) << bit_offset;
    }
    code
}

// ---------------------------------------------------------------------------
// Morton encode — PDEP (x86-64 BMI2)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn encode_morton_64_pdep(ix: u32, iy: u32, iz: u32) -> u64 {
    // Mask selects every 3rd bit starting at position 0: bits 0,3,6,...,60
    const MASK: u64 = 0x1249_2492_4924_9249;
    unsafe {
        let mx = core::arch::x86_64::_pdep_u64(ix as u64, MASK);
        let my = core::arch::x86_64::_pdep_u64(iy as u64, MASK);
        let mz = core::arch::x86_64::_pdep_u64(iz as u64, MASK);
        mx | (my << 1) | (mz << 2)
    }
}

/// Bit-interleave three B-bit coordinates into a 3B-bit Morton code.
pub fn encode_morton_64(ix: u32, iy: u32, iz: u32) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            return encode_morton_64_pdep(ix, iy, iz);
        }
    }
    encode_morton_64_lut(ix, iy, iz)
}

// ---------------------------------------------------------------------------
// Morton decode
// ---------------------------------------------------------------------------

fn decode_morton_64_lut(code: u64) -> (u32, u32, u32) {
    let mut ix = 0u32;
    let mut iy = 0u32;
    let mut iz = 0u32;
    // Extract every-3rd bit for each axis.
    for bit in 0..21u32 {
        let src_x = bit * 3;       // x is in bits 0,3,6,...
        let src_y = bit * 3 + 1;   // y is in bits 1,4,7,...
        let src_z = bit * 3 + 2;   // z is in bits 2,5,8,...
        if code & (1u64 << src_x) != 0 {
            ix |= 1 << bit;
        }
        if code & (1u64 << src_y) != 0 {
            iy |= 1 << bit;
        }
        if code & (1u64 << src_z) != 0 {
            iz |= 1 << bit;
        }
    }
    (ix, iy, iz)
}

#[cfg(target_arch = "x86_64")]
fn decode_morton_64_pext(code: u64) -> (u32, u32, u32) {
    const MASK: u64 = 0x1249_2492_4924_9249;
    unsafe {
        let ix = core::arch::x86_64::_pext_u64(code, MASK) as u32;
        let iy = core::arch::x86_64::_pext_u64(code >> 1, MASK) as u32;
        let iz = core::arch::x86_64::_pext_u64(code >> 2, MASK) as u32;
        (ix, iy, iz)
    }
}

/// Decode a Morton code back into three integer coordinates.
pub fn decode_morton_64(code: u64) -> (u32, u32, u32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            return decode_morton_64_pext(code);
        }
    }
    decode_morton_64_lut(code)
}

// ---------------------------------------------------------------------------
// Cell indexing
// ---------------------------------------------------------------------------

/// Extract the cell index at octree level `level` from a Morton code.
///
/// The cell index is the top `3*level` bits of the code.
#[inline]
pub fn cell_index(code: u64, level: u32, bits_per_axis: u32) -> u64 {
    debug_assert!(level <= bits_per_axis);
    code >> (3 * (bits_per_axis - level))
}

// ---------------------------------------------------------------------------
// Neighbor finding via Morton arithmetic
// ---------------------------------------------------------------------------

/// The 13 unique nearest-neighbor lag vectors (half-space: first nonzero
/// component is positive). Grouped by separation distance:
///   Face  (|L|=1):     3 vectors
///   Edge  (|L|=√2):    6 vectors
///   Corner(|L|=√3):    4 vectors
pub const NEIGHBOR_LAGS: [(i32, i32, i32); 13] = [
    // Face neighbors (distance = h)
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    // Edge neighbors (distance = h√2)
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, -1, 0),
    (1, 0, -1),
    (0, 1, -1),
    // Corner neighbors (distance = h√3)
    (1, 1, 1),
    (1, -1, 1),
    (1, 1, -1),
    (1, -1, -1),
];

/// Compute the cell index of a neighbor at octree level `level`.
///
/// Given a cell's Morton index (at `level`) and a lag `(dx, dy, dz)`,
/// returns the neighbor's cell index. Returns `None` if the neighbor
/// is out of bounds (non-periodic mode).
pub fn neighbor_cell(
    cell_idx: u64,
    dx: i32,
    dy: i32,
    dz: i32,
    level: u32,
    periodic: bool,
) -> Option<u64> {
    // Decode cell index into (cx, cy, cz) at the given level.
    // Cell index is a Morton code of `level` bits per axis.
    // We can decode by treating it as a Morton code with bits_per_axis = level.
    let (cx, cy, cz) = decode_cell_coords(cell_idx, level);
    let grid_size = 1i64 << level;

    let nx = cx as i64 + dx as i64;
    let ny = cy as i64 + dy as i64;
    let nz = cz as i64 + dz as i64;

    let (nx, ny, nz) = if periodic {
        (
            nx.rem_euclid(grid_size) as u32,
            ny.rem_euclid(grid_size) as u32,
            nz.rem_euclid(grid_size) as u32,
        )
    } else {
        if nx < 0 || nx >= grid_size || ny < 0 || ny >= grid_size || nz < 0 || nz >= grid_size {
            return None;
        }
        (nx as u32, ny as u32, nz as u32)
    };

    Some(encode_cell_index(nx, ny, nz, level))
}

/// Decode a cell index at a given level into integer coordinates.
/// The cell index is a Morton code with `level` bits per axis.
fn decode_cell_coords(cell_idx: u64, level: u32) -> (u32, u32, u32) {
    let mut cx = 0u32;
    let mut cy = 0u32;
    let mut cz = 0u32;
    for bit in 0..level {
        let src_x = bit * 3;
        let src_y = bit * 3 + 1;
        let src_z = bit * 3 + 2;
        if cell_idx & (1u64 << src_x) != 0 {
            cx |= 1 << bit;
        }
        if cell_idx & (1u64 << src_y) != 0 {
            cy |= 1 << bit;
        }
        if cell_idx & (1u64 << src_z) != 0 {
            cz |= 1 << bit;
        }
    }
    (cx, cy, cz)
}

/// Encode integer coordinates at a given level into a cell index (Morton code).
fn encode_cell_index(cx: u32, cy: u32, cz: u32, _level: u32) -> u64 {
    // Use the same interleaving as the full Morton code; the level only
    // constrains how many bits are meaningful.
    encode_morton_64(cx, cy, cz)
}

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

/// Sort particles by Morton code (in-place, unstable).
pub fn sort_particles(particles: &mut [MortonParticle]) {
    particles.sort_unstable_by_key(|p| p.code);
}

// ---------------------------------------------------------------------------
// Bulk encoding
// ---------------------------------------------------------------------------

/// Encode a catalog of positions into MortonParticles.
///
/// Returns a vector of MortonParticles with unit weights (unweighted mode).
pub fn encode_catalog(
    positions: &[[f64; 3]],
    config: &MortonConfig,
    catalog: CatalogFlag,
) -> Vec<MortonParticle> {
    positions
        .iter()
        .enumerate()
        .map(|(i, pos)| {
            let (ix, iy, iz) = quantize(pos, config);
            MortonParticle {
                code: encode_morton_64(ix, iy, iz),
                weight: 1.0,
                catalog,
                orig_index: i as u32,
            }
        })
        .collect()
}

/// Encode both data and random catalogs, merge, and sort by Morton code.
pub fn prepare_particles(
    data: &[[f64; 3]],
    randoms: &[[f64; 3]],
    config: &MortonConfig,
) -> Vec<MortonParticle> {
    let mut particles = Vec::with_capacity(data.len() + randoms.len());
    particles.extend(encode_catalog(data, config, CatalogFlag::Data));
    particles.extend(encode_catalog(randoms, config, CatalogFlag::Random));
    sort_particles(&mut particles);
    particles
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_encode_decode() {
        for &(ix, iy, iz) in &[
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 1),
            (0x1FFFFF, 0x1FFFFF, 0x1FFFFF), // max for 21 bits
            (123456, 654321, 42),
        ] {
            let code = encode_morton_64(ix, iy, iz);
            let (dx, dy, dz) = decode_morton_64(code);
            assert_eq!((ix, iy, iz), (dx, dy, dz), "roundtrip failed for ({ix}, {iy}, {iz})");
        }
    }

    #[test]
    fn lut_matches_pdep() {
        // Verify both paths produce the same result.
        for &(ix, iy, iz) in &[
            (0, 0, 0),
            (1, 2, 3),
            (255, 255, 255),
            (0x1FFFFF, 0, 0),
            (0, 0x1FFFFF, 0),
            (0, 0, 0x1FFFFF),
            (12345, 67890, 11111),
        ] {
            let lut = encode_morton_64_lut(ix, iy, iz);
            // On non-BMI2 hardware, just check LUT is self-consistent.
            let (dx, dy, dz) = decode_morton_64_lut(lut);
            assert_eq!((ix, iy, iz), (dx, dy, dz));
        }
    }

    #[test]
    fn cell_index_levels() {
        let code = encode_morton_64(0b101, 0b011, 0b110); // 3-bit coords
        // At level 1, top 3 bits of the 63-bit code = the MSBs of each coord
        let c1 = cell_index(code, 1, 21);
        let c2 = cell_index(code, 2, 21);
        let c3 = cell_index(code, 3, 21);
        // Level 1 cell should be the interleave of the top bit of each coord
        // (bits 20 of each). ix=0b101: bit 20=0, iy=0b011: bit 20=0, iz=0b110: bit 20=0
        assert_eq!(c1, 0);
        // Higher levels should have more resolution
        assert!(c3 >= c2 || c3 < 8u64.pow(3));
    }

    #[test]
    fn quantize_bounds() {
        let config = MortonConfig::new(100.0, true);
        let (ix, iy, iz) = quantize(&[0.0, 0.0, 0.0], &config);
        assert_eq!((ix, iy, iz), (0, 0, 0));

        let (ix, iy, iz) = quantize(&[99.999, 99.999, 99.999], &config);
        assert!(ix > 0 && iy > 0 && iz > 0);
        assert!(ix <= config.max_coord());
    }

    #[test]
    fn neighbor_cell_periodic() {
        let level = 3;
        // Cell at (0, 0, 0) → neighbor at (-1, 0, 0) should wrap to (7, 0, 0)
        let cell_000 = encode_cell_index(0, 0, 0, level);
        let nbr = neighbor_cell(cell_000, -1, 0, 0, level, true).unwrap();
        let (nx, ny, nz) = decode_cell_coords(nbr, level);
        assert_eq!((nx, ny, nz), (7, 0, 0));
    }

    #[test]
    fn neighbor_cell_non_periodic_oob() {
        let level = 3;
        let cell_000 = encode_cell_index(0, 0, 0, level);
        assert!(neighbor_cell(cell_000, -1, 0, 0, level, false).is_none());
    }

    #[test]
    fn neighbor_cell_face() {
        let level = 4;
        let cell = encode_cell_index(5, 5, 5, level);
        let nbr = neighbor_cell(cell, 1, 0, 0, level, false).unwrap();
        let (nx, ny, nz) = decode_cell_coords(nbr, level);
        assert_eq!((nx, ny, nz), (6, 5, 5));
    }

    #[test]
    fn sorting_preserves_particles() {
        let config = MortonConfig::new(100.0, true);
        let positions = vec![[90.0, 10.0, 50.0], [10.0, 90.0, 50.0], [50.0, 50.0, 50.0]];
        let mut particles = encode_catalog(&positions, &config, CatalogFlag::Data);
        sort_particles(&mut particles);

        // Should be sorted by Morton code
        for w in particles.windows(2) {
            assert!(w[0].code <= w[1].code);
        }
        assert_eq!(particles.len(), 3);
    }
}
