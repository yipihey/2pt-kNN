// Brute-force kNN compute shader for the twopoint pipeline.
//
// Each thread handles one query point, scanning all data points to find
// the k nearest neighbors. Results are k Euclidean distances per query,
// sorted ascending.
//
// This is the GPU equivalent of tree/mod.rs::nearest_k() — same inputs,
// same outputs, but massively parallel.

struct Params {
    n_data: u32,
    n_queries: u32,
    k_max: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

// Private per-thread heap for k-nearest distances (squared).
// Fixed at k_max=8 — the maximum supported by the pipeline.
// Maintained in sorted order (ascending) for efficient insertion.
var<private> heap: array<f32, 8>;
var<private> heap_size: u32;

@compute @workgroup_size(64)
fn knn_search(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= params.n_queries) {
        return;
    }

    let q = queries[qi].xyz;
    let k = params.k_max;

    // Initialize heap with infinity
    for (var i = 0u; i < 8u; i++) {
        heap[i] = 3.4028235e+38;
    }

    // Scan all data points — the brute-force core
    for (var di = 0u; di < params.n_data; di++) {
        let d = data[di].xyz;
        let diff = q - d;
        let dist_sq = dot(diff, diff);

        // Skip self-pairs (distance ~ 0)
        if (dist_sq < 1e-20) {
            continue;
        }

        // Insert into sorted heap if smaller than current k-th largest
        if (dist_sq < heap[k - 1u]) {
            heap[k - 1u] = dist_sq;

            // Bubble down to maintain ascending sort
            for (var j = k - 1u; j > 0u; j--) {
                if (heap[j] < heap[j - 1u]) {
                    let tmp = heap[j];
                    heap[j] = heap[j - 1u];
                    heap[j - 1u] = tmp;
                } else {
                    break;
                }
            }
        }
    }

    // Write k Euclidean distances (sqrt of squared) to output buffer
    let base = qi * k;
    for (var i = 0u; i < k; i++) {
        results[base + i] = sqrt(heap[i]);
    }
}
