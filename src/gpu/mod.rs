//! GPU-accelerated kNN via WebGPU compute shaders.
//!
//! Provides a drop-in replacement for the CPU kNN query path. The GPU handles
//! the distance computation (the bottleneck), while the statistical pipeline
//! (pair-count density, ξ estimation, CDFs, dilution ladder) runs on CPU.
//!
//! The WGSL shader performs brute-force kNN: for each query point, scan all
//! data points and maintain a sorted heap of the k nearest distances.
//! At N ≤ 500k this runs in seconds on a modern GPU.

use wgpu::util::DeviceExt;

use crate::estimator::{KnnDistances, KnnDistributions};
use crate::validation::KnnBackend;

/// GPU kNN backend using wgpu compute shaders.
pub struct GpuKnn {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// Parameters passed to the WGSL shader as a uniform buffer.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    n_data: u32,
    n_queries: u32,
    k_max: u32,
    _pad: u32,
}

impl GpuKnn {
    /// Initialize the GPU pipeline. Returns None if no GPU adapter is available.
    pub async fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("twopoint-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: 1 << 30, // 1 GB
                        max_buffer_size: 1 << 30,
                        ..wgpu::Limits::downlevel_defaults()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("knn-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("knn.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("knn-bgl"),
            entries: &[
                // params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // data storage (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // queries storage (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // results storage (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("knn-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("knn-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("knn_search"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    /// Run brute-force kNN on the GPU.
    ///
    /// Converts f64 → f32 for GPU, runs the compute shader, converts back.
    /// Returns the same KnnDistributions type as the CPU path.
    pub async fn query_distances(
        &self,
        data: &[[f64; 3]],
        queries: &[[f64; 3]],
        k_max: usize,
    ) -> KnnDistributions {
        let n_data = data.len() as u32;
        let n_queries = queries.len() as u32;
        let k = k_max as u32;

        // Convert f64 positions to f32 vec4 (padded with 0)
        let data_f32: Vec<[f32; 4]> = data
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 0.0])
            .collect();
        let query_f32: Vec<[f32; 4]> = queries
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 0.0])
            .collect();

        let params = GpuParams {
            n_data,
            n_queries,
            k_max: k,
            _pad: 0,
        };

        // Create GPU buffers
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let data_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("data"),
            contents: bytemuck::cast_slice(&data_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let query_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("queries"),
            contents: bytemuck::cast_slice(&query_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let result_size = (n_queries * k) as u64 * 4; // f32 per distance
        let result_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results"),
            size: result_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: result_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("knn-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: query_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let workgroups = (n_queries + 63) / 64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("knn-encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("knn-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&result_buf, 0, &staging_buf, 0, result_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv().expect("GPU readback failed");

        let result_data = slice.get_mapped_range();
        let distances_f32: &[f32] = bytemuck::cast_slice(&result_data);

        // Convert f32 distances back to f64 and package as KnnDistributions
        let per_query: Vec<KnnDistances> = (0..n_queries as usize)
            .map(|qi| {
                let start = qi * k_max;
                let end = start + k_max;
                let distances: Vec<f64> = distances_f32[start..end]
                    .iter()
                    .map(|&d| d as f64)
                    .collect();
                KnnDistances { distances }
            })
            .collect();

        drop(result_data);
        staging_buf.unmap();

        KnnDistributions {
            per_query,
            k_max,
        }
    }
}

// Note: GpuKnn implements KnnBackend asynchronously. Since KnnBackend is sync,
// we provide a wrapper that blocks on the async GPU call. In WASM, the async
// version is called directly from the async wasm entry point.
impl KnnBackend for GpuKnn {
    fn query_distances(
        &self,
        data: &[[f64; 3]],
        queries: &[[f64; 3]],
        k_max: usize,
    ) -> KnnDistributions {
        // On native, use pollster to block. On WASM, this path shouldn't
        // be used — call query_distances directly in async context.
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(self.query_distances(data, queries, k_max))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (data, queries, k_max);
            // WASM can't block — this is a fallback that panics.
            // The WASM entry point should use the async API directly.
            panic!("Use async GpuKnn::query_distances in WASM context");
        }
    }
}
