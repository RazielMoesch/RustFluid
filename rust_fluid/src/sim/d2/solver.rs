//! WGPU pipeline ownership and dispatch for the D2Q9 solver.

use crate::gpu::utils::bgl_storage_entry;
use crate::gpu::utils::create_bgl;
use crate::sim::common::precision::Precision;
use crate::sim::common::precision::PrecisionConfig;
use crate::sim::d2::boundary::Boundary2D;
use crate::sim::d2::buffers::SimBuffers2D;
use crate::sim::d2::collision::Collision2D;
use crate::sim::d2::config::SimulationConfig2D;
use crate::sim::d2::lattice::Lattice2D;
use crate::sim::d2::shader::compiler::ShaderCompiler2D;

/// A compiled D2Q9 simulation and its alternating population state.
pub struct Lbm2D {
    pub buffers: SimBuffers2D,
    pub config: SimulationConfig2D,

    init_pipeline: wgpu::ComputePipeline,
    step_even_pipeline: wgpu::ComputePipeline,
    step_odd_pipeline: wgpu::ComputePipeline,
    extract_even_pipeline: wgpu::ComputePipeline,
    extract_odd_pipeline: wgpu::ComputePipeline,

    wg_x: u32,
    wg_y: u32,

    pub step_count: u32,
}

impl Lbm2D {
    pub fn new(
        device: &wgpu::Device,
        config: SimulationConfig2D,
        precision: Precision,
        lattice: &dyn Lattice2D,
        collision: &dyn Collision2D,
        boundaries: &[&dyn Boundary2D],
    ) -> Self {
        let precision_cfg = PrecisionConfig::from(precision);

        let (init_wgsl, step_even_wgsl, step_odd_wgsl, extract_wgsl) =
            ShaderCompiler2D::compile(lattice, collision, boundaries, &precision_cfg);

        let (init_bgl, step_bgl, extract_bgl) = Self::create_bgls(device);

        let buffers = SimBuffers2D::new(
            device,
            config.nx,
            config.ny,
            lattice.q(),
            precision.bytes_per_population(),
            config.num_boundary_configs,
            &init_bgl,
            &step_bgl,
            &extract_bgl,
        );

        let init_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBM2D Init Shader"),
            source: wgpu::ShaderSource::Wgsl(init_wgsl.into()),
        });

        let step_even_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBM2D Step Even Shader"),
            source: wgpu::ShaderSource::Wgsl(step_even_wgsl.into()),
        });

        let step_odd_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBM2D Step Odd Shader"),
            source: wgpu::ShaderSource::Wgsl(step_odd_wgsl.into()),
        });

        let extract_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBM2D Extract Shader"),
            source: wgpu::ShaderSource::Wgsl(extract_wgsl.into()),
        });

        let init_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LBM2D Init Pipeline Layout"),
            bind_group_layouts: &[Some(&init_bgl)],
            immediate_size: 0,
        });

        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM2D Init Pipeline"),
            layout: Some(&init_pipeline_layout),
            module: &init_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    ("100", config.nx as f64),
                    ("101", config.ny as f64),
                    ("102", config.rho_init as f64),
                    ("103", config.u_x_init as f64),
                    ("104", config.u_y_init as f64),
                    ("105", config.wgs_x as f64),
                    ("106", config.wgs_y as f64),
                ],
                ..Default::default()
            },
            cache: None,
        });

        let wg_x = (config.nx + config.wgs_x - 1) / config.wgs_x;
        let wg_y = (config.ny + config.wgs_y - 1) / config.wgs_y;

        let step_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LBM2D Step Pipeline Layout"),
            bind_group_layouts: &[Some(&step_bgl)],
            immediate_size: 0,
        });

        let step_even_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM2D Step Even Pipeline"),
            layout: Some(&step_pipeline_layout),
            module: &step_even_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    ("100", config.nx as f64),
                    ("101", config.ny as f64),
                    ("102", config.omega as f64),
                    ("103", config.wgs_x as f64),
                    ("104", config.wgs_y as f64),
                ],
                ..Default::default()
            },
            cache: None,
        });

        let step_odd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM2D Step Odd Pipeline"),
            layout: Some(&step_pipeline_layout),
            module: &step_odd_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    ("100", config.nx as f64),
                    ("101", config.ny as f64),
                    ("102", config.omega as f64),
                    ("103", config.wgs_x as f64),
                    ("104", config.wgs_y as f64),
                ],
                ..Default::default()
            },
            cache: None,
        });

        let extract_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("LBM2D Extract Pipeline Layout"),
                bind_group_layouts: &[Some(&extract_bgl)],
                immediate_size: 0,
            });

        let extract_even_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LBM2D Extract Even Pipeline"),
                layout: Some(&extract_pipeline_layout),
                module: &extract_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("100", config.nx as f64),
                        ("101", config.ny as f64),
                        ("105", config.wgs_x as f64),
                        ("106", config.wgs_y as f64),
                        ("107", 0.0),
                    ],
                    ..Default::default()
                },
                cache: None,
            });

        let extract_odd_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LBM2D Extract Odd Pipeline"),
                layout: Some(&extract_pipeline_layout),
                module: &extract_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("100", config.nx as f64),
                        ("101", config.ny as f64),
                        ("105", config.wgs_x as f64),
                        ("106", config.wgs_y as f64),
                        ("107", 1.0),
                    ],
                    ..Default::default()
                },
                cache: None,
            });

        Self {
            buffers,
            config,
            init_pipeline,
            step_even_pipeline,
            step_odd_pipeline,
            extract_even_pipeline,
            extract_odd_pipeline,
            wg_x,
            wg_y,
            step_count: 0,
        }
    }

    fn create_bgls(
        device: &wgpu::Device,
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
    ) {
        let mut entries = vec![
            bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, false),
            bgl_storage_entry(2, wgpu::ShaderStages::COMPUTE, true),
        ];

        let init_bgl = create_bgl(device, &entries);

        entries[0] = bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, true);
        entries[1] = bgl_storage_entry(2, wgpu::ShaderStages::COMPUTE, true);
        entries.push(bgl_storage_entry(1, wgpu::ShaderStages::COMPUTE, false));
        entries.push(bgl_storage_entry(3, wgpu::ShaderStages::COMPUTE, true));

        let step_bgl = create_bgl(device, &entries);

        entries.clear();
        entries.push(bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, true));
        entries.push(bgl_storage_entry(1, wgpu::ShaderStages::COMPUTE, false));

        let extract_bgl = create_bgl(device, &entries);

        (init_bgl, step_bgl, extract_bgl)
    }

    pub fn init(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBM2D Init Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.init_pipeline);
        pass.set_bind_group(0, &self.buffers.init_bg, &[]);
        pass.dispatch_workgroups(self.wg_x, self.wg_y, 1);
    }

    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBM2D Step Pass"),
            timestamp_writes: None,
        });

        if self.step_count % 2 == 0 {
            pass.set_pipeline(&self.step_even_pipeline);
            pass.set_bind_group(0, &self.buffers.step_bg_a, &[]);
        } else {
            pass.set_pipeline(&self.step_odd_pipeline);
            pass.set_bind_group(0, &self.buffers.step_bg_b, &[]);
        }

        pass.dispatch_workgroups(self.wg_x, self.wg_y, 1);

        self.step_count += 1;
    }

    pub fn step_multiple(&mut self, encoder: &mut wgpu::CommandEncoder, steps: u32) {
        for _ in 0..steps {
            self.step(encoder);
        }
    }

    pub fn extract(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBM2D Extract Pass"),
            timestamp_writes: None,
        });

        if self.step_count % 2 == 0 {
            pass.set_pipeline(&self.extract_even_pipeline);
            pass.set_bind_group(0, &self.buffers.extract_bg_a, &[]);
        } else {
            pass.set_pipeline(&self.extract_odd_pipeline);
            pass.set_bind_group(0, &self.buffers.extract_bg_b, &[]);
        }

        pass.dispatch_workgroups(self.wg_x, self.wg_y, 1);
    }
}
