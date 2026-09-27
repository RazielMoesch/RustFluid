use crate::sim::{
    buffers::SimBuffers2D,
    lattices::{CompiledLattice2D, Lattice2D},
};

pub struct LBM2D {
    pub buffers: SimBuffers2D,

    init_pipeline: wgpu::ComputePipeline,
    step_even_pipeline: wgpu::ComputePipeline,
    step_odd_pipeline: wgpu::ComputePipeline,
    extract_even_pipeline: wgpu::ComputePipeline,
    extract_odd_pipeline: wgpu::ComputePipeline,

    wg_x: u32,
    wg_y: u32,

    pub step_count: u32,
}

impl LBM2D {
    pub fn new(
        device: &wgpu::Device,
        nx: u32,
        ny: u32,
        rho_init: f32,
        u_x_init: f32,
        u_y_init: f32,
        wgs_x: u32,
        wgs_y: u32,
        omega: f32,
        lattice: Lattice2D,
        num_boundary_configs: u32,
    ) -> Self {
        let compiled = lattice.compile(device);

        let CompiledLattice2D {
            q,
            bytes_per_population,
            precision: _,
            lattice_name: _,
            collision_name: _,
            init_bgl,
            step_bgl,
            extract_bgl,
            init_wgsl,
            step_even_wgsl,
            step_odd_wgsl,
            extract_wgsl,
        } = compiled;

        let buffers = SimBuffers2D::new(
            device,
            nx,
            ny,
            q,
            bytes_per_population,
            num_boundary_configs,
            &init_bgl,
            &step_bgl,
            &extract_bgl,
        );

        let init_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBM Init Shader"),
            source: wgpu::ShaderSource::Wgsl(init_wgsl.into()),
        });

        let step_even_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBM Step Even Shader"),
            source: wgpu::ShaderSource::Wgsl(step_even_wgsl.into()),
        });

        let step_odd_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBM Step Odd Shader"),
            source: wgpu::ShaderSource::Wgsl(step_odd_wgsl.into()),
        });

        let extract_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBM Extract Shader"),
            source: wgpu::ShaderSource::Wgsl(extract_wgsl.into()),
        });

        let init_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LBM Init Pipeline Layout"),
            bind_group_layouts: &[Some(&init_bgl)],
            immediate_size: 0,
        });

        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM Init Pipeline"),
            layout: Some(&init_pipeline_layout),
            module: &init_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    ("100", nx as f64),
                    ("101", ny as f64),
                    ("102", rho_init as f64),
                    ("103", u_x_init as f64),
                    ("104", u_y_init as f64),
                    ("105", wgs_x as f64),
                    ("106", wgs_y as f64),
                ],
                ..Default::default()
            },
            cache: None,
        });

        let wg_x = (nx + wgs_x - 1) / wgs_x;
        let wg_y = (ny + wgs_y - 1) / wgs_y;

        let step_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LBM Step Pipeline Layout"),
            bind_group_layouts: &[Some(&step_bgl)],
            immediate_size: 0,
        });

        let step_even_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM Step Even Pipeline"),
            layout: Some(&step_pipeline_layout),
            module: &step_even_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    ("100", nx as f64),
                    ("101", ny as f64),
                    ("102", omega as f64),
                    ("103", wgs_x as f64),
                    ("104", wgs_y as f64),
                ],
                ..Default::default()
            },
            cache: None,
        });

        let step_odd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM Step Odd Pipeline"),
            layout: Some(&step_pipeline_layout),
            module: &step_odd_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    ("100", nx as f64),
                    ("101", ny as f64),
                    ("102", omega as f64),
                    ("103", wgs_x as f64),
                    ("104", wgs_y as f64),
                ],
                ..Default::default()
            },
            cache: None,
        });

        let extract_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("LBM Extract Pipeline Layout"),
                bind_group_layouts: &[Some(&extract_bgl)],
                immediate_size: 0,
            });

        let extract_even_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LBM Extract Even Pipeline"),
                layout: Some(&extract_pipeline_layout),
                module: &extract_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("100", nx as f64),
                        ("101", ny as f64),
                        ("105", wgs_x as f64),
                        ("106", wgs_y as f64),
                        ("107", 0.0),
                    ],
                    ..Default::default()
                },
                cache: None,
            });

        let extract_odd_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LBM Extract Odd Pipeline"),
                layout: Some(&extract_pipeline_layout),
                module: &extract_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("100", nx as f64),
                        ("101", ny as f64),
                        ("105", wgs_x as f64),
                        ("106", wgs_y as f64),
                        ("107", 1.0),
                    ],
                    ..Default::default()
                },
                cache: None,
            });

        Self {
            buffers,
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

    pub fn init(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBM Init Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.init_pipeline);
        pass.set_bind_group(0, &self.buffers.init_bg, &[]);
        pass.dispatch_workgroups(self.wg_x, self.wg_y, 1);
    }

    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBM Step Pass"),
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
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBM Multi-Step Pass"),
            timestamp_writes: None,
        });

        for _ in 0..steps {
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
    }

    pub fn extract(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBM Extract Pass"),
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
