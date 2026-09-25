use crate::sim::{buffers::SimBuffers2D, lattices::{Lattice2D}};

pub struct LBM2D {
    pub buffers: SimBuffers2D,

    init_pipeline: wgpu::ComputePipeline,
    step_even_pipeline: wgpu::ComputePipeline,
    step_odd_pipeline: wgpu::ComputePipeline,
    extract_pipeline: wgpu::ComputePipeline,

    nx: u32,
    ny: u32,

    wgs_x: u32,
    wgs_y: u32,

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
    ) -> Self {
        
        let (init_bgl, step_bgl, extract_bgl) = match lattice {
            Lattice2D::D2Q9(ref d2q9) => d2q9.bgls(device),
        };
        
        let q = match lattice {
            Lattice2D::D2Q9(_) => 9,
        };

        let buffers = SimBuffers2D::new(device, nx, ny, q, &init_bgl, &step_bgl, &extract_bgl);

        let (init_wgsl, step_even_wgsl, step_odd_wgsl, extract_wgsl) = match lattice {
            Lattice2D::D2Q9(ref d2q9) => d2q9.wgsl(),
        };

        let init_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Init Module Shader"),
            source: wgpu::ShaderSource::Wgsl(init_wgsl.into()),
        });

        let step_even_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Step Module Shader"),
            source: wgpu::ShaderSource::Wgsl(step_even_wgsl.into()),
        });

        let step_odd_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Step Module Shader"),
            source: wgpu::ShaderSource::Wgsl(step_odd_wgsl.into()),
        });

        let extract_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Extract Module Shader"),
            source: wgpu::ShaderSource::Wgsl(extract_wgsl.into())
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

        let step_even_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LBM Step Even Pipeline Layout"),
            bind_group_layouts: &[Some(&step_bgl)],
            immediate_size: 0,
        });

        let step_even_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM Step Pipeline"),
            layout: Some(&step_even_pipeline_layout),
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

        let step_odd_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LBM Step Odd Pipeline Layout"),
            bind_group_layouts: &[Some(&step_bgl)],
            immediate_size: 0,
        });

        let step_odd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM Step Pipeline"),
            layout: Some(&step_odd_pipeline_layout),
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

        let extract_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LBM Extract Pipeline Layout"),
            bind_group_layouts: &[Some(&extract_bgl)],
            immediate_size: 0
        });

        let extract_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM Extract Pipeline"),
            layout: Some(&extract_pipeline_layout),
            module: &extract_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    ("100", nx as f64),
                    ("101", ny as f64),
                    ("105", wgs_x as f64),
                    ("106", wgs_y as f64)
                ],
                ..Default::default()
            },
            cache: None
        });

        Self {
            buffers,
            init_pipeline,
            step_even_pipeline,
            step_odd_pipeline,
            extract_pipeline,
            nx,
            ny,
            wgs_x,
            wgs_y,
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

        let wg_x = (self.nx + self.wgs_x - 1) / self.wgs_x;
        let wg_y = (self.ny + self.wgs_y - 1) / self.wgs_y;

        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBM Step Pass"),
            timestamp_writes: None,
        });

        if self.step_count % 2 == 0 {
            pass.set_pipeline(&self.step_even_pipeline);

            let bg = if self.step_count % 2 == 0 {
                &self.buffers.step_bg_a
            } else {
                &self.buffers.step_bg_b
            };

            pass.set_bind_group(0, bg, &[]);
            
            let wg_x = (self.nx + self.wgs_x - 1) / self.wgs_x;
            let wg_y = (self.ny + self.wgs_y - 1) / self.wgs_y;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        else {
            pass.set_pipeline(&self.step_odd_pipeline);

            let bg = if self.step_count % 2 == 0 {
                &self.buffers.step_bg_a
            } else {
                &self.buffers.step_bg_b
            };

            pass.set_bind_group(0, bg, &[]);
            
            let wg_x = (self.nx + self.wgs_x - 1) / self.wgs_x;
            let wg_y = (self.ny + self.wgs_y - 1) / self.wgs_y;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.step_count += 1;
    }
    
    pub fn step_multiple(&mut self, encoder: &mut wgpu::CommandEncoder, steps: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBM Multi-Step Pass"),
            timestamp_writes: None,
        });
        
        
        
        for _ in 0..steps {
            pass.set_pipeline(
                if self.step_count % 2 == 0 {
                    &self.step_even_pipeline
                } else {
                    &self.step_odd_pipeline
                }
            );
        
            let wg_x = (self.nx + self.wgs_x - 1) / self.wgs_x;
            let wg_y = (self.ny + self.wgs_y - 1) / self.wgs_y;
            let bg = if self.step_count % 2 == 0 {
                &self.buffers.step_bg_a
            } else {
                &self.buffers.step_bg_b
            };
            
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
            self.step_count += 1;
        }
    }

    pub fn extract(&mut self, encoder: &mut wgpu::CommandEncoder) {

        if self.step_count % 2 != 0 {
            {
                self.step(encoder);
            }
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBM Extract Pass"),
            timestamp_writes: None
        });

        pass.set_pipeline(&self.extract_pipeline);


        let wg_x = (self.nx + self.wgs_x - 1) / self.wgs_x;
        let wg_y = (self.ny + self.wgs_y - 1) / self.wgs_y;

        let bg = if self.step_count % 2 == 0 {
            &self.buffers.extract_bg_a
        } else {
            &self.buffers.extract_bg_b
        };

        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);

    }

}