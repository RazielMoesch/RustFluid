use crate::sim::common::precision::Precision;
use crate::sim::d3::boundary::Boundary3D;
use crate::sim::d3::collision::Collision3D;
use crate::sim::d3::config::SimulationConfig3D;
use crate::sim::d3::config::InitType;
use crate::sim::d3::lattice::Lattice3D;
use crate::sim::d3::shader::compiler::ShaderCompiler3D;
use crate::sim::common::precision::PrecisionConfig;
use crate::sim::d3::buffers::SimBuffers3D;

pub struct Lbm3D {
    pub buffers: SimBuffers3D,
    pub config: SimulationConfig3D,
    init_pipeline: wgpu::ComputePipeline,
    step_even_pipeline: wgpu::ComputePipeline,
    step_odd_pipeline: wgpu::ComputePipeline,
    extract_even_pipeline: wgpu::ComputePipeline,
    extract_odd_pipeline: wgpu::ComputePipeline,
    wg_x: u32,
    wg_y: u32,
    wg_z: u32,
    pub step_count: u32,
}

impl Lbm3D {
    pub fn new(
        device: &wgpu::Device,
        config: SimulationConfig3D,
        precision: Precision,
        lattice: &dyn Lattice3D,
        collision: &dyn Collision3D,
        boundaries: &[&dyn Boundary3D],
    ) -> Self {
        let prec_cfg = PrecisionConfig::from(precision);

        let (init_src, even_src, odd_src, extract_src) = ShaderCompiler3D::compile(
            lattice, collision, boundaries, &prec_cfg, config.pure_fluid, config.init_type, &config
        );

        let init_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBM3D Init Shader"),
            source: wgpu::ShaderSource::Wgsl(init_src.into()),
        });

        let even_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBM3D Step Even Shader"),
            source: wgpu::ShaderSource::Wgsl(even_src.into()),
        });

        let odd_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBM3D Step Odd Shader"),
            source: wgpu::ShaderSource::Wgsl(odd_src.into()),
        });

        let extract_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBM3D Extract Shader"),
            source: wgpu::ShaderSource::Wgsl(extract_src.into()),
        });

        let wg_x = (config.nx + config.wgs_x - 1) / config.wgs_x;
        let wg_y = (config.ny + config.wgs_y - 1) / config.wgs_y;
        let wg_z = (config.nz + config.wgs_z - 1) / config.wgs_z;

        let (init_bgl, step_bgl, extract_bgl) = Self::create_bgls(device);
        let bytes_per_pop = if prec_cfg.pop_type == "f32" { 4 } else { 2 };
        let buffers = SimBuffers3D::new(device, config.nx, config.ny, config.nz, 19, bytes_per_pop, config.num_boundary_configs, &init_bgl, &step_bgl, &extract_bgl);

        let init_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LBM3D Init Pipeline Layout"),
            bind_group_layouts: &[Some(&init_bgl)],
            immediate_size: 0,
        });

        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM3D Init Pipeline"),
            layout: Some(&init_pipeline_layout),
            module: &init_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let step_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LBM3D Step Pipeline Layout"),
            bind_group_layouts: &[Some(&step_bgl)],
            immediate_size: 0,
        });

        let step_even_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM3D Step Even Pipeline"),
            layout: Some(&step_pipeline_layout),
            module: &even_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let step_odd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM3D Step Odd Pipeline"),
            layout: Some(&step_pipeline_layout),
            module: &odd_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let extract_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LBM3D Extract Pipeline Layout"),
            bind_group_layouts: &[Some(&extract_bgl)],
            immediate_size: 0,
        });

        let extract_even_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM3D Extract Even Pipeline"),
            layout: Some(&extract_pipeline_layout),
            module: &extract_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let extract_odd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBM3D Extract Odd Pipeline"),
            layout: Some(&extract_pipeline_layout),
            module: &extract_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
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
            wg_z,
            step_count: 0,
        }
    }

    fn create_bgls(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
        fn bgl_storage_entry(binding: u32, visibility: wgpu::ShaderStages, read_only: bool) -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        }
        
        let init_entries = [
            bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, false), // fa
            bgl_storage_entry(2, wgpu::ShaderStages::COMPUTE, true),  // flags (read-only)
        ];

        let step_entries = [
            bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, true),  // input pop
            bgl_storage_entry(1, wgpu::ShaderStages::COMPUTE, false), // output pop
            bgl_storage_entry(2, wgpu::ShaderStages::COMPUTE, true),  // flags
            bgl_storage_entry(3, wgpu::ShaderStages::COMPUTE, true),  // boundary configs
        ];

        let extract_entries = [
            bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, true),  // input pop
            bgl_storage_entry(1, wgpu::ShaderStages::COMPUTE, false), // macro_data
        ];

        let init_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LBM3D Init BGL"),
            entries: &init_entries,
        });

        let step_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LBM3D Step BGL"),
            entries: &step_entries,
        });

        let extract_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LBM3D Extract BGL"),
            entries: &extract_entries,
        });

        (init_bgl, step_bgl, extract_bgl)
    }

    pub fn write_buffers(&self, queue: &wgpu::Queue, flags: &[u32], bcs: &[f32]) {
        queue.write_buffer(&self.buffers.flags, 0, bytemuck::cast_slice(flags));
        queue.write_buffer(&self.buffers.boundary_configs, 0, bytemuck::cast_slice(bcs));
    }

    pub fn init(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBM3D Init Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.init_pipeline);
        pass.set_bind_group(0, &self.buffers.init_bg, &[]);
        pass.dispatch_workgroups(self.wg_x, self.wg_y, self.wg_z);
    }

    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBM3D Step Pass"),
            timestamp_writes: None,
        });

        if self.step_count % 2 == 0 {
            pass.set_pipeline(&self.step_even_pipeline);
            pass.set_bind_group(0, &self.buffers.step_bg_a, &[]);
        } else {
            pass.set_pipeline(&self.step_odd_pipeline);
            pass.set_bind_group(0, &self.buffers.step_bg_b, &[]);
        }

        pass.dispatch_workgroups(self.wg_x, self.wg_y, self.wg_z);
        self.step_count += 1;
    }

    pub fn step_multiple(&mut self, encoder: &mut wgpu::CommandEncoder, steps: u32) {
        for _ in 0..steps {
            self.step(encoder);
        }
    }

    pub fn extract(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LBM3D Extract Pass"),
            timestamp_writes: None,
        });

        if self.step_count % 2 == 0 {
            pass.set_pipeline(&self.extract_even_pipeline);
            pass.set_bind_group(0, &self.buffers.extract_bg_a, &[]);
        } else {
            pass.set_pipeline(&self.extract_odd_pipeline);
            pass.set_bind_group(0, &self.buffers.extract_bg_b, &[]);
        }

        pass.dispatch_workgroups(self.wg_x, self.wg_y, self.wg_z);
    }

    pub fn download_macro_data(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<[f32; 4]> {
        let size = (self.config.nx * self.config.ny * self.config.nz * 16) as wgpu::BufferAddress;
        
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Macro Data Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Macro Data Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.buffers.macro_data, 0, &staging_buffer, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let data = if let Ok(d) = data { d } else { panic!("Map failed") };
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }
}
