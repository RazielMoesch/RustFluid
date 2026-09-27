pub mod components_2d;
pub mod components_3d;

use crate::gpu::utils::{bg_entry, bgl_storage_entry, create_bgl};
use crate::render::components_2d::{RENDER_CURL_2D, RENDER_VELOCITY_2D};

pub enum RenderMode2D {
    Velocity,
    Curl,
}

impl RenderMode2D {
    pub fn wgsl(&self) -> &'static str {
        match self {
            RenderMode2D::Velocity => RENDER_VELOCITY_2D,
            RenderMode2D::Curl => RENDER_CURL_2D,
        }
    }
}

pub struct Render2D {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub camera_buffer: wgpu::Buffer,
}

impl Render2D {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        macro_buffer: &wgpu::Buffer,
        flags_buffer: &wgpu::Buffer,
        nx: u32,
        ny: u32,
        mode: RenderMode2D,
        scale_limit: f64,
        min_threshold: f64,
    ) -> Self {
        let bgl_entries = vec![
            bgl_storage_entry(0, wgpu::ShaderStages::FRAGMENT, true),
            bgl_storage_entry(1, wgpu::ShaderStages::FRAGMENT, true),
            crate::gpu::utils::bgl_uniform_entry(2, wgpu::ShaderStages::VERTEX),
        ];
        let bgl = create_bgl(device, &bgl_entries);

        use wgpu::util::DeviceExt;
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::cast_slice(&[glam::Mat4::IDENTITY.to_cols_array()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render2D Bind Group"),
            layout: &bgl,
            entries: &[
                bg_entry(0, macro_buffer.as_entire_binding()),
                bg_entry(1, flags_buffer.as_entire_binding()),
                bg_entry(2, camera_buffer.as_entire_binding()),
            ],
        });

        let shader_source = mode.wgsl();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render2D Shader Module"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render2D Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let comp_opts = wgpu::PipelineCompilationOptions {
            constants: &[
                ("100", nx as f64),
                ("101", ny as f64),
                ("102", scale_limit),
                ("103", min_threshold),
            ],
            ..Default::default()
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render2D Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: comp_opts.clone(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: comp_opts,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group,
            camera_buffer,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, matrix: glam::Mat4) {
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[matrix.to_cols_array()]),
        );
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);

        render_pass.draw(0..6, 0..1);
    }
}
