pub mod components_2d;
pub mod components_3d;
pub mod mesh;
pub mod particles;

use crate::gpu::utils::{
    bg_entry, bgl_sampler_entry, bgl_storage_entry, bgl_storage_texture_entry, bgl_texture_entry,
    create_bgl,
};
use crate::render::components_2d::{RENDER_CURL_2D, RENDER_VELOCITY_2D};
use crate::render::components_3d::{COMPUTE_VORTICITY_3D, RENDER_VORTICITY_3D};

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

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum RenderMode3D {
    QCriterion,
    FlowStreams,
}

impl RenderMode3D {
    pub fn wgsl(&self) -> &'static str {
        match self {
            RenderMode3D::QCriterion => RENDER_VORTICITY_3D,
            RenderMode3D::FlowStreams => RENDER_VORTICITY_3D, // we won't use it directly for streams
        }
    }
}

pub struct Render3D {
    pub compute_pipeline: wgpu::ComputePipeline,
    pub compute_bind_group: wgpu::BindGroup,
    
    pub vorticity_texture: wgpu::Texture,
    pub vorticity_view: wgpu::TextureView,
    
    pub colormap_texture: wgpu::Texture,
    pub colormap_view: wgpu::TextureView,
    
    pub sampler: wgpu::Sampler,

    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub camera_buffer: wgpu::Buffer,
    pub wg_x: u32,
    pub wg_y: u32,
    pub wg_z: u32,
}

impl Render3D {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        macro_buffer: &wgpu::Buffer,
        flags_buffer: &wgpu::Buffer,
        nx: u32,
        ny: u32,
        nz: u32,
        mode: RenderMode3D,
        _scale_limit: f64,
        _min_threshold: f64,
        depth_view: &wgpu::TextureView,
    ) -> Self {
        let vorticity_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Vorticity 3D Texture"),
            size: wgpu::Extent3d {
                width: nx,
                height: ny,
                depth_or_array_layers: nz,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let vorticity_view = vorticity_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let colormap_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Colormap 1D Texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D1,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let mut colormap_data = [0u8; 256 * 4];
        for i in 0..256 {
            let t = i as f32 / 255.0;
            let c1 = [0.05, 0.1, 0.4]; // more visible blue for 'slow'
            let c2 = [0.215, 0.055, 0.320];
            let c3 = [0.725, 0.190, 0.250];
            let c4 = [0.965, 0.535, 0.115];
            let c5 = [0.990, 0.985, 0.650];

            let mix = |a: [f32; 3], b: [f32; 3], f: f32| -> [f32; 3] {
                [
                    a[0] * (1.0 - f) + b[0] * f,
                    a[1] * (1.0 - f) + b[1] * f,
                    a[2] * (1.0 - f) + b[2] * f,
                ]
            };

            let c = if t < 0.25 {
                mix(c1, c2, t * 4.0)
            } else if t < 0.50 {
                mix(c2, c3, (t - 0.25) * 4.0)
            } else if t < 0.75 {
                mix(c3, c4, (t - 0.50) * 4.0)
            } else {
                mix(c4, c5, (t - 0.75) * 4.0)
            };
            
            // Map alpha to be fully visible always
            let a: f32 = 1.0;

            colormap_data[i * 4] = (c[0] * 255.0) as u8;
            colormap_data[i * 4 + 1] = (c[1] * 255.0) as u8;
            colormap_data[i * 4 + 2] = (c[2] * 255.0) as u8;
            colormap_data[i * 4 + 3] = (a * 255.0).clamp(0.0, 255.0) as u8;
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &colormap_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &colormap_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let colormap_view = colormap_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let compute_bgl_entries = vec![
            bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, true),
            bgl_storage_entry(1, wgpu::ShaderStages::COMPUTE, true),
            bgl_storage_texture_entry(
                2,
                wgpu::ShaderStages::COMPUTE,
                wgpu::TextureFormat::Rgba32Float,
                wgpu::StorageTextureAccess::WriteOnly,
                wgpu::TextureViewDimension::D3,
            ),
        ];
        let compute_bgl = create_bgl(device, &compute_bgl_entries);

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vorticity Compute Bind Group"),
            layout: &compute_bgl,
            entries: &[
                bg_entry(0, macro_buffer.as_entire_binding()),
                bg_entry(1, flags_buffer.as_entire_binding()),
                bg_entry(2, wgpu::BindingResource::TextureView(&vorticity_view)),
            ],
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vorticity Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_VORTICITY_3D.into()),
        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vorticity Compute Pipeline Layout"),
            bind_group_layouts: &[Some(&compute_bgl)],
            immediate_size: 0,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Vorticity Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    ("100", nx as f64),
                    ("101", ny as f64),
                    ("102", nz as f64),
                ],
                ..Default::default()
            },
            cache: None,
        });

        let wg_x = (nx + 7) / 8;
        let wg_y = (ny + 7) / 8;
        let wg_z = (nz + 1) / 2;

        let bgl_entries = vec![
            bgl_texture_entry(
                0,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::TextureViewDimension::D3,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_sampler_entry(1, wgpu::ShaderStages::FRAGMENT, true),
            bgl_texture_entry(
                2,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::TextureViewDimension::D1,
                wgpu::TextureSampleType::Float { filterable: true },
            ),
            crate::gpu::utils::bgl_uniform_entry(3, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
            bgl_texture_entry(
                4,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::TextureViewDimension::D2,
                wgpu::TextureSampleType::Depth,
            ),
        ];
        let bgl = create_bgl(device, &bgl_entries);

        use wgpu::util::DeviceExt;
        
        let initial_cam_data = vec![0.0f32; 40];
        
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer 3D"),
            contents: bytemuck::cast_slice(&initial_cam_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render3D Bind Group"),
            layout: &bgl,
            entries: &[
                bg_entry(0, wgpu::BindingResource::TextureView(&vorticity_view)),
                bg_entry(1, wgpu::BindingResource::Sampler(&sampler)),
                bg_entry(2, wgpu::BindingResource::TextureView(&colormap_view)),
                bg_entry(3, camera_buffer.as_entire_binding()),
                bg_entry(4, wgpu::BindingResource::TextureView(depth_view)),
            ],
        });

        let shader_source = mode.wgsl();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render3D Shader Module"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render3D Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let comp_opts = wgpu::PipelineCompilationOptions {
            constants: &[
                ("100", nx as f64),
                ("101", ny as f64),
                ("102", nz as f64),
            ],
            ..Default::default()
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render3D Pipeline"),
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            compute_pipeline,
            compute_bind_group,
            vorticity_texture,
            vorticity_view,
            colormap_texture,
            colormap_view,
            sampler,
            pipeline,
            bind_group,
            camera_buffer,
            wg_x,
            wg_y,
            wg_z,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, inv_view_proj: glam::Mat4, view_proj: glam::Mat4, eye: glam::Vec3, max_speed: f32, iso_q: f32, mode: u32) {
        let mut data = [0.0f32; 40];
        data[0..16].copy_from_slice(&inv_view_proj.to_cols_array());
        data[16..32].copy_from_slice(&view_proj.to_cols_array());
        data[32] = eye.x;
        data[33] = eye.y;
        data[34] = eye.z;
        data[35] = 1.0; // pad
        data[36] = max_speed;
        data[37] = iso_q;
        data[38] = f32::from_bits(mode);
        data[39] = 0.0; // pad
        
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&data),
        );
    }

    pub fn compute_vorticity(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Vorticity Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.compute_pipeline);
        pass.set_bind_group(0, &self.compute_bind_group, &[]);
        pass.dispatch_workgroups(self.wg_x, self.wg_y, self.wg_z);
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);

        render_pass.draw(0..6, 0..1);
    }

    pub fn update_depth_view(&mut self, device: &wgpu::Device, depth_view: &wgpu::TextureView) {
        let bgl = self.pipeline.get_bind_group_layout(0);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Render3D Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render3D Bind Group"),
            layout: &bgl,
            entries: &[
                crate::gpu::utils::bg_entry(0, wgpu::BindingResource::TextureView(&self.vorticity_view)),
                crate::gpu::utils::bg_entry(1, wgpu::BindingResource::Sampler(&sampler)),
                crate::gpu::utils::bg_entry(2, wgpu::BindingResource::TextureView(&self.colormap_view)),
                crate::gpu::utils::bg_entry(3, self.camera_buffer.as_entire_binding()),
                crate::gpu::utils::bg_entry(4, wgpu::BindingResource::TextureView(depth_view)),
            ],
        });
    }
}

