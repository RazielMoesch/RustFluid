use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

pub struct MeshRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub bind_group: wgpu::BindGroup,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
}

impl MeshRenderer {
    pub fn new(
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        format: wgpu::TextureFormat,
        nx: u32,
        ny: u32,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Mesh Depth Texture"),
            size: wgpu::Extent3d { width: nx, height: ny, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mesh Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mesh Bind Group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(MESH_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            bind_group,
            depth_texture,
            depth_view,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Mesh Depth Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

pub fn generate_sphere_mesh(cx: f32, cy: f32, cz: f32, r: f32, lat_segments: u32, lon_segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for y in 0..=lat_segments {
        let v = y as f32 / lat_segments as f32;
        let theta = v * std::f32::consts::PI;

        for x in 0..=lon_segments {
            let u = x as f32 / lon_segments as f32;
            let phi = u * 2.0 * std::f32::consts::PI;

            let nx = phi.cos() * theta.sin();
            let ny = phi.sin() * theta.sin();
            let nz = theta.cos();

            vertices.push(Vertex {
                position: [cx + nx * r, cy + ny * r, cz + nz * r],
                normal: [nx, ny, nz],
            });
        }
    }

    for y in 0..lat_segments {
        for x in 0..lon_segments {
            let first = y * (lon_segments + 1) + x;
            let second = first + lon_segments + 1;

            indices.push(first);
            indices.push(second);
            indices.push(first + 1);

            indices.push(second);
            indices.push(second + 1);
            indices.push(first + 1);
        }
    }

    (vertices, indices)
}

pub fn generate_cylinder_mesh(cx: f32, cy: f32, r: f32, height: f32, segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Side vertices
    for i in 0..=segments {
        let theta = (i as f32 / segments as f32) * 2.0 * std::f32::consts::PI;
        let nx = theta.cos();
        let ny = theta.sin();

        // Bottom ring (z = 0)
        vertices.push(Vertex {
            position: [cx + nx * r, cy + ny * r, 0.0],
            normal: [nx, ny, 0.0],
        });
        
        // Top ring (z = height)
        vertices.push(Vertex {
            position: [cx + nx * r, cy + ny * r, height],
            normal: [nx, ny, 0.0],
        });
    }

    for i in 0..segments {
        let first = i * 2;
        let second = first + 1;
        let third = first + 2;
        let fourth = first + 3;

        indices.push(first);
        indices.push(second);
        indices.push(third);

        indices.push(second);
        indices.push(fourth);
        indices.push(third);
    }

    (vertices, indices)
}

const MESH_SHADER: &str = r#"
struct Uniforms {
    inv_view_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    max_speed: f32,
    iso_q: f32,
    mode: u32,
    pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> VOut {
    var out: VOut;
    let world_pos = vec4<f32>(position, 1.0);
    out.position = uniforms.view_proj * world_pos;
    out.normal = normal;
    out.world_pos = position;
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, -0.2));
    let normal = normalize(in.normal);
    let view_dir = normalize(uniforms.eye.xyz - in.world_pos);
    let half_vec = normalize(light_dir + view_dir);
    
    let diff = max(dot(normal, light_dir), 0.0);
    let spec = pow(max(dot(normal, half_vec), 0.0), 32.0);
    
    // Nice metallic bright look for obstacles so it feels fully solid/opaque
    let obj_color = vec3<f32>(0.8, 0.85, 0.9) * (diff + 0.4) + vec3<f32>(1.0, 1.0, 1.0) * spec * 0.8;
    
    return vec4<f32>(obj_color, 1.0);
}
"#;
