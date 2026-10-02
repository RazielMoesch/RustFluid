//! Compute-advected particles rendered as speed-colored flow streaks.

use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// GPU particle position with speed stored in the fourth component.
pub struct Particle {
    pub pos_speed: [f32; 4], // x, y, z, speed
}

/// Owns particle initialization, advection, clipping, and render pipelines.
pub struct ParticleRenderer {
    pub compute_pipeline: wgpu::ComputePipeline,
    pub render_pipeline: wgpu::RenderPipeline,
    pub particle_buffer: wgpu::Buffer,
    pub bind_group_compute: wgpu::BindGroup,
    pub bind_group_render: wgpu::BindGroup,
    pub num_particles: u32,
    clip_buffer: wgpu::Buffer,
    nx: u32,
    ny: u32,
    nz: u32,
}

impl ParticleRenderer {
    pub fn new(
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        macro_buffer: &wgpu::Buffer,
        flags_buffer: &wgpu::Buffer,
        format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        nx: u32,
        ny: u32,
        nz: u32,
        num_particles: u32,
        thickness: f32,
        colormap_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> Self {
        // Initialize particles randomly at the inlet (X = 5)
        let mut initial_particles = Vec::with_capacity(num_particles as usize);
        for i in 0..num_particles {
            let y = (i as f32 * 13.7) % ny as f32;
            let z = (i as f32 * 7.3) % nz as f32;
            let x = (i as f32 * 19.1) % nx as f32; // Scatter throughout domain initially
            initial_particles.push(Particle {
                pos_speed: [x, y, z, 0.0],
            });
        }

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&initial_particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        });
        let clip_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Clip Uniform"),
            contents: bytemuck::cast_slice(&[nx as f32, 0.0, 0.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Compute Pipeline for Advection
        let bgl_compute = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Compute BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // macro_data
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // flags
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // particles
                    binding: 2,
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

        let bind_group_compute = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle Compute BG"),
            layout: &bgl_compute,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: macro_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flags_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: particle_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_COMPUTE.into()),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl_compute)],
                immediate_size: 0,
            });

        let comp_opts = wgpu::PipelineCompilationOptions {
            constants: &[
                ("100", nx as f64),
                ("101", ny as f64),
                ("102", nz as f64),
                ("103", thickness as f64),
            ],
            ..Default::default()
        };

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Particle Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: comp_opts.clone(),
            cache: None,
        });

        // Render Pipeline for Lines
        let bgl_render = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Render BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // camera
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // colormap
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D1,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // sampler
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // macro_data (for velocity trail)
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Domain-space clipping parameters
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group_render = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle Render BG"),
            layout: &bgl_render,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(colormap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: macro_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: clip_buffer.as_entire_binding(),
                },
            ],
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Render Shader"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_RENDER.into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl_render)],
                immediate_size: 0,
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Particle>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4],
                })],
                compilation_options: comp_opts.clone(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: comp_opts,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            compute_pipeline,
            render_pipeline,
            particle_buffer,
            bind_group_compute,
            bind_group_render,
            num_particles,
            clip_buffer,
            nx,
            ny,
            nz,
        }
    }

    pub fn set_visible_fraction(
        &self,
        queue: &wgpu::Queue,
        fraction: f32,
        axis: u32,
        from_max: bool,
    ) {
        let fraction = fraction.clamp(0.0, 1.0);
        let extent = match axis {
            1 => self.ny as f32,
            2 => self.nz as f32,
            _ => self.nx as f32,
        };
        let limit = if from_max {
            if fraction == 0.0 {
                extent + 1.0
            } else {
                extent * (1.0 - fraction)
            }
        } else if fraction == 0.0 {
            -1.0
        } else {
            extent * fraction
        };
        queue.write_buffer(
            &self.clip_buffer,
            0,
            bytemuck::cast_slice(&[limit, axis as f32, if from_max { 1.0 } else { 0.0 }, 0.0]),
        );
    }

    pub fn advect<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        pass.set_pipeline(&self.compute_pipeline);
        pass.set_bind_group(0, &self.bind_group_compute, &[]);
        let wgs = (self.num_particles + 255) / 256;
        pass.dispatch_workgroups(wgs, 1, 1);
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.bind_group_render, &[]);
        pass.set_vertex_buffer(0, self.particle_buffer.slice(..));
        // Draw 6 vertices (a quad) for every instance (particle)
        pass.draw(0..6, 0..self.num_particles);
    }
}

const PARTICLE_COMPUTE: &str = r#"
@id(100) override NX: f32 = 384.0;
@id(101) override NY: f32 = 96.0;
@id(102) override NZ: f32 = 32.0;
@id(103) override THICKNESS: f32 = 0.05;

struct Particle {
    pos_speed: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> macro_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;

fn hash(i: u32) -> f32 {
    var x = i;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return f32(x) / f32(0xFFFFFFFFu);
}

fn sample_velocity(p: vec3<f32>) -> vec3<f32> {
    let ix = u32(clamp(p.x, 0.0, f32(NX) - 1.0));
    let iy = u32(clamp(p.y, 0.0, f32(NY) - 1.0));
    let iz = u32(clamp(p.z, 0.0, f32(NZ) - 1.0));
    let idx = ix + iy * u32(NX) + iz * u32(NX) * u32(NY);
    let m = macro_data[idx];
    return m.xyz; // return velocity
}

fn is_solid(p: vec3<f32>) -> bool {
    let ix = u32(clamp(p.x, 0.0, f32(NX) - 1.0));
    let iy = u32(clamp(p.y, 0.0, f32(NY) - 1.0));
    let iz = u32(clamp(p.z, 0.0, f32(NZ) - 1.0));
    let idx = ix + iy * u32(NX) + iz * u32(NX) * u32(NY);
    let f = flags[idx];
    let btype = f >> 24u;
    return btype == 1u; // 1u is BOUNCE_BACK (solid)
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&particles)) { return; }
    
    var p = particles[i];
    var pos = p.pos_speed.xyz;
    let v = sample_velocity(pos);
    
    // Advect particle
    let dt = 2.5; // Slower advection speed to prevent strobing/wagon-wheel effect
    pos += v * dt;
    let speed = length(v);
    
    // Check bounds and solid collisions
    let out_of_bounds = pos.x < 0.0 || pos.x > f32(NX) || pos.y < 0.0 || pos.y > f32(NY) || pos.z < 0.0 || pos.z > f32(NZ);
    let hit_solid = is_solid(pos);
    
    // If dead, respawn at the inlet for a proper wind tunnel stream!
    if (out_of_bounds || hit_solid) {
        let r1 = hash(i * 13u + u32(abs(pos.x) * 1000.0));
        let r2 = hash(i * 17u + u32(abs(pos.y) * 1000.0));
        let r3 = hash(i * 19u + u32(abs(pos.z) * 1000.0));
        // Spawn safely inside the bounds to prevent getting stuck
        pos = vec3<f32>(r3 * 10.0 + 2.0, r1 * (f32(NY) - 4.0) + 2.0, r2 * (f32(NZ) - 4.0) + 2.0);
    }
    
    particles[i].pos_speed = vec4<f32>(pos, speed);
}
"#;

const PARTICLE_RENDER: &str = r#"
@id(100) override NX: f32 = 384.0;
@id(101) override NY: f32 = 96.0;
@id(102) override NZ: f32 = 32.0;
@id(103) override THICKNESS: f32 = 0.05;

struct Uniforms {
    inv_view_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    max_speed: f32,
    iso_q: f32,
    mode: u32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var colormap: texture_1d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read> macro_data: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> clip: vec4<f32>;

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) speed: f32,
    @location(1) tail_fade: f32,
    @location(2) world_pos: vec3<f32>,
}

fn sample_velocity(p: vec3<f32>) -> vec3<f32> {
    let ix = u32(clamp(p.x, 0.0, f32(NX) - 1.0));
    let iy = u32(clamp(p.y, 0.0, f32(NY) - 1.0));
    let iz = u32(clamp(p.z, 0.0, f32(NZ) - 1.0));
    let idx = ix + iy * u32(NX) + iz * u32(NX) * u32(NY);
    return macro_data[idx].xyz; // velocity is in xyz, rho is in w
}

@vertex
fn vs(@builtin(vertex_index) v_idx: u32, @location(0) pos_speed: vec4<f32>) -> VOut {
    var out: VOut;
    let speed = pos_speed.w;
    let p0 = pos_speed.xyz;
    let v = sample_velocity(p0);
    
    // Tail length matches the advection scaling
    let tail_vec = normalize(v) * (speed * 25.0 + 2.0);
    let p1 = p0 - tail_vec;
    
    // Create a camera-facing quad
    let view_dir = normalize(uniforms.eye.xyz - p0);
    var dir = tail_vec;
    if (length(dir) < 0.001) { dir = vec3<f32>(1.0, 0.0, 0.0); }
    var normal = cross(normalize(dir), view_dir);
    if (length(normal) < 0.001) { normal = vec3<f32>(0.0, 1.0, 0.0); }
    normal = normalize(normal);
    
    // Quad vertices (TriangleList):
    // 0: p0 + w, 1: p1 - w, 2: p0 - w, 3: p0 + w, 4: p1 + w, 5: p1 - w
    let is_p1 = (v_idx == 1u || v_idx == 4u || v_idx == 5u);
    let is_pos = (v_idx == 0u || v_idx == 3u || v_idx == 4u);
    let sign = select(-1.0, 1.0, is_pos);
    
    let p = select(p0, p1, is_p1);
    let fade = select(1.0, 0.0, is_p1);
    let world_pos = p + normal * (THICKNESS * sign);
    
    out.position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.speed = speed;
    out.tail_fade = fade;
    out.world_pos = world_pos;
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let coord = select(in.world_pos.x, select(in.world_pos.y, in.world_pos.z, clip.y > 1.5), clip.y > 0.5);
    if (clip.z < 0.5) {
        if (coord > clip.x) { discard; }
    } else {
        if (coord < clip.x) { discard; }
    }
    let normalized = clamp(in.speed / uniforms.max_speed, 0.0, 1.0);
    let sample = textureSampleLevel(colormap, samp, normalized, 0.0);
    let color = sample.rgb;
    
    // Fade toward the tail to show flow direction, but keep it faintly visible.
    // A fully transparent endpoint makes the
    // already narrow projected quad disappear under sub-pixel coverage.
    let alpha = sample.a * mix(0.2, 1.0, in.tail_fade);
    
    return vec4<f32>(color, alpha);
}
"#;
