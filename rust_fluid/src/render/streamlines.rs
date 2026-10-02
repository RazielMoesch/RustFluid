//! Trilinearly integrated streamlines rendered as instanced ribbons.

use wgpu::util::DeviceExt;

/// Compute and render resources for a seeded three-dimensional streamline set.
pub struct StreamlineRenderer {
    pub compute_pipeline: wgpu::ComputePipeline,
    pub render_pipeline: wgpu::RenderPipeline,
    pub point_buffer: wgpu::Buffer,
    pub valid_counts_buffer: wgpu::Buffer,
    pub seed_buffer: wgpu::Buffer,
    pub bind_group_compute: wgpu::BindGroup,
    pub bind_group_render: wgpu::BindGroup,
    pub num_seeds: u32,
    pub max_points: u32,
    clip_buffer: wgpu::Buffer,
    nx: u32,
    ny: u32,
    nz: u32,
}

impl StreamlineRenderer {
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
        seed_bounds_min: [f32; 3],
        seed_bounds_max: [f32; 3],
        grid_size: [u32; 3],
        max_points: u32,
        thickness: f32,
        colormap_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> Self {
        // generate seeds upstream
        let mut seeds: Vec<[f32; 4]> = Vec::new();
        let grid_x = grid_size[0].max(1);
        let grid_y = grid_size[1].max(1);
        let grid_z = grid_size[2].max(1);
        let num_seeds = grid_x * grid_y * grid_z;

        for z_idx in 0..grid_z {
            for y_idx in 0..grid_y {
                for x_idx in 0..grid_x {
                    let x_frac = if grid_x > 1 {
                        x_idx as f32 / (grid_x - 1) as f32
                    } else {
                        0.5
                    };
                    let y_frac = if grid_y > 1 {
                        y_idx as f32 / (grid_y - 1) as f32
                    } else {
                        0.5
                    };
                    let z_frac = if grid_z > 1 {
                        z_idx as f32 / (grid_z - 1) as f32
                    } else {
                        0.5
                    };

                    let x = seed_bounds_min[0] + x_frac * (seed_bounds_max[0] - seed_bounds_min[0]);
                    let y = seed_bounds_min[1] + y_frac * (seed_bounds_max[1] - seed_bounds_min[1]);
                    let z = seed_bounds_min[2] + z_frac * (seed_bounds_max[2] - seed_bounds_min[2]);

                    seeds.push([x, y, z, 0.0]); // padding in w
                }
            }
        }

        let seed_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Streamline Seed Buffer"),
            contents: bytemuck::cast_slice(&seeds),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let points_size = (num_seeds * max_points * 16) as wgpu::BufferAddress; // 16 bytes per vec4
        let point_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Streamline Point Buffer"),
            size: points_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let counts_size = (num_seeds * 4) as wgpu::BufferAddress;
        let valid_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Streamline Counts Buffer"),
            size: counts_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let clip_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Streamline Clip Uniform"),
            contents: bytemuck::cast_slice(&[nx as f32, 0.0, 0.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl_compute = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Streamline Compute BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
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
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("Streamline Compute BG"),
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
                    resource: seed_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: point_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: valid_counts_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Streamline Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(STREAMLINE_COMPUTE.into()),
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
                ("103", max_points as f64),
                ("104", thickness as f64),
            ],
            ..Default::default()
        };

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Streamline Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: comp_opts.clone(),
            cache: None,
        });

        let bgl_render = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Streamline Render BGL"),
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
                wgpu::BindGroupLayoutEntry {
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
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
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
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
            label: Some("Streamline Render BG"),
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
                    resource: point_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: valid_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: clip_buffer.as_entire_binding(),
                },
            ],
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Streamline Render Shader"),
            source: wgpu::ShaderSource::Wgsl(STREAMLINE_RENDER.into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl_render)],
                immediate_size: 0,
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Streamline Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: comp_opts.clone(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
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
            // Streamlines are an analysis overlay.  The shared mesh depth
            // attachment was rejecting the entire draw on affected drivers.
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
            point_buffer,
            valid_counts_buffer,
            seed_buffer,
            bind_group_compute,
            bind_group_render,
            num_seeds,
            max_points,
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

    pub fn compute<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        pass.set_pipeline(&self.compute_pipeline);
        pass.set_bind_group(0, &self.bind_group_compute, &[]);
        let wgs = (self.num_seeds + 63) / 64;
        pass.dispatch_workgroups(wgs, 1, 1);
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.bind_group_render, &[]);
        // Each instance is a seed. Draw (max_points - 1) * 6 vertices per instance for quads.
        let vertices_per_instance = (self.max_points - 1) * 6;
        pass.draw(0..vertices_per_instance, 0..self.num_seeds);
    }
}

const STREAMLINE_COMPUTE: &str = r#"
@id(100) override NX: f32 = 384.0;
@id(101) override NY: f32 = 96.0;
@id(102) override NZ: f32 = 32.0;
@id(103) override MAX_POINTS: u32 = 256u;
@id(104) override THICKNESS: f32 = 0.1;

@group(0) @binding(0) var<storage, read> macro_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read> seeds: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> points: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> valid_counts: array<u32>;

fn is_solid(p: vec3<f32>) -> bool {
    if (p.x < 0.0 || p.x > f32(NX)-1.0 || p.y < 0.0 || p.y > f32(NY)-1.0 || p.z < 0.0 || p.z > f32(NZ)-1.0) {
        return true;
    }
    let ix = u32(p.x);
    let iy = u32(p.y);
    let iz = u32(p.z);
    let idx = ix + iy * u32(NX) + iz * u32(NX) * u32(NY);
    let f = flags[idx];
    let btype = f >> 24u;
    return btype == 1u; // 1u is BOUNCE_BACK (solid)
}

fn sample_velocity_trilinear(p: vec3<f32>) -> vec3<f32> {
    let nx = u32(NX);
    let ny = u32(NY);
    
    let x = clamp(p.x, 0.0, f32(NX) - 1.001);
    let y = clamp(p.y, 0.0, f32(NY) - 1.001);
    let z = clamp(p.z, 0.0, f32(NZ) - 1.001);
    
    let ix = u32(x);
    let iy = u32(y);
    let iz = u32(z);
    
    let fx = x - f32(ix);
    let fy = y - f32(iy);
    let fz = z - f32(iz);
    
    let i000 = ix + iy * nx + iz * nx * ny;
    let i100 = i000 + 1u;
    let i010 = i000 + nx;
    let i110 = i010 + 1u;
    let i001 = i000 + nx * ny;
    let i101 = i001 + 1u;
    let i011 = i001 + nx;
    let i111 = i011 + 1u;
    
    let v000 = macro_data[i000].xyz;
    let v100 = macro_data[i100].xyz;
    let v010 = macro_data[i010].xyz;
    let v110 = macro_data[i110].xyz;
    let v001 = macro_data[i001].xyz;
    let v101 = macro_data[i101].xyz;
    let v011 = macro_data[i011].xyz;
    let v111 = macro_data[i111].xyz;
    
    let v00 = mix(v000, v100, vec3<f32>(fx));
    let v10 = mix(v010, v110, vec3<f32>(fx));
    let v01 = mix(v001, v101, vec3<f32>(fx));
    let v11 = mix(v011, v111, vec3<f32>(fx));
    
    let v0 = mix(v00, v10, vec3<f32>(fy));
    let v1 = mix(v01, v11, vec3<f32>(fy));
    
    return mix(v0, v1, vec3<f32>(fz));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let seed_idx = id.x;
    if (seed_idx >= arrayLength(&seeds)) { return; }

    var p = seeds[seed_idx].xyz;
    var valid_points = 0u;

    let base_idx = seed_idx * MAX_POINTS;
    let initial_v = sample_velocity_trilinear(p);
    points[base_idx] = vec4<f32>(p, length(initial_v));
    valid_points = 1u;

    let dt = 1.0; // Voxel integration length
    
    for (var step = 1u; step < MAX_POINTS; step += 1u) {
        let v1 = sample_velocity_trilinear(p);
        let speed1 = length(v1);
        if (!(speed1 >= 1e-6 && speed1 < 1e6)) { break; }

        let midpoint = p + 0.5 * dt * (v1 / speed1);
        if (is_solid(midpoint)) { break; }

        let v2 = sample_velocity_trilinear(midpoint);
        let speed2 = length(v2);
        if (!(speed2 >= 1e-6 && speed2 < 1e6)) { break; }

        let next_p = p + dt * (v2 / speed2);
        if (is_solid(next_p)) { break; }

        p = next_p;
        points[base_idx + step] = vec4<f32>(p, speed2);
        valid_points += 1u;
    }
    
    valid_counts[seed_idx] = valid_points;
}
"#;

const STREAMLINE_RENDER: &str = r#"
@id(100) override NX: f32 = 384.0;
@id(101) override NY: f32 = 96.0;
@id(102) override NZ: f32 = 32.0;
@id(103) override MAX_POINTS: u32 = 256u;
@id(104) override THICKNESS: f32 = 0.1;

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
@group(0) @binding(3) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> valid_counts: array<u32>;
@group(0) @binding(5) var<uniform> clip: vec4<f32>;

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) speed: f32,
    @location(1) world_pos: vec3<f32>,
}

@vertex
fn vs(@builtin(vertex_index) v_idx: u32, @builtin(instance_index) i_idx: u32) -> VOut {
    var out: VOut;
    let seed_idx = i_idx;
    let valid_count = valid_counts[seed_idx];
    
    let segment_idx = v_idx / 6u;
    let vert_in_quad = v_idx % 6u;
    
    if (segment_idx + 1u >= valid_count || segment_idx + 1u >= MAX_POINTS) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.speed = 0.0;
        out.world_pos = vec3<f32>(0.0);
        return out;
    }
    
    let p0_data = points[seed_idx * MAX_POINTS + segment_idx];
    let p1_data = points[seed_idx * MAX_POINTS + segment_idx + 1u];
    
    let p0 = p0_data.xyz;
    let p1 = p1_data.xyz;
    
    let is_p1 = (vert_in_quad == 2u || vert_in_quad == 4u || vert_in_quad == 5u);
    let is_pos_offset = (vert_in_quad == 0u || vert_in_quad == 3u || vert_in_quad == 5u);
    
    let p = select(p0, p1, is_p1);
    let speed = select(p0_data.w, p1_data.w, is_p1);
    
    let line_dir = p1 - p0;
    if (length(line_dir) < 0.001) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.speed = 0.0;
        out.world_pos = vec3<f32>(0.0);
        return out;
    }
    
    let dir = normalize(line_dir);
    let view_dir = normalize(uniforms.eye.xyz - p);
    var normal = cross(dir, view_dir);
    if (length(normal) < 0.001) {
        normal = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        normal = normalize(normal);
    }
    
    let thickness = THICKNESS; // Thickness in world units (voxels)
    let sign = select(-1.0, 1.0, is_pos_offset);
    let world_pos = p + normal * (thickness * sign);
    
    out.position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.speed = speed;
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
    return vec4<f32>(sample.rgb, 1.0); // Full opacity for verification
}
"#;
