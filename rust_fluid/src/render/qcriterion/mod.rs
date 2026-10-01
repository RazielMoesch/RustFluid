pub mod tables;

use crate::gpu::utils::{
    bg_entry, bgl_sampler_entry, bgl_storage_entry, bgl_storage_texture_entry, bgl_texture_entry,
    bgl_uniform_entry, create_bgl,
};

// ── Vertex layout (matches WGSL struct: 7 × f32 = 28 bytes per vertex) ──────

/// Stride in f32 elements per vertex in the flat storage buffer.
const VERTEX_STRIDE_F32: u32 = 7;
/// Stride in bytes per vertex.
const VERTEX_STRIDE_BYTES: u64 = (VERTEX_STRIDE_F32 as u64) * 4;

// ── WGSL shader sources ─────────────────────────────────────────────────────

/// Compute shader: velocity → Q-criterion (r32float 3D texture).
/// Boundary-aware derivatives (central/forward/backward), optional smoothing.
const COMPUTE_Q_SHADER: &str = r#"
@id(100) override NX: u32 = 128;
@id(101) override NY: u32 = 128;
@id(102) override NZ: u32 = 128;

@group(0) @binding(0) var<storage, read> macro_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var q_out: texture_storage_3d<r32float, write>;

const FLAG_TYPE_SHIFT: u32 = 24u;

fn is_solid(x: i32, y: i32, z: i32) -> bool {
    if (x < 0 || x >= i32(NX) || y < 0 || y >= i32(NY) || z < 0 || z >= i32(NZ)) {
        return true;
    }
    let idx = u32(x) + u32(y) * NX + u32(z) * NX * NY;
    return (flags[idx] >> FLAG_TYPE_SHIFT) != 0u;
}

fn get_u(x: i32, y: i32, z: i32, valid: ptr<function, bool>) -> vec3<f32> {
    if (x < 0 || x >= i32(NX) || y < 0 || y >= i32(NY) || z < 0 || z >= i32(NZ)) {
        *valid = false;
        return vec3<f32>(0.0);
    }
    let idx = u32(x) + u32(y) * NX + u32(z) * NX * NY;
    if ((flags[idx] >> FLAG_TYPE_SHIFT) != 0u) {
        *valid = false;
        return vec3<f32>(0.0);
    }
    *valid = true;
    return macro_data[idx].xyz;
}

@compute @workgroup_size(8, 8, 2)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    if (x >= NX || y >= NY || z >= NZ) { return; }

    let xi = i32(x);
    let yi = i32(y);
    let zi = i32(z);

    // Solid cells get Q = 0 (effectively excluded from MC)
    if (is_solid(xi, yi, zi)) {
        textureStore(q_out, vec3<i32>(xi, yi, zi), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    var vc = true;
    let u_c = get_u(xi, yi, zi, &vc);
    if (!vc) {
        textureStore(q_out, vec3<i32>(xi, yi, zi), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Boundary-aware velocity derivatives
    var vr = true; let u_r = get_u(xi + 1, yi, zi, &vr);
    var vl = true; let u_l = get_u(xi - 1, yi, zi, &vl);
    var vt = true; let u_t = get_u(xi, yi + 1, zi, &vt);
    var vb = true; let u_b = get_u(xi, yi - 1, zi, &vb);
    var vf = true; let u_f = get_u(xi, yi, zi + 1, &vf);
    var vk = true; let u_k = get_u(xi, yi, zi - 1, &vk);

    // dU/dx
    var ux_x = 0.0; var uy_x = 0.0; var uz_x = 0.0;
    var deriv_valid = false;
    if (vr && vl) {
        ux_x = (u_r.x - u_l.x) * 0.5; uy_x = (u_r.y - u_l.y) * 0.5; uz_x = (u_r.z - u_l.z) * 0.5;
        deriv_valid = true;
    } else if (vr) {
        ux_x = u_r.x - u_c.x; uy_x = u_r.y - u_c.y; uz_x = u_r.z - u_c.z;
        deriv_valid = true;
    } else if (vl) {
        ux_x = u_c.x - u_l.x; uy_x = u_c.y - u_l.y; uz_x = u_c.z - u_l.z;
        deriv_valid = true;
    }

    // dU/dy
    var ux_y = 0.0; var uy_y = 0.0; var uz_y = 0.0;
    if (vt && vb) {
        ux_y = (u_t.x - u_b.x) * 0.5; uy_y = (u_t.y - u_b.y) * 0.5; uz_y = (u_t.z - u_b.z) * 0.5;
        deriv_valid = true;
    } else if (vt) {
        ux_y = u_t.x - u_c.x; uy_y = u_t.y - u_c.y; uz_y = u_t.z - u_c.z;
        deriv_valid = true;
    } else if (vb) {
        ux_y = u_c.x - u_b.x; uy_y = u_c.y - u_b.y; uz_y = u_c.z - u_b.z;
        deriv_valid = true;
    }

    // dU/dz
    var ux_z = 0.0; var uy_z = 0.0; var uz_z = 0.0;
    if (vf && vk) {
        ux_z = (u_f.x - u_k.x) * 0.5; uy_z = (u_f.y - u_k.y) * 0.5; uz_z = (u_f.z - u_k.z) * 0.5;
        deriv_valid = true;
    } else if (vf) {
        ux_z = u_f.x - u_c.x; uy_z = u_f.y - u_c.y; uz_z = u_f.z - u_c.z;
        deriv_valid = true;
    } else if (vk) {
        ux_z = u_c.x - u_k.x; uy_z = u_c.y - u_k.y; uz_z = u_c.z - u_k.z;
        deriv_valid = true;
    }

    if (!deriv_valid) {
        textureStore(q_out, vec3<i32>(xi, yi, zi), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Strain rate tensor S and vorticity tensor Omega
    let s_xx = ux_x;
    let s_yy = uy_y;
    let s_zz = uz_z;
    let s_xy = 0.5 * (ux_y + uy_x);
    let s_xz = 0.5 * (ux_z + uz_x);
    let s_yz = 0.5 * (uy_z + uz_y);
    let norm_S2 = s_xx*s_xx + s_yy*s_yy + s_zz*s_zz + 2.0*(s_xy*s_xy + s_xz*s_xz + s_yz*s_yz);

    let o_xy = 0.5 * (ux_y - uy_x);
    let o_xz = 0.5 * (ux_z - uz_x);
    let o_yz = 0.5 * (uy_z - uz_y);
    let norm_O2 = 2.0 * (o_xy*o_xy + o_xz*o_xz + o_yz*o_yz);

    let q = 0.5 * (norm_O2 - norm_S2);
    textureStore(q_out, vec3<i32>(xi, yi, zi), vec4<f32>(q, 0.0, 0.0, 0.0));
}
"#;

/// Compute shader: GPU Marching Cubes — reads Q texture, emits triangles via atomic append.
/// Outputs vertices as flat f32 array: [px, py, pz, nx, ny, nz, speed] per vertex.
const MARCHING_CUBES_SHADER: &str = r#"
@id(100) override NX: u32 = 128;
@id(101) override NY: u32 = 128;
@id(102) override NZ: u32 = 128;

@group(0) @binding(0) var q_tex: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> macro_data: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> flags_buf: array<u32>;
@group(0) @binding(3) var<storage, read_write> vertices: array<f32>;
@group(0) @binding(4) var<storage, read_write> indirect: array<atomic<u32>, 4>;
@group(0) @binding(5) var<storage, read> edge_table: array<u32, 256>;
@group(0) @binding(6) var<storage, read> tri_table: array<i32, 4096>;

struct MCParams {
    iso_q: f32,
    max_speed: f32,
    max_vertices: u32,
    pad: u32,
};
@group(0) @binding(7) var<uniform> params: MCParams;

const FLAG_TYPE_SHIFT: u32 = 24u;

// Edge → two corner indices
const EDGE_CONN: array<vec2<u32>, 12> = array<vec2<u32>, 12>(
    vec2<u32>(0u, 1u), vec2<u32>(1u, 2u), vec2<u32>(2u, 3u), vec2<u32>(3u, 0u),
    vec2<u32>(4u, 5u), vec2<u32>(5u, 6u), vec2<u32>(6u, 7u), vec2<u32>(7u, 4u),
    vec2<u32>(0u, 4u), vec2<u32>(1u, 5u), vec2<u32>(2u, 6u), vec2<u32>(3u, 7u)
);

// Corner offsets within the cube
const CORNER_POS: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0)
);

fn global_idx(x: u32, y: u32, z: u32) -> u32 {
    return x + y * NX + z * NX * NY;
}

fn corner_global(cx: u32, cy: u32, cz: u32, corner: u32) -> u32 {
    let off = CORNER_POS[corner];
    return global_idx(cx + u32(off.x), cy + u32(off.y), cz + u32(off.z));
}

fn is_solid_at(idx: u32) -> bool {
    return (flags_buf[idx] >> FLAG_TYPE_SHIFT) == 1u;
}

// Analytic trilinear Q gradient within the cube, using the 8 corner Q values.
// f = fractional position (0..1) within the cube.
fn cube_gradient(q: array<f32, 8>, f: vec3<f32>) -> vec3<f32> {
    let fx = f.x; let fy = f.y; let fz = f.z;

    let dx = (1.0 - fz) * ((1.0 - fy) * (q[1] - q[0]) + fy * (q[2] - q[3]))
           + fz          * ((1.0 - fy) * (q[5] - q[4]) + fy * (q[6] - q[7]));

    let dy = (1.0 - fz) * ((1.0 - fx) * (q[3] - q[0]) + fx * (q[2] - q[1]))
           + fz          * ((1.0 - fx) * (q[7] - q[4]) + fx * (q[6] - q[5]));

    let dz = (1.0 - fy) * ((1.0 - fx) * (q[4] - q[0]) + fx * (q[5] - q[1]))
           + fy          * ((1.0 - fx) * (q[7] - q[3]) + fx * (q[6] - q[2]));

    return vec3<f32>(dx, dy, dz);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = gid.x;
    let cy = gid.y;
    let cz = gid.z;

    // Each thread handles one cube spanning (cx,cy,cz) to (cx+1,cy+1,cz+1).
    if (cx >= NX - 1u || cy >= NY - 1u || cz >= NZ - 1u) { return; }

    // Load Q at 8 corners
    var q: array<f32, 8>;
    q[0] = textureLoad(q_tex, vec3<i32>(i32(cx),     i32(cy),     i32(cz)),     0).r;
    q[1] = textureLoad(q_tex, vec3<i32>(i32(cx + 1u), i32(cy),     i32(cz)),     0).r;
    q[2] = textureLoad(q_tex, vec3<i32>(i32(cx + 1u), i32(cy + 1u), i32(cz)),     0).r;
    q[3] = textureLoad(q_tex, vec3<i32>(i32(cx),     i32(cy + 1u), i32(cz)),     0).r;
    q[4] = textureLoad(q_tex, vec3<i32>(i32(cx),     i32(cy),     i32(cz + 1u)), 0).r;
    q[5] = textureLoad(q_tex, vec3<i32>(i32(cx + 1u), i32(cy),     i32(cz + 1u)), 0).r;
    q[6] = textureLoad(q_tex, vec3<i32>(i32(cx + 1u), i32(cy + 1u), i32(cz + 1u)), 0).r;
    q[7] = textureLoad(q_tex, vec3<i32>(i32(cx),     i32(cy + 1u), i32(cz + 1u)), 0).r;

    // Build case index: bit set when Q < iso_q (standard MC convention)
    var case_idx = 0u;
    if (q[0] < params.iso_q) { case_idx |= 1u; }
    if (q[1] < params.iso_q) { case_idx |= 2u; }
    if (q[2] < params.iso_q) { case_idx |= 4u; }
    if (q[3] < params.iso_q) { case_idx |= 8u; }
    if (q[4] < params.iso_q) { case_idx |= 16u; }
    if (q[5] < params.iso_q) { case_idx |= 32u; }
    if (q[6] < params.iso_q) { case_idx |= 64u; }
    if (q[7] < params.iso_q) { case_idx |= 128u; }

    // Trivial cases: all inside or all outside
    if (case_idx == 0u || case_idx == 255u) { return; }

    // Count triangles for this case
    var num_tris = 0u;
    let table_base = case_idx * 16u;
    for (var i = 0u; i < 16u; i += 3u) {
        if (tri_table[table_base + i] < 0) { break; }
        num_tris += 1u;
    }
    if (num_tris == 0u) { return; }

    // Atomic append: reserve space in the vertex buffer
    let num_verts = num_tris * 3u;
    let base_vertex = atomicAdd(&indirect[0], num_verts);

    // Capacity check: don't write if overflow
    if (base_vertex + num_verts > params.max_vertices) { return; }

    let origin = vec3<f32>(f32(cx), f32(cy), f32(cz));

    // Generate each triangle
    var vi = 0u;
    for (var i = 0u; i < 16u; i += 3u) {
        let e0 = tri_table[table_base + i];
        if (e0 < 0) { break; }
        let e1 = tri_table[table_base + i + 1u];
        let e2 = tri_table[table_base + i + 2u];

        // Emit 3 vertices
        for (var v = 0u; v < 3u; v++) {
            var edge_idx: u32;
            if (v == 0u) { edge_idx = u32(e0); }
            else if (v == 1u) { edge_idx = u32(e1); }
            else { edge_idx = u32(e2); }

            let c0 = EDGE_CONN[edge_idx].x;
            let c1 = EDGE_CONN[edge_idx].y;

            let q0 = q[c0];
            let q1 = q[c1];

            // Sub-voxel interpolation factor
            let dq = q1 - q0;
            var t: f32;
            if (abs(dq) < 1e-10) {
                t = 0.5;
            } else {
                t = clamp((params.iso_q - q0) / dq, 0.0, 1.0);
            }

            // Interpolated position (voxel coordinates)
            let p0 = CORNER_POS[c0];
            let p1 = CORNER_POS[c1];
            let frac = p0 + t * (p1 - p0);  // fractional within cube [0,1]
            let pos = origin + frac;

            // Smooth normal from analytic trilinear gradient
            let grad = cube_gradient(q, frac);
            let grad_len = length(grad);
            var normal: vec3<f32>;
            if (grad_len > 1e-8) {
                normal = -grad / grad_len; // outward from high-Q region
            } else {
                normal = vec3<f32>(0.0, 1.0, 0.0);
            }

            // Interpolated velocity and speed for coloring
            let idx0 = corner_global(cx, cy, cz, c0);
            let idx1 = corner_global(cx, cy, cz, c1);
            let vel0 = macro_data[idx0].xyz;
            let vel1 = macro_data[idx1].xyz;
            let vel = vel0 + t * (vel1 - vel0);
            let speed = length(vel);

            // Write vertex: 7 f32s = [px, py, pz, nx, ny, nz, speed]
            let out_idx = (base_vertex + vi) * 7u;
            vertices[out_idx + 0u] = pos.x;
            vertices[out_idx + 1u] = pos.y;
            vertices[out_idx + 2u] = pos.z;
            vertices[out_idx + 3u] = normal.x;
            vertices[out_idx + 4u] = normal.y;
            vertices[out_idx + 5u] = normal.z;
            vertices[out_idx + 6u] = speed;

            vi += 1u;
        }
    }
}
"#;

/// Tiny compute shader: clamp indirect draw vertex_count to max_vertices.
const FINALIZE_SHADER: &str = r#"
@id(200) override MAX_VERTICES: u32 = 1000000;

@group(0) @binding(0) var<storage, read_write> indirect: array<atomic<u32>, 4>;

@compute @workgroup_size(1)
fn main() {
    let count = atomicLoad(&indirect[0]);
    if (count > MAX_VERTICES) {
        atomicStore(&indirect[0], MAX_VERTICES);
    }
}
"#;

/// Vertex + fragment shader for rendering MC triangle surfaces.
/// Blinn-Phong with velocity-colormap coloring.
const RENDER_Q_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    max_speed: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var colormap_tex: texture_1d<f32>;
@group(0) @binding(2) var color_sampler: sampler;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) speed: f32,
};

@vertex
fn vs(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) speed: f32,
) -> VOut {
    var out: VOut;
    out.clip_pos = camera.view_proj * vec4<f32>(position, 1.0);
    out.world_pos = position;
    out.normal = normal;
    out.speed = speed;
    return out;
}

fn ACESFilm(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let normal = normalize(in.normal);
    let view_dir = normalize(camera.eye.xyz - in.world_pos);

    let light_dir = normalize(vec3<f32>(0.5, 1.0, -0.2));

    // Velocity-based colormap
    let norm_speed = clamp(in.speed / camera.max_speed, 0.0, 1.0);
    let base_color = textureSampleLevel(colormap_tex, color_sampler, norm_speed, 0.0).rgb;

    // Simpler, vibrant lighting to preserve colormap
    let ambient = 0.25;
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let diffuse = n_dot_l * 0.75;

    let half_vec = normalize(light_dir + view_dir);
    let n_dot_h = max(dot(normal, half_vec), 0.0);
    let specular = pow(n_dot_h, 64.0) * 0.3;

    // Rim light for silhouette definition
    let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 3.0) * 0.15;

    // Multiply base color by diffuse/ambient, then add specular/rim on top
    let lit_color = base_color * (ambient + diffuse) + vec3<f32>(specular) + base_color * rim;

    return vec4<f32>(min(lit_color, vec3<f32>(1.0)), 1.0);
}
"#;

// ── QCriterionRenderer3D ─────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct QCriterionRenderer3D {
    // Q-field compute
    q_texture: wgpu::Texture,
    q_view: wgpu::TextureView,
    q_compute_pipeline: wgpu::ComputePipeline,
    q_compute_bg: wgpu::BindGroup,
    q_wg: [u32; 3],

    // Marching Cubes compute
    vertex_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    mc_pipeline: wgpu::ComputePipeline,
    mc_bg: wgpu::BindGroup,
    mc_params_buffer: wgpu::Buffer,
    mc_wg: [u32; 3],

    // Finalize compute (clamp vertex count)
    finalize_pipeline: wgpu::ComputePipeline,
    finalize_bg: wgpu::BindGroup,

    // Triangle rasterization
    render_pipeline: wgpu::RenderPipeline,
    render_bg: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,

    // Configuration
    nx: u32,
    ny: u32,
    nz: u32,
    max_vertices: u32,
}

impl QCriterionRenderer3D {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        macro_buffer: &wgpu::Buffer,
        flags_buffer: &wgpu::Buffer,
        _camera_buffer: &wgpu::Buffer,
        nx: u32,
        ny: u32,
        nz: u32,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        // ── Capacity ────────────────────────────────────────────────────
        let total_cubes = ((nx - 1) * (ny - 1) * (nz - 1)) as u64;
        let max_triangles = (total_cubes / 8).min(2_000_000) as u32;
        let max_vertices = max_triangles * 3;

        // ── Q texture (r32float 3D) ─────────────────────────────────────
        let q_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Q-Criterion 3D Texture"),
            size: wgpu::Extent3d {
                width: nx,
                height: ny,
                depth_or_array_layers: nz,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let q_view = q_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // ── Vertex buffer (STORAGE + VERTEX) ────────────────────────────
        let vertex_buf_size = max_vertices as u64 * VERTEX_STRIDE_BYTES;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MC Vertex Buffer"),
            size: vertex_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        // ── Indirect draw buffer ────────────────────────────────────────
        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MC Indirect Draw Args"),
            size: 16, // 4 × u32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── MC lookup table buffers ─────────────────────────────────────
        use wgpu::util::DeviceExt;

        let edge_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Edge Table"),
            contents: bytemuck::cast_slice(&tables::EDGE_TABLE),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let tri_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MC Triangle Table"),
            contents: bytemuck::cast_slice(&tables::TRI_TABLE),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ── MC params uniform ───────────────────────────────────────────
        let mc_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MC Params"),
            size: 16, // iso_q(f32) + max_speed(f32) + max_vertices(u32) + pad(u32)
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Colormap texture (same inferno-style as existing renderer) ──
        let colormap_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Q Colormap Texture"),
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
            let c1 = [0.05, 0.1, 0.4];
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

            colormap_data[i * 4] = (c[0] * 255.0) as u8;
            colormap_data[i * 4 + 1] = (c[1] * 255.0) as u8;
            colormap_data[i * 4 + 2] = (c[2] * 255.0) as u8;
            colormap_data[i * 4 + 3] = 255;
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
            label: Some("Q Colormap Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ── Override constants for grid dimensions ──────────────────────

        let grid_comp_opts = wgpu::PipelineCompilationOptions {
            constants: &[
                ("100", nx as f64),
                ("101", ny as f64),
                ("102", nz as f64),
            ],
            ..Default::default()
        };

        // ═══════════════════════════════════════════════════════════════
        // Pipeline 1: Q Compute
        // ═══════════════════════════════════════════════════════════════

        let q_compute_bgl = create_bgl(
            device,
            &[
                bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, true),  // macro_data
                bgl_storage_entry(1, wgpu::ShaderStages::COMPUTE, true),  // flags
                bgl_storage_texture_entry(
                    2,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::TextureFormat::R32Float,
                    wgpu::StorageTextureAccess::WriteOnly,
                    wgpu::TextureViewDimension::D3,
                ),
            ],
        );

        let q_compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Q Compute Bind Group"),
            layout: &q_compute_bgl,
            entries: &[
                bg_entry(0, macro_buffer.as_entire_binding()),
                bg_entry(1, flags_buffer.as_entire_binding()),
                bg_entry(2, wgpu::BindingResource::TextureView(&q_view)),
            ],
        });

        let q_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Q Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_Q_SHADER.into()),
        });

        let q_compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Q Compute Pipeline Layout"),
            bind_group_layouts: &[Some(&q_compute_bgl)],
            immediate_size: 0,
        });

        let q_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Q Compute Pipeline"),
                layout: Some(&q_compute_layout),
                module: &q_shader,
                entry_point: Some("main"),
                compilation_options: grid_comp_opts.clone(),
                cache: None,
            });

        let q_wg = [
            (nx + 7) / 8,
            (ny + 7) / 8,
            (nz + 1) / 2,
        ];

        // ═══════════════════════════════════════════════════════════════
        // Pipeline 2: Marching Cubes
        // ═══════════════════════════════════════════════════════════════

        let mc_bgl = create_bgl(
            device,
            &[
                bgl_texture_entry(
                    0,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::TextureViewDimension::D3,
                    wgpu::TextureSampleType::Float { filterable: false },
                ), // q_tex
                bgl_storage_entry(1, wgpu::ShaderStages::COMPUTE, true),  // macro_data
                bgl_storage_entry(2, wgpu::ShaderStages::COMPUTE, true),  // flags
                bgl_storage_entry(3, wgpu::ShaderStages::COMPUTE, false), // vertices
                bgl_storage_entry(4, wgpu::ShaderStages::COMPUTE, false), // indirect
                bgl_storage_entry(5, wgpu::ShaderStages::COMPUTE, true),  // edge_table
                bgl_storage_entry(6, wgpu::ShaderStages::COMPUTE, true),  // tri_table
                bgl_uniform_entry(7, wgpu::ShaderStages::COMPUTE),        // params
            ],
        );

        let mc_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MC Compute Bind Group"),
            layout: &mc_bgl,
            entries: &[
                bg_entry(0, wgpu::BindingResource::TextureView(&q_view)),
                bg_entry(1, macro_buffer.as_entire_binding()),
                bg_entry(2, flags_buffer.as_entire_binding()),
                bg_entry(3, vertex_buffer.as_entire_binding()),
                bg_entry(4, indirect_buffer.as_entire_binding()),
                bg_entry(5, edge_table_buffer.as_entire_binding()),
                bg_entry(6, tri_table_buffer.as_entire_binding()),
                bg_entry(7, mc_params_buffer.as_entire_binding()),
            ],
        });

        let mc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Marching Cubes Shader"),
            source: wgpu::ShaderSource::Wgsl(MARCHING_CUBES_SHADER.into()),
        });

        let mc_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MC Pipeline Layout"),
            bind_group_layouts: &[Some(&mc_bgl)],
            immediate_size: 0,
        });

        let mc_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Marching Cubes Pipeline"),
            layout: Some(&mc_layout),
            module: &mc_shader,
            entry_point: Some("main"),
            compilation_options: grid_comp_opts.clone(),
            cache: None,
        });

        // MC workgroups: one thread per cube, workgroup size (4,4,4)
        let mc_wg = [
            (nx - 1 + 3) / 4,
            (ny - 1 + 3) / 4,
            (nz - 1 + 3) / 4,
        ];

        // ═══════════════════════════════════════════════════════════════
        // Pipeline 3: Finalize (clamp indirect vertex count)
        // ═══════════════════════════════════════════════════════════════

        let finalize_bgl = create_bgl(
            device,
            &[bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, false)],
        );

        let finalize_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Finalize Bind Group"),
            layout: &finalize_bgl,
            entries: &[bg_entry(0, indirect_buffer.as_entire_binding())],
        });

        let finalize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Finalize Shader"),
            source: wgpu::ShaderSource::Wgsl(FINALIZE_SHADER.into()),
        });

        let finalize_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Finalize Pipeline Layout"),
            bind_group_layouts: &[Some(&finalize_bgl)],
            immediate_size: 0,
        });

        let finalize_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Finalize Pipeline"),
                layout: Some(&finalize_layout),
                module: &finalize_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("200", max_vertices as f64)],
                    ..Default::default()
                },
                cache: None,
            });

        // ═══════════════════════════════════════════════════════════════
        // Pipeline 4: Triangle Rasterization
        // ═══════════════════════════════════════════════════════════════

        let render_bgl = create_bgl(
            device,
            &[
                bgl_uniform_entry(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                bgl_texture_entry(
                    1,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::TextureViewDimension::D1,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                bgl_sampler_entry(2, wgpu::ShaderStages::FRAGMENT, true),
            ],
        );

        // Camera buffer: view_proj(64) + eye(16) + max_speed(4) + pad(12) = 96 bytes
        // We share the camera buffer passed in from DefaultRenderer3D for the
        // view_proj and eye. But we need our own because the layout differs.
        // Actually, let's create our own small camera buffer.
        let own_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Q Surface Camera Buffer"),
            size: 96,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Q Render Bind Group"),
            layout: &render_bgl,
            entries: &[
                bg_entry(0, own_camera_buffer.as_entire_binding()),
                bg_entry(1, wgpu::BindingResource::TextureView(&colormap_view)),
                bg_entry(2, wgpu::BindingResource::Sampler(&sampler)),
            ],
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Q Render Shader"),
            source: wgpu::ShaderSource::Wgsl(RENDER_Q_SHADER.into()),
        });

        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Q Render Pipeline Layout"),
            bind_group_layouts: &[Some(&render_bgl)],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Q Surface Render Pipeline"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: VERTEX_STRIDE_BYTES,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3, // position
                        1 => Float32x3, // normal
                        2 => Float32,   // speed
                    ],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
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
                cull_mode: None, // Disabled until winding is verified
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
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
            q_texture,
            q_view,
            q_compute_pipeline,
            q_compute_bg,
            q_wg,

            vertex_buffer,
            indirect_buffer,
            mc_pipeline,
            mc_bg,
            mc_params_buffer,
            mc_wg,

            finalize_pipeline,
            finalize_bg,

            render_pipeline,
            render_bg,
            camera_buffer: own_camera_buffer,

            nx,
            ny,
            nz,
            max_vertices,
        }
    }

    /// Update camera uniforms. Called every frame from prepare().
    pub fn update_camera(
        &self,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        eye: glam::Vec3,
        max_speed: f32,
    ) {
        let mut data = [0.0f32; 24]; // 96 bytes / 4 = 24 f32s
        data[0..16].copy_from_slice(&view_proj.to_cols_array());
        data[16] = eye.x;
        data[17] = eye.y;
        data[18] = eye.z;
        data[19] = 1.0; // pad
        data[20] = max_speed;
        // 21..23 = pad

        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Update MC params (iso_q, max_speed). Called every frame.
    pub fn update_params(&self, queue: &wgpu::Queue, iso_q: f32, max_speed: f32) {
        let data: [u32; 4] = [
            iso_q.to_bits(),
            max_speed.to_bits(),
            self.max_vertices,
            0, // pad
        ];
        queue.write_buffer(&self.mc_params_buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Run Q compute + Marching Cubes + finalize. Call in the prepare phase.
    pub fn prepare(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, iso_q: f32, max_speed: f32) {
        // Update MC params
        self.update_params(queue, iso_q, max_speed);

        // Clear indirect draw args: vertex_count=0, instance_count=1, first_vertex=0, first_instance=0
        queue.write_buffer(
            &self.indirect_buffer,
            0,
            bytemuck::cast_slice(&[0u32, 1, 0, 0]),
        );

        // Pass 1: Compute Q field
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Q Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.q_compute_pipeline);
            pass.set_bind_group(0, &self.q_compute_bg, &[]);
            pass.dispatch_workgroups(self.q_wg[0], self.q_wg[1], self.q_wg[2]);
        }

        // Pass 2: Marching Cubes triangle generation
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Marching Cubes Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mc_pipeline);
            pass.set_bind_group(0, &self.mc_bg, &[]);
            pass.dispatch_workgroups(self.mc_wg[0], self.mc_wg[1], self.mc_wg[2]);
        }

        // Pass 3: Clamp vertex count to capacity
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MC Finalize Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.finalize_pipeline);
            pass.set_bind_group(0, &self.finalize_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Render the generated MC triangles. Call from within a render pass that has depth.
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.render_bg, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw_indirect(&self.indirect_buffer, 0);
    }
}
