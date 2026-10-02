//! Embedded WGSL for planar velocity and curl visualization.

/// Full-screen velocity-magnitude renderer.
pub(super) const RENDER_VELOCITY_2D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(102) override SCALE_LIMIT: f32 = 0.125;
@id(103) override MIN_THRESHOLD: f32 = 0.001;

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>
};

@vertex
fn vs(@builtin(vertex_index) vidx: u32) -> VOut {
    var pos_arr = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0)
    );
    let p = pos_arr[vidx];
    let domain_aspect = f32(NX) / f32(NY);
    let pos = vec4<f32>((p.x * 2.0 - 1.0) * domain_aspect, 1.0 - p.y * 2.0, 0.0, 1.0);

    var out: VOut;
    out.position = camera_matrix * pos;
    out.uv = p;
    return out;
}

@group(0) @binding(0) var<storage, read> macro_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<uniform> camera_matrix: mat4x4<f32>;

const FLAG_TYPE_SHIFT: u32 = 24u;

// ── Inferno colormap ──
// Perceptually uniform, goes: black → purple → magenta → orange → yellow
fn inferno(t: f32) -> vec3<f32> {
    let s = clamp(t, 0.0, 1.0);
    let c0 = vec3<f32>(0.001, 0.000, 0.014);
    let c1 = vec3<f32>(0.258, 0.038, 0.406);
    let c2 = vec3<f32>(0.578, 0.148, 0.404);
    let c3 = vec3<f32>(0.865, 0.317, 0.226);
    let c4 = vec3<f32>(0.987, 0.645, 0.040);
    let c5 = vec3<f32>(0.988, 0.998, 0.645);

    if (s < 0.2) { return mix(c0, c1, s * 5.0); }
    if (s < 0.4) { return mix(c1, c2, (s - 0.2) * 5.0); }
    if (s < 0.6) { return mix(c2, c3, (s - 0.4) * 5.0); }
    if (s < 0.8) { return mix(c3, c4, (s - 0.6) * 5.0); }
    return mix(c4, c5, (s - 0.8) * 5.0);
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let x = u32(in.uv.x * f32(NX));
    let y = u32(in.uv.y * f32(NY));

    let cx = clamp(x, 0u, NX - 1u);
    let cy = clamp(y, 0u, NY - 1u);
    let idx = cx + (cy * NX);

    let raw_flag = flags[idx];
    let boundary_type = raw_flag >> FLAG_TYPE_SHIFT;

    if (boundary_type == 1u) {
        // Solid boundaries (obstacles) are rendered as a neutral grey
        return vec4<f32>(0.3, 0.3, 0.3, 1.0);
    }

    let u = macro_data[idx].xy;
    let speed = length(u);

    // Power-law scaling: expands low-speed detail, compresses high end
    let normalized = clamp((speed - MIN_THRESHOLD) / (SCALE_LIMIT - MIN_THRESHOLD), 0.0, 1.0);
    let t = pow(normalized, 0.45);

    let color = inferno(t);
    return vec4<f32>(color, 1.0);
}
"#;

pub(super) const RENDER_CURL_2D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(102) override SCALE_LIMIT: f32 = 0.0125;
@id(103) override MIN_THRESHOLD: f32 = 0.001;

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>
};

@vertex
fn vs(@builtin(vertex_index) vidx: u32) -> VOut {
    var pos_arr = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0)
    );
    let p = pos_arr[vidx];
    let domain_aspect = f32(NX) / f32(NY);
    let pos = vec4<f32>((p.x * 2.0 - 1.0) * domain_aspect, 1.0 - p.y * 2.0, 0.0, 1.0);

    var out: VOut;
    out.position = camera_matrix * pos;
    out.uv = p;
    return out;
}

@group(0) @binding(0) var<storage, read> macro_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<uniform> camera_matrix: mat4x4<f32>;

const FLAG_TYPE_SHIFT: u32 = 24u;

fn get_u(x: i32, y: i32) -> vec2<f32> {
    let cx = clamp(x, 0, i32(NX) - 1);
    let cy = clamp(y, 0, i32(NY) - 1);
    let idx = u32(cx) + u32(cy) * NX;
    
    let raw_flag = flags[idx];
    let boundary_type = raw_flag >> FLAG_TYPE_SHIFT;
    if (boundary_type == 1u) {
        return vec2<f32>(0.0, 0.0);
    }
    
    return macro_data[idx].xy;
}

// ── Fire-and-Ice diverging colormap ──
// Negative → blue glow,  zero → black,  positive → red/orange glow
fn diverging(t: f32) -> vec3<f32> {
    let s = clamp(t, -1.0, 1.0);

    // Shared center (near-black)
    let center = vec3<f32>(0.01, 0.01, 0.02);
    // Blue stops
    let b1 = vec3<f32>(0.02, 0.08, 0.45);
    let b2 = vec3<f32>(0.08, 0.40, 1.00);
    // Red stops
    let r1 = vec3<f32>(0.50, 0.06, 0.01);
    let r2 = vec3<f32>(1.00, 0.42, 0.05);

    if (s < 0.0) {
        let a = -s;
        if (a < 0.5) { return mix(center, b1, a * 2.0); }
        return mix(b1, b2, (a - 0.5) * 2.0);
    } else {
        let a = s;
        if (a < 0.5) { return mix(center, r1, a * 2.0); }
        return mix(r1, r2, (a - 0.5) * 2.0);
    }
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let x = i32(in.uv.x * f32(NX));
    let y = i32(in.uv.y * f32(NY));

    let cx = clamp(x, 0, i32(NX) - 1);
    let cy = clamp(y, 0, i32(NY) - 1);
    let idx = u32(cx) + u32(cy) * NX;

    let raw_flag = flags[idx];
    let boundary_type = raw_flag >> FLAG_TYPE_SHIFT;

    if (boundary_type == 1u) {
        // Solid boundaries (obstacles) are rendered as a neutral grey
        return vec4<f32>(0.3, 0.3, 0.3, 1.0);
    }

    let u_right = get_u(x + 1, y);
    let u_left  = get_u(x - 1, y);
    let u_up    = get_u(x, y + 1);
    let u_down  = get_u(x, y - 1);

    let dy_dx = (u_right.y - u_left.y) * 0.5;
    let dx_dy = (u_up.x - u_down.x) * 0.5;
    let curl = dy_dx - dx_dy;

    let abs_c = abs(curl);
    let normalized = clamp((abs_c - MIN_THRESHOLD) / (SCALE_LIMIT - MIN_THRESHOLD), 0.0, 1.0);

    // Sign-preserving power scaling — expands small-curl detail
    let scaled = sign(curl) * pow(normalized, 0.5);

    let color = diverging(scaled);
    return vec4<f32>(color, 1.0);
}
"#;
