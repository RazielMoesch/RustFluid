pub(super) const COMPUTE_VORTICITY_3D: &str = r#"
@id(100) override NX: u32 = 128;
@id(101) override NY: u32 = 128;
@id(102) override NZ: u32 = 128;

@group(0) @binding(0) var<storage, read> macro_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var volume: texture_storage_3d<rgba32float, write>;

const FLAG_TYPE_SHIFT: u32 = 24u;

fn get_u(x: i32, y: i32, z: i32, is_valid: ptr<function, bool>) -> vec3<f32> {
    if (x < 0 || x >= i32(NX) || y < 0 || y >= i32(NY) || z < 0 || z >= i32(NZ)) {
        *is_valid = false;
        return vec3<f32>(0.0);
    }
    let idx = u32(x) + u32(y) * NX + u32(z) * NX * NY;
    
    let raw_flag = flags[idx];
    let boundary_type = raw_flag >> FLAG_TYPE_SHIFT;
    if (boundary_type == 1u) {
        *is_valid = false;
        return vec3<f32>(0.0);
    }
    return macro_data[idx].xyz;
}

fn get_u_smooth(x: i32, y: i32, z: i32, is_valid: ptr<function, bool>) -> vec3<f32> {
    var valid_c = true;
    let u_c = get_u(x, y, z, &valid_c);
    if (!valid_c) {
        *is_valid = false;
        return vec3<f32>(0.0);
    }
    
    var u_sum = vec3<f32>(0.0);
    var n = 0.0;
    
    var v = true;
    let u1 = get_u(x + 1, y, z, &v); if (v) { u_sum += u1; n += 1.0; }
    let u2 = get_u(x - 1, y, z, &v); if (v) { u_sum += u2; n += 1.0; }
    let u3 = get_u(x, y + 1, z, &v); if (v) { u_sum += u3; n += 1.0; }
    let u4 = get_u(x, y - 1, z, &v); if (v) { u_sum += u4; n += 1.0; }
    let u5 = get_u(x, y, z + 1, &v); if (v) { u_sum += u5; n += 1.0; }
    let u6 = get_u(x, y, z - 1, &v); if (v) { u_sum += u6; n += 1.0; }
    
    *is_valid = true;
    if (n > 0.0) {
        let alpha = 0.25; // 25% smoothing
        return u_c * (1.0 - alpha) + (u_sum / n) * alpha;
    }
    return u_c;
}

@compute @workgroup_size(8, 8, 2)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if (x >= NX || y >= NY || z >= NZ) {
        return;
    }

    let idx = x + y * NX + z * NX * NY;
    let raw_flag = flags[idx];
    let boundary_type = raw_flag >> FLAG_TYPE_SHIFT;
    let xi = i32(x);
    let yi = i32(y);
    let zi = i32(z);

    if (boundary_type == 1u) {
        // r: Q, g: solid mask, b: speed, a: valid
        textureStore(volume, vec3<i32>(xi, yi, zi), vec4<f32>(0.0, 1.0, 0.0, 0.0));
        return;
    }

    var valid = true;
    let u_c = get_u_smooth(xi, yi, zi, &valid);
    
    if (!valid) {
        textureStore(volume, vec3<i32>(xi, yi, zi), vec4<f32>(0.0, 0.0, length(u_c), 0.0));
        return;
    }

    var vr = true; let u_r = get_u_smooth(xi + 1, yi, zi, &vr);
    var vl = true; let u_l = get_u_smooth(xi - 1, yi, zi, &vl);
    var vt = true; let u_t = get_u_smooth(xi, yi + 1, zi, &vt);
    var vb = true; let u_b = get_u_smooth(xi, yi - 1, zi, &vb);
    var vf = true; let u_f = get_u_smooth(xi, yi, zi + 1, &vf);
    var vk = true; let u_k = get_u_smooth(xi, yi, zi - 1, &vk);

    var ux_x = 0.0; var uy_x = 0.0; var uz_x = 0.0;
    if (vr && vl) {
        ux_x = (u_r.x - u_l.x) * 0.5; uy_x = (u_r.y - u_l.y) * 0.5; uz_x = (u_r.z - u_l.z) * 0.5;
    } else if (vr) {
        ux_x = u_r.x - u_c.x; uy_x = u_r.y - u_c.y; uz_x = u_r.z - u_c.z;
    } else if (vl) {
        ux_x = u_c.x - u_l.x; uy_x = u_c.y - u_l.y; uz_x = u_c.z - u_l.z;
    }

    var ux_y = 0.0; var uy_y = 0.0; var uz_y = 0.0;
    if (vt && vb) {
        ux_y = (u_t.x - u_b.x) * 0.5; uy_y = (u_t.y - u_b.y) * 0.5; uz_y = (u_t.z - u_b.z) * 0.5;
    } else if (vt) {
        ux_y = u_t.x - u_c.x; uy_y = u_t.y - u_c.y; uz_y = u_t.z - u_c.z;
    } else if (vb) {
        ux_y = u_c.x - u_b.x; uy_y = u_c.y - u_b.y; uz_y = u_c.z - u_b.z;
    }

    var ux_z = 0.0; var uy_z = 0.0; var uz_z = 0.0;
    if (vf && vk) {
        ux_z = (u_f.x - u_k.x) * 0.5; uy_z = (u_f.y - u_k.y) * 0.5; uz_z = (u_f.z - u_k.z) * 0.5;
    } else if (vf) {
        ux_z = u_f.x - u_c.x; uy_z = u_f.y - u_c.y; uz_z = u_f.z - u_c.z;
    } else if (vk) {
        ux_z = u_c.x - u_k.x; uy_z = u_c.y - u_k.y; uz_z = u_c.z - u_k.z;
    }

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

    let norm_Omega2 = 2.0*(o_xy*o_xy + o_xz*o_xz + o_yz*o_yz);

    let q_crit = 0.5 * (norm_Omega2 - norm_S2);
    let speed = length(u_c);

    textureStore(volume, vec3<i32>(xi, yi, zi), vec4<f32>(q_crit, 0.0, speed, 1.0));
}
"#;

pub(super) const RENDER_VORTICITY_3D: &str = r#"
@id(100) override NX: u32 = 128;
@id(101) override NY: u32 = 128;
@id(102) override NZ: u32 = 128;

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>
};

@vertex
fn vs(@builtin(vertex_index) vidx: u32) -> VOut {
    var pos_arr = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );
    let p = pos_arr[vidx];

    var out: VOut;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = p; // -1 to 1
    return out;
}

struct Uniforms {
    inv_view_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    max_speed: f32,
    iso_q: f32,
    mode: u32,
    pad: f32,
};

@group(0) @binding(0) var volume_tex: texture_3d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var colormap_tex: texture_1d<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var depth_tex: texture_depth_2d;

fn intersect_aabb(ro: vec3<f32>, rd: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>, tN: ptr<function, f32>, tF: ptr<function, f32>) -> bool {
    let m = 1.0 / rd;
    let n = m * ro;
    let k = abs(m) * ((boxMax - boxMin) / 2.0);
    let t1 = -n - k + m * ((boxMax + boxMin) / 2.0);
    let t2 = -n + k + m * ((boxMax + boxMin) / 2.0);
    
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    
    let t0 = max(max(tmin.x, tmin.y), tmin.z);
    let t1_out = min(min(tmax.x, tmax.y), tmax.z);
    
    *tN = max(0.0, t0);
    *tF = t1_out;
    
    return t1_out > t0 && t1_out > 0.0;
}

fn sample_volume(p: vec3<f32>) -> vec4<f32> {
    let uvw = p; // p is in voxel coordinates [0, NX]
    if (uvw.x < 0.0 || uvw.x >= f32(NX)-1.0 || uvw.y < 0.0 || uvw.y >= f32(NY)-1.0 || uvw.z < 0.0 || uvw.z >= f32(NZ)-1.0) {
        return vec4<f32>(0.0);
    }
    
    // Correct half-voxel offset for textureLoad interpolation
    let p_shifted = uvw - vec3<f32>(0.5);
    let p0 = vec3<i32>(floor(p_shifted));
    let f = fract(p_shifted);
    
    let v000 = textureLoad(volume_tex, p0 + vec3<i32>(0, 0, 0), 0);
    let v100 = textureLoad(volume_tex, p0 + vec3<i32>(1, 0, 0), 0);
    let v010 = textureLoad(volume_tex, p0 + vec3<i32>(0, 1, 0), 0);
    let v110 = textureLoad(volume_tex, p0 + vec3<i32>(1, 1, 0), 0);
    let v001 = textureLoad(volume_tex, p0 + vec3<i32>(0, 0, 1), 0);
    let v101 = textureLoad(volume_tex, p0 + vec3<i32>(1, 0, 1), 0);
    let v011 = textureLoad(volume_tex, p0 + vec3<i32>(0, 1, 1), 0);
    let v111 = textureLoad(volume_tex, p0 + vec3<i32>(1, 1, 1), 0);
    
    let w000 = (1.0 - f.x) * (1.0 - f.y) * (1.0 - f.z) * v000.a;
    let w100 = f.x * (1.0 - f.y) * (1.0 - f.z) * v100.a;
    let w010 = (1.0 - f.x) * f.y * (1.0 - f.z) * v010.a;
    let w110 = f.x * f.y * (1.0 - f.z) * v110.a;
    let w001 = (1.0 - f.x) * (1.0 - f.y) * f.z * v001.a;
    let w101 = f.x * (1.0 - f.y) * f.z * v101.a;
    let w011 = (1.0 - f.x) * f.y * f.z * v011.a;
    let w111 = f.x * f.y * f.z * v111.a;

    let sum_w = w000 + w100 + w010 + w110 + w001 + w101 + w011 + w111;
    
    if (sum_w < 1e-4) {
        return vec4<f32>(0.0);
    }
    
    let val = (
        v000 * w000 + v100 * w100 + v010 * w010 + v110 * w110 +
        v001 * w001 + v101 * w101 + v011 * w011 + v111 * w111
    ) / sum_w;
    
    return vec4<f32>(val.rgb, 1.0);
}

fn get_gradient_q(p: vec3<f32>) -> vec3<f32> {
    let h = 1.0; // Must be >= 1.0 to smooth over trilinear cell boundary derivative discontinuities
    let dx = vec3<f32>(h, 0.0, 0.0);
    let dy = vec3<f32>(0.0, h, 0.0);
    let dz = vec3<f32>(0.0, 0.0, h);
    
    let grad = vec3<f32>(
        sample_volume(p + dx).r - sample_volume(p - dx).r,
        sample_volume(p + dy).r - sample_volume(p - dy).r,
        sample_volume(p + dz).r - sample_volume(p - dz).r
    );
    let l2 = dot(grad, grad);
    if (l2 < 1e-12) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(grad);
}

fn get_obstacle_normal(p: vec3<f32>) -> vec3<f32> {
    let h = 1.0;
    let dx = vec3<f32>(h, 0.0, 0.0);
    let dy = vec3<f32>(0.0, h, 0.0);
    let dz = vec3<f32>(0.0, 0.0, h);
    
    let grad = vec3<f32>(
        sample_volume(p + dx).g - sample_volume(p - dx).g,
        sample_volume(p + dy).g - sample_volume(p - dy).g,
        sample_volume(p + dz).g - sample_volume(p - dz).g
    );
    return normalize(-grad + vec3<f32>(0.0001));
}

fn hash13(p3: vec3<f32>) -> f32 {
    var p3_ = fract(p3 * 0.1031);
    p3_ += dot(p3_, p3_.zyx + 31.32);
    return fract((p3_.x + p3_.y) * p3_.z);
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
    let ndc = vec4<f32>(in.uv.x, in.uv.y, 0.5, 1.0);
    var ray_target = uniforms.inv_view_proj * ndc;
    ray_target = ray_target / ray_target.w;
    let ro = uniforms.eye.xyz;
    let rd = normalize(ray_target.xyz - ro);

    let boxMin = vec3<f32>(0.0, 0.0, 0.0);
    let boxMax = vec3<f32>(f32(NX), f32(NY), f32(NZ));

    var tN = 0.0;
    var tF = 0.0;
    
    if (!intersect_aabb(ro, rd, boxMin, boxMax, &tN, &tF)) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    tN = max(0.0, tN);

    // Read depth buffer and calculate world position to limit tF
    let dimensions = vec2<f32>(textureDimensions(depth_tex));
    let depth_val = textureLoad(depth_tex, vec2<i32>(in.position.xy), 0);
    if (depth_val < 1.0) {
        let ndc_depth = vec4<f32>(in.uv.x, in.uv.y, depth_val, 1.0);
        var world_pos = uniforms.inv_view_proj * ndc_depth;
        world_pos = world_pos / world_pos.w;
        let dist_to_mesh = distance(ro, world_pos.xyz);
        tF = min(tF, dist_to_mesh);
    }

    let dt = 0.5; // Half voxel step size for crisp surface
    let jitter = hash13(ro + rd * 100.0) * dt;
    var t_curr = tN + jitter;
    
    var acc_color = vec3<f32>(0.0);
    var acc_alpha = 0.0;
    
    let light_dir = normalize(vec3<f32>(0.5, 1.0, -0.2));
    
    var last_q = 0.0;
    var last_t = t_curr;
    var started = false;

    if (uniforms.mode == 2u) {
        // Diagnostic Mode: Central slice (Z = NZ/2)
        let z_plane = f32(NZ) * 0.5;
        if (abs(rd.z) > 0.001) {
            let t = (z_plane - ro.z) / rd.z;
            if (t > 0.0) {
                let p = ro + rd * t;
                if (p.x >= 0.0 && p.x < f32(NX) && p.y >= 0.0 && p.y < f32(NY)) {
                    let tex_val = sample_volume(p);
                    if (tex_val.a < 0.5) {
                        return vec4<f32>(1.0, 0.0, 1.0, 1.0); // Magenta for invalid
                    } else if (in.uv.x < 0.0) {
                        // Left half: signed Q slice
                        let q = tex_val.r;
                        if (q > 0.0) {
                            return vec4<f32>(q * 50000.0, 0.0, 0.0, 1.0); // Red positive
                        } else {
                            return vec4<f32>(0.0, 0.0, -q * 50000.0, 1.0); // Blue negative
                        }
                    } else {
                        // Right half: Speed slice
                        let speed = tex_val.b;
                        let norm = clamp(speed / uniforms.max_speed, 0.0, 1.0);
                        return textureSampleLevel(colormap_tex, samp, norm, 0.0);
                    }
                }
            }
        }
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    } else if (uniforms.mode == 1u) {
        // Volume mode (Smoke/Density)
        let g_phase = 0.3;
        let cos_theta = dot(rd, light_dir);
        let phase = (1.0 - g_phase*g_phase) / pow(1.0 + g_phase*g_phase - 2.0*g_phase*cos_theta, 1.5);
        
        for (var i = 0; i < 2000; i++) {
            if (t_curr > tF || acc_alpha >= 0.99) { break; }
            let p_curr = ro + rd * t_curr;
            let tex_val = sample_volume(p_curr);
            let q_val = tex_val.r;
            let solid = tex_val.g;
            let speed = tex_val.b;
            
            if (q_val > uniforms.iso_q) {
                let normalized_val = clamp(speed / uniforms.max_speed, 0.0, 1.0);
                let color_sample = textureSampleLevel(colormap_tex, samp, normalized_val, 0.0);
                let base_color = color_sample.rgb;
                let density = color_sample.a * 15.0; // scale density
                
                if (density > 0.0) {
                    let alpha = 1.0 - exp(-density * dt);
                    
                    var shadow_den = 0.0;
                    let sdt = dt * 2.5; 
                    var sp = p_curr + light_dir * (sdt + hash13(p_curr * 10.0) * sdt);
                    for (var j = 0; j < 6; j++) {
                        let sv = sample_volume(sp).r;
                        if (sv > uniforms.iso_q) {
                            let snormalized = clamp(sample_volume(sp).b / uniforms.max_speed, 0.0, 1.0);
                            let scolor = textureSampleLevel(colormap_tex, samp, snormalized, 0.0);
                            shadow_den += scolor.a * 15.0 * sdt;
                        }
                        sp += light_dir * sdt;
                    }
                    let transmission = exp(-shadow_den * 1.2);
                    
                    let normal = -get_gradient_q(p_curr);
                    let directional = max(dot(normal, light_dir), 0.0) * 0.4 + 0.6;
                    
                    let ambient = vec3<f32>(0.02, 0.04, 0.08) * density;
                    let emission_strength = pow(normalized_val, 2.0) * 8.0;
                    let emission = base_color * emission_strength;
                    let scattered = base_color * transmission * directional * phase * 2.0;
                    
                    let final_light = scattered + ambient + emission;
                    
                    acc_color += final_light * alpha * (1.0 - acc_alpha);
                    acc_alpha += alpha * (1.0 - acc_alpha);
                }
            }
            t_curr += dt;
        }
    } else {
        // Isosurface Mode
        let dt_iso = 0.25; // Smaller step size to catch thin vortex sheets
        var t_curr = tN; // No jitter for deterministic solid surface
        var last_q = 0.0;
        var last_t = t_curr;
        var started = false;

        for (var i = 0; i < 4000; i++) {
        if (t_curr > tF) { break; }
        let p_curr = ro + rd * t_curr;
        let tex_val = sample_volume(p_curr);
        
        let q_val = tex_val.r;
        let solid = tex_val.g;
        let speed = tex_val.b;
        let valid = tex_val.a;
        
        if (valid > 0.5) {
            if (q_val >= uniforms.iso_q) {
                if (started && last_q < uniforms.iso_q) {
                    // Bisection refinement for sub-voxel precision
                    var lo = last_t;
                    var hi = t_curr;
                    for (var j = 0; j < 5; j++) {
                        let mid = 0.5 * (lo + hi);
                        let q_mid = sample_volume(ro + rd * mid).r;
                        if (q_mid >= uniforms.iso_q) {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    let hit_t = 0.5 * (lo + hi);
                    
                    let hit_p = ro + rd * hit_t;
                    let hit_speed = sample_volume(hit_p).b;
                    
                    let normal = -get_gradient_q(hit_p); // Negative Q gradient points outwards
                    let view_dir = normalize(ro - hit_p);
                    
                    let normalized_speed = clamp(hit_speed / uniforms.max_speed, 0.0, 1.0);
                    let base_color = textureSampleLevel(colormap_tex, samp, normalized_speed, 0.0).rgb;
                    
                    let diff = max(dot(normal, light_dir), 0.0) * 0.8 + 0.2;
                    let half_vec = normalize(light_dir + view_dir);
                    let spec = pow(max(dot(normal, half_vec), 0.0), 16.0) * 0.15; // Soft specular, not white blown out
                    
                    let emission = base_color * normalized_speed * 0.2; // Tiny emission so we can see inside dark areas
                    let final_color = base_color * diff + vec3<f32>(spec) + emission;
                    
                    acc_color = final_color;
                    acc_alpha = 1.0;
                    break;
                }
            }
            last_q = q_val;
            last_t = t_curr;
            started = true;
        } else {
            started = false;
        }
        
        t_curr += dt_iso;
    }
    }

    // Convert to sRGB visually (if not handled by render target)
    acc_color = ACESFilm(acc_color * 1.5);
    
    return vec4<f32>(acc_color, acc_alpha);
}
"#;
