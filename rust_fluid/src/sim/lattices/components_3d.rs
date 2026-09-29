// ═══════════════════════════════════════════════════════════════════════════
//  BOUNDARY CONDITIONS (A-A PATTERN: EVEN AND ODD VARIANTS)
// ═══════════════════════════════════════════════════════════════════════════

// --------------------------------------- FLUID BOUNDARIES ---------------------------------------
pub(super) const FLUID_PULL_STREAMING_EVEN: &str = r#"
    pulled_f = load_fa(neighbour_idx + i * TOTAL_CELLS);
"#;

pub(super) const FLUID_PULL_STREAMING_ODD: &str = r#"
    pulled_f = load_fa(neighbour_idx + OPP[i] * TOTAL_CELLS);
"#;

// --------------------------------------- SOLID BOUNDARIES ---------------------------------------
pub(super) const SOLID_BOUNCE_BACK_EVEN: &str = r#"
    // Pull from our own cell's opposite direction 
    pulled_f = load_fa(cell_idx + OPP[i] * TOTAL_CELLS);
"#;

pub(super) const SOLID_BOUNCE_BACK_ODD: &str = r#"
    // Pull from our own cell's opposite direction (stored non-inverted in this step)
    pulled_f = load_fa(cell_idx + i * TOTAL_CELLS);
"#;

pub(super) const FREE_SLIP_Y_EVEN: &str = r#"
    // Specular reflection for top/bottom walls (Flips Y, preserves X, Z)
    pulled_f = load_fa(cell_idx + REFLECT_Y[i] * TOTAL_CELLS);
"#;

pub(super) const FREE_SLIP_Y_ODD: &str = r#"
    // Inverted memory read for the specular reflection
    pulled_f = load_fa(cell_idx + OPP[REFLECT_Y[i]] * TOTAL_CELLS);
"#;

pub(super) const FREE_SLIP_X_EVEN: &str = r#"
    // Specular reflection for left/right walls (Flips X, preserves Y, Z)
    pulled_f = load_fa(cell_idx + REFLECT_X[i] * TOTAL_CELLS);
"#;

pub(super) const FREE_SLIP_X_ODD: &str = r#"
    // Inverted memory read for the specular reflection
    pulled_f = load_fa(cell_idx + OPP[REFLECT_X[i]] * TOTAL_CELLS);
"#;

pub(super) const FREE_SLIP_Z_EVEN: &str = r#"
    // Specular reflection for front/back walls (Flips Z, preserves X, Y)
    pulled_f = load_fa(cell_idx + REFLECT_Z[i] * TOTAL_CELLS);
"#;

pub(super) const FREE_SLIP_Z_ODD: &str = r#"
    // Inverted memory read for the specular reflection
    pulled_f = load_fa(cell_idx + OPP[REFLECT_Z[i]] * TOTAL_CELLS);
"#;

// --------------------------------------- INLET BOUNDARIES ---------------------------------------
pub(super) const INLET_EQUILIBRIUM_EVEN: &str = r#"
    let cfg = boundary_configs[cfg_id];
    let eu = (f32(ex) * cfg.vel.x) + (f32(ey) * cfg.vel.y) + (f32(ez) * cfg.vel.z);
    let u_sq = dot(cfg.vel, cfg.vel);
    pulled_f = WEIGHTS[i] * cfg.density * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - 1.5 * u_sq);
"#;

pub(super) const INLET_EQUILIBRIUM_ODD: &str = r#"
    let cfg = boundary_configs[cfg_id];
    let eu = (f32(ex) * cfg.vel.x) + (f32(ey) * cfg.vel.y) + (f32(ez) * cfg.vel.z);
    let u_sq = dot(cfg.vel, cfg.vel);
    pulled_f = WEIGHTS[i] * cfg.density * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - 1.5 * u_sq);
"#;

// --------------------------------------- POST-STREAMING BOUNDARIES ---------------------------------------
pub(super) const ZOU_HE_LEFT_VELOCITY: &str = r#"
    let my_flag = flags[cell_idx];
    let my_type = my_flag >> FLAG_TYPE_SHIFT;
    
    if (my_type == 5u) {
        let cfg_id = my_flag & FLAG_ID_MASK;
        let cfg = boundary_configs[cfg_id];
        let ux = cfg.vel.x;
        let uy = cfg.vel.y;
        let uz = cfg.vel.z;
        
        let rho_in = (f_local[0] + f_local[3] + f_local[4] + f_local[5] + f_local[6] + 
                      f_local[15] + f_local[16] + f_local[17] + f_local[18] + 
                      2.0 * (f_local[2] + f_local[8] + f_local[10] + f_local[12] + f_local[14])) / (1.0 - ux);
        
        f_local[1] = f_local[2] + (1.0 / 3.0) * rho_in * ux;
        
        let diff_y = 0.5 * (f_local[4] - f_local[3]);
        let diff_z = 0.5 * (f_local[6] - f_local[5]);
        
        f_local[7]  = f_local[10] + diff_y + (1.0 / 6.0) * rho_in * ux + 0.5 * rho_in * uy;
        f_local[9]  = f_local[8]  - diff_y + (1.0 / 6.0) * rho_in * ux - 0.5 * rho_in * uy;
        f_local[11] = f_local[14] + diff_z + (1.0 / 6.0) * rho_in * ux + 0.5 * rho_in * uz;
        f_local[13] = f_local[12] - diff_z + (1.0 / 6.0) * rho_in * ux - 0.5 * rho_in * uz;
        
        rho = rho_in;
        u = cfg.vel * rho_in;
    }
"#;

// --------------------------------------- OUTLET BOUNDARIES ---------------------------------------

pub(super) const OUTLET_ZERO_GRADIENT_EVEN: &str = r#"
    // Pull from our own cell in the same direction (zero gradient extrapolation)
    pulled_f = load_fa(cell_idx + i * TOTAL_CELLS);
"#;

pub(super) const OUTLET_ZERO_GRADIENT_ODD: &str = r#"
    // Pull from our own cell in the same direction
    pulled_f = load_fa(cell_idx + OPP[i] * TOTAL_CELLS);
"#;

// ═══════════════════════════════════════════════════════════════════════════
//  COLLISION LOGIC (A-A Pattern compatible, operates strictly on f_local)
// ═══════════════════════════════════════════════════════════════════════════

pub(super) const BGK_COLLISION: &str = r#"
    if (rho > 0.0) {
        u = u / rho;
    }

    let force = vec3<f32>(FORCE_X, FORCE_Y, FORCE_Z);
    u += force * 0.5;

    let u_sq = dot(u, u);
    let u_sq_term = 1.5 * u_sq;

    for (var i: u32 = 0u; i < Q; i += 1u) {
        let eu = (f32(EX[i]) * u.x) + (f32(EY[i]) * u.y) + (f32(EZ[i]) * u.z);
        let feq = WEIGHTS[i] * rho * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - u_sq_term);
        f_local[i] = f_local[i] - OMEGA * (f_local[i] - feq);
    }
"#;

pub(super) const MRT_COLLISION: &str = r#"
    if (rho > 0.0) {
        u = u / rho;
    }

    let force = vec3<f32>(FORCE_X, FORCE_Y, FORCE_Z);
    u += force * 0.5;

    let u_sq = dot(u, u);
    let u_sq_term = 1.5 * u_sq;
    let OMEGA_EFF = OMEGA;

    // Two-Relaxation-Time (TRT) approximation for D3Q19 which achieves MRT stability.
    let s_plus = OMEGA_EFF;
    let s_minus = 8.0 * (2.0 - s_plus) / (8.0 - s_plus);
    
    var f_pre = f_local;

    for (var i: u32 = 0u; i < 19u; i += 1u) {
        let eu = (f32(EX[i]) * u.x) + (f32(EY[i]) * u.y) + (f32(EZ[i]) * u.z);
        let feq = WEIGHTS[i] * rho * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - u_sq_term);
        
        let opp = OPP[i];
        let eu_opp = (f32(EX[opp]) * u.x) + (f32(EY[opp]) * u.y) + (f32(EZ[opp]) * u.z);
        let feq_opp = WEIGHTS[opp] * rho * (1.0 + 3.0 * eu_opp + 4.5 * (eu_opp * eu_opp) - u_sq_term);
        
        let f_plus = 0.5 * (f_pre[i] + f_pre[opp]);
        let f_minus = 0.5 * (f_pre[i] - f_pre[opp]);
        
        let feq_plus = 0.5 * (feq + feq_opp);
        let feq_minus = 0.5 * (feq - feq_opp);
        
        f_local[i] = f_local[i] - s_plus * (f_plus - feq_plus) - s_minus * (f_minus - feq_minus);
    }
"#;

// ═══════════════════════════════════════════════════════════════════════════
//  BASE TEMPLATES (EVEN, ODD, INIT, EXTRACT)
// ═══════════════════════════════════════════════════════════════════════════

pub(super) const BASE_STEP_EVEN_3D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(110) override NZ: u32 = 1080;
@id(102) override OMEGA: f32 = 1.0;
@id(103) override wgs_x: u32 = 8;
@id(104) override wgs_y: u32 = 8;
@id(111) override wgs_z: u32 = 1;
@id(120) override FORCE_X: f32 = 0.0;
@id(121) override FORCE_Y: f32 = 0.0;
@id(122) override FORCE_Z: f32 = 0.0;
@id(123) override PERIODIC_X: u32 = 0u;

override TOTAL_CELLS = NX * NY * NZ;

//{Q}
const FLAG_TYPE_SHIFT: u32 = 24u;
const FLAG_ID_MASK: u32 = 0x00FFFFFFu;
//{EX}
//{EY}
//{EZ}
//{WEIGHTS}
//{OPP}
//{REFLECT_Y}
//{REFLECT_X}
//{REFLECT_Z}

struct BoundaryConfig {
    vel: vec3<f32>,
    density: f32,
}

@group(0) @binding(0) var<storage, read> fa: array<f32>; // POP_FA
@group(0) @binding(1) var<storage, read_write> fb: array<f32>; // POP_FB
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> boundary_configs: array<BoundaryConfig>;

//{PRECISION_HELPERS}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= NX || y >= NY || z >= NZ) { return; }

    let cell_idx = x + (y * NX) + (z * NX * NY);
    let cell_pos = vec3<i32>(i32(x), i32(y), i32(z));
    var f_local: array<f32, Q>;
    var rho: f32 = 0.0;
    var u = vec3<f32>(0.0, 0.0, 0.0);

    // --- STREAMING (PULL) ---
    for (var i: u32 = 0u; i < Q; i += 1u) {
        let ex = EX[i];
        let ey = EY[i];
        let ez = EZ[i];
        let neighbour_pos = cell_pos - vec3<i32>(ex, ey, ez);
        var pulled_f: f32 = 0.0;

        var nx_pos = neighbour_pos.x;
        var ny_pos = neighbour_pos.y;
        var nz_pos = neighbour_pos.z;

        if (PERIODIC_X == 1u) {
            if (nx_pos < 0) { nx_pos += i32(NX); }
            else if (nx_pos >= i32(NX)) { nx_pos -= i32(NX); }
            
            if (ny_pos < 0) { ny_pos += i32(NY); }
            else if (ny_pos >= i32(NY)) { ny_pos -= i32(NY); }
            
            if (nz_pos < 0) { nz_pos += i32(NZ); }
            else if (nz_pos >= i32(NZ)) { nz_pos -= i32(NZ); }
        }

        if (nx_pos < 0 || ny_pos < 0 || nz_pos < 0 || nx_pos >= i32(NX) || ny_pos >= i32(NY) || nz_pos >= i32(NZ)) {
            // Out of bounds domain limits
            //{BOUNCE_BACK_LOGIC_EVEN}
        } else {
            let neighbour_idx = u32(nx_pos) + (u32(ny_pos) * NX) + (u32(nz_pos) * NX * NY);
            let raw_flag = flags[neighbour_idx];
            let boundary_type = raw_flag >> FLAG_TYPE_SHIFT;
            let cfg_id = raw_flag & FLAG_ID_MASK;

            switch boundary_type {
                case 0u: {
                    //{FLUID_PULL_LOGIC_EVEN}
                }
                case 1u: {
                    //{BOUNCE_BACK_LOGIC_EVEN}
                }
                case 2u: {
                    //{EQUILIBRIUM_INLET_LOGIC_EVEN}
                }
                case 3u: {
                    //{ZERO_GRADIENT_OUTLET_LOGIC_EVEN}
                }
                case 4u: {
                    //{FREE_SLIP_Y_LOGIC_EVEN}
                }
                case 5u: {
                    //{FLUID_PULL_LOGIC_EVEN}
                }
                case 6u: {
                    //{FREE_SLIP_X_LOGIC_EVEN}
                }
                case 7u: {
                    //{FREE_SLIP_Z_LOGIC_EVEN}
                }
                default: {
                    pulled_f = 0.0;
                }
            }
        }

        f_local[i] = pulled_f;
        rho += pulled_f;
        u += vec3<f32>(f32(ex), f32(ey), f32(ez)) * pulled_f;
    }

    // --- POST-STREAMING CORRECTION ---
    //{POST_STREAMING_CORRECTION}

    // --- COLLISION ---
    //{COLLISION_LOGIC}

    // --- WRITE (INVERTED) ---
    for (var i: u32 = 0u; i < Q; i += 1u) {
        store_fb(cell_idx + OPP[i] * TOTAL_CELLS, f_local[i]);
    }
}
"#;

pub(super) const BASE_STEP_ODD_3D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(110) override NZ: u32 = 1080;
@id(102) override OMEGA: f32 = 1.0;
@id(103) override wgs_x: u32 = 8;
@id(104) override wgs_y: u32 = 8;
@id(111) override wgs_z: u32 = 1;
@id(120) override FORCE_X: f32 = 0.0;
@id(121) override FORCE_Y: f32 = 0.0;
@id(122) override FORCE_Z: f32 = 0.0;
@id(123) override PERIODIC_X: u32 = 0u;

override TOTAL_CELLS = NX * NY * NZ;

//{Q}
const FLAG_TYPE_SHIFT: u32 = 24u;
const FLAG_ID_MASK: u32 = 0x00FFFFFFu;
//{EX}
//{EY}
//{EZ}
//{WEIGHTS}
//{OPP}
//{REFLECT_Y}
//{REFLECT_X}
//{REFLECT_Z}

struct BoundaryConfig {
    vel: vec3<f32>,
    density: f32,
}

@group(0) @binding(0) var<storage, read> fa: array<f32>; // POP_FA
@group(0) @binding(1) var<storage, read_write> fb: array<f32>; // POP_FB
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> boundary_configs: array<BoundaryConfig>;

//{PRECISION_HELPERS}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= NX || y >= NY || z >= NZ) { return; }

    let cell_idx = x + (y * NX) + (z * NX * NY);
    let cell_pos = vec3<i32>(i32(x), i32(y), i32(z));
    var f_local: array<f32, Q>;
    var rho: f32 = 0.0;
    var u = vec3<f32>(0.0, 0.0, 0.0);

    // --- STREAMING (PULL) ---
    for (var i: u32 = 0u; i < Q; i += 1u) {
        let ex = EX[i];
        let ey = EY[i];
        let ez = EZ[i];
        let neighbour_pos = cell_pos - vec3<i32>(ex, ey, ez);
        var pulled_f: f32 = 0.0;

        var nx_pos = neighbour_pos.x;
        var ny_pos = neighbour_pos.y;
        var nz_pos = neighbour_pos.z;

        if (PERIODIC_X == 1u) {
            if (nx_pos < 0) { nx_pos += i32(NX); }
            else if (nx_pos >= i32(NX)) { nx_pos -= i32(NX); }
            
            if (ny_pos < 0) { ny_pos += i32(NY); }
            else if (ny_pos >= i32(NY)) { ny_pos -= i32(NY); }
            
            if (nz_pos < 0) { nz_pos += i32(NZ); }
            else if (nz_pos >= i32(NZ)) { nz_pos -= i32(NZ); }
        }

        if (nx_pos < 0 || ny_pos < 0 || nz_pos < 0 || nx_pos >= i32(NX) || ny_pos >= i32(NY) || nz_pos >= i32(NZ)) {
            // Out of bounds domain limits
            //{BOUNCE_BACK_LOGIC_ODD}
        } else {
            let neighbour_idx = u32(nx_pos) + (u32(ny_pos) * NX) + (u32(nz_pos) * NX * NY);
            let raw_flag = flags[neighbour_idx];
            let boundary_type = raw_flag >> FLAG_TYPE_SHIFT;
            let cfg_id = raw_flag & FLAG_ID_MASK;

            switch boundary_type {
                case 0u: {
                    //{FLUID_PULL_LOGIC_ODD}
                }
                case 1u: {
                    //{BOUNCE_BACK_LOGIC_ODD}
                }
                case 2u: {
                    //{EQUILIBRIUM_INLET_LOGIC_ODD}
                }
                case 3u: {
                    //{ZERO_GRADIENT_OUTLET_LOGIC_ODD}
                }
                case 4u: {
                    //{FREE_SLIP_Y_LOGIC_ODD}
                }
                case 5u: {
                    //{FLUID_PULL_LOGIC_ODD}
                }
                case 6u: {
                    //{FREE_SLIP_X_LOGIC_ODD}
                }
                case 7u: {
                    //{FREE_SLIP_Z_LOGIC_ODD}
                }
                default: {
                    pulled_f = 0.0;
                }
            }
        }

        f_local[i] = pulled_f;
        rho += pulled_f;
        u += vec3<f32>(f32(ex), f32(ey), f32(ez)) * pulled_f;
    }

    // --- POST-STREAMING CORRECTION ---
    //{POST_STREAMING_CORRECTION}

    // --- COLLISION ---
    //{COLLISION_LOGIC}

    // --- WRITE (NON-INVERTED) ---
    for (var i: u32 = 0u; i < Q; i += 1u) {
        store_fb(cell_idx + i * TOTAL_CELLS, f_local[i]);
    }
}
"#;

pub(super) const PURE_FLUID_STEP_EVEN_3D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(110) override NZ: u32 = 1080;
@id(102) override OMEGA: f32 = 1.0;
@id(103) override wgs_x: u32 = 8;
@id(104) override wgs_y: u32 = 8;
@id(111) override wgs_z: u32 = 1;

override TOTAL_CELLS = NX * NY * NZ;

//{Q}
//{EX}
//{EY}
//{EZ}
//{WEIGHTS}
//{OPP}

@group(0) @binding(0) var<storage, read> fa: array<f32>; // POP_FA
@group(0) @binding(1) var<storage, read_write> fb: array<f32>; // POP_FB

//{PRECISION_HELPERS}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= NX || y >= NY || z >= NZ) { return; }

    let cell_idx = x + (y * NX) + (z * NX * NY);
    let cell_pos = vec3<i32>(i32(x), i32(y), i32(z));
    var f_local: array<f32, Q>;
    var rho: f32 = 0.0;
    var u = vec3<f32>(0.0, 0.0, 0.0);

    for (var i: u32 = 0u; i < Q; i += 1u) {
        let ex = EX[i];
        let ey = EY[i];
        let ez = EZ[i];
        
        var nx_pos = i32(x) - ex;
        var ny_pos = i32(y) - ey;
        var nz_pos = i32(z) - ez;

        // Periodic boundaries
        if (nx_pos < 0) { nx_pos += i32(NX); } else if (nx_pos >= i32(NX)) { nx_pos -= i32(NX); }
        if (ny_pos < 0) { ny_pos += i32(NY); } else if (ny_pos >= i32(NY)) { ny_pos -= i32(NY); }
        if (nz_pos < 0) { nz_pos += i32(NZ); } else if (nz_pos >= i32(NZ)) { nz_pos -= i32(NZ); }

        let neighbour_idx = u32(nx_pos) + (u32(ny_pos) * NX) + (u32(nz_pos) * NX * NY);
        
        var pulled_f: f32;
        //{FLUID_PULL_LOGIC_EVEN}
        
        f_local[i] = pulled_f;
        rho += pulled_f;
        u += vec3<f32>(f32(ex), f32(ey), f32(ez)) * pulled_f;
    }

    //{COLLISION_LOGIC}

    for (var i: u32 = 0u; i < Q; i += 1u) {
        store_fb(cell_idx + OPP[i] * TOTAL_CELLS, f_local[i]);
    }
}
"#;

pub(super) const PURE_FLUID_STEP_ODD_3D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(110) override NZ: u32 = 1080;
@id(102) override OMEGA: f32 = 1.0;
@id(103) override wgs_x: u32 = 8;
@id(104) override wgs_y: u32 = 8;
@id(111) override wgs_z: u32 = 1;

override TOTAL_CELLS = NX * NY * NZ;

//{Q}
//{EX}
//{EY}
//{EZ}
//{WEIGHTS}
//{OPP}

@group(0) @binding(0) var<storage, read> fa: array<f32>; // POP_FA
@group(0) @binding(1) var<storage, read_write> fb: array<f32>; // POP_FB

//{PRECISION_HELPERS}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    let z = id.z;

    if (x >= NX || y >= NY || z >= NZ) { return; }

    let cell_idx = x + (y * NX) + (z * NX * NY);
    let cell_pos = vec3<i32>(i32(x), i32(y), i32(z));
    var f_local: array<f32, Q>;
    var rho: f32 = 0.0;
    var u = vec3<f32>(0.0, 0.0, 0.0);

    for (var i: u32 = 0u; i < Q; i += 1u) {
        let ex = EX[i];
        let ey = EY[i];
        let ez = EZ[i];
        
        var nx_pos = i32(x) - ex;
        var ny_pos = i32(y) - ey;
        var nz_pos = i32(z) - ez;

        // Periodic boundaries
        if (nx_pos < 0) { nx_pos += i32(NX); } else if (nx_pos >= i32(NX)) { nx_pos -= i32(NX); }
        if (ny_pos < 0) { ny_pos += i32(NY); } else if (ny_pos >= i32(NY)) { ny_pos -= i32(NY); }
        if (nz_pos < 0) { nz_pos += i32(NZ); } else if (nz_pos >= i32(NZ)) { nz_pos -= i32(NZ); }

        let neighbour_idx = u32(nx_pos) + (u32(ny_pos) * NX) + (u32(nz_pos) * NX * NY);
        
        var pulled_f: f32;
        //{FLUID_PULL_LOGIC_ODD}
        
        f_local[i] = pulled_f;
        rho += pulled_f;
        u += vec3<f32>(f32(ex), f32(ey), f32(ez)) * pulled_f;
    }

    //{COLLISION_LOGIC}

    for (var i: u32 = 0u; i < Q; i += 1u) {
        store_fb(cell_idx + i * TOTAL_CELLS, f_local[i]);
    }
}
"#;

pub(super) const BASE_INIT_3D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(110) override NZ: u32 = 1080;

@id(102) override RHO_INIT: f32 = 1.0;
@id(103) override U_X_INIT: f32 = 0.0;
@id(104) override U_Y_INIT: f32 = 0.0;
@id(112) override U_Z_INIT: f32 = 0.0;

@id(105) override wgs_x: u32 = 8;
@id(106) override wgs_y: u32 = 8;
@id(111) override wgs_z: u32 = 1;

override TOTAL_CELLS = NX * NY * NZ;
//{Q}

const FLAG_TYPE_SHIFT: u32 = 24u;

//{EX}
//{EY}
//{EZ}

//{WEIGHTS}

@group(0) @binding(0) var<storage, read_write> fa: array<f32>; // POP_STORAGE
@group(0) @binding(2) var<storage, read> flags: array<u32>;

//{PRECISION_HELPERS}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

    let x = id.x;
    let y = id.y;
    let z = id.z;

    if ( x >= NX || y >= NY || z >= NZ ) {
        return;
    }

    let cell_idx = x + ( y * NX ) + (z * NX * NY);
    let raw_flag = flags[cell_idx];
    let boundary_type = raw_flag >> FLAG_TYPE_SHIFT;

    var rho = RHO_INIT;
    var u = vec3<f32>(U_X_INIT, U_Y_INIT, U_Z_INIT);

    if ( boundary_type == 1u ) {
        u = vec3<f32>(0.0, 0.0, 0.0);
    }

    let u_sq = dot( u, u );
    let u_sq_term = 1.5 * u_sq;
    for ( var i=0u; i < Q; i += 1u ) {
        let eu = ( f32(EX[i]) * u.x ) + ( f32(EY[i]) * u.y ) + ( f32(EZ[i]) * u.z );
        let feq = WEIGHTS[i] * rho * ( 1.0 + 3.0 * eu + 4.5 * ( eu * eu ) - u_sq_term );
        store_fa(cell_idx + i * TOTAL_CELLS, feq);
    }

}
"#;

pub(super) const BASE_EXTRACT_3D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(110) override NZ: u32 = 1080;
@id(105) override wgs_x: u32 = 8;
@id(106) override wgs_y: u32 = 8;
@id(111) override wgs_z: u32 = 1;
@id(107) override is_even: u32 = 1u;

override TOTAL_CELLS = NX * NY * NZ;

//{Q}
//{EX}
//{EY}
//{EZ}
//{OPP}

@group(0) @binding(0) var<storage, read> fa: array<f32>; // POP_FA
@group(0) @binding(1) var<storage, read_write> macro_data: array<vec4<f32>>;

//{PRECISION_HELPERS}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

    let x = id.x;
    let y = id.y;
    let z = id.z;

    if ( x >= NX || y >= NY || z >= NZ ) {
        return;
    }

    let cell_idx = x + ( y * NX ) + (z * NX * NY);
    var rho = 0.0;
    var u = vec3<f32>(0.0, 0.0, 0.0);

    for (var i: u32 = 0u; i < Q; i += 1u) {

        let read_idx = select(i, OPP[i], is_even != 0u);
        let pop = load_fa(read_idx * TOTAL_CELLS + cell_idx);
        rho += pop;
        u += vec3<f32>(f32(EX[i]), f32(EY[i]), f32(EZ[i])) * pop;

    } 

    if ( rho > 0.0 ) {
        u = u / rho;
    }

    macro_data[cell_idx] = vec4<f32>(u.x, u.y, u.z, rho);

}
"#;

// ═══════════════════════════════════════════════════════════════════════════
//  LATTICE SPECIFIC CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

// --------------------------------------- D3Q19 CONSTANTS ---------------------------------------
pub(super) const D3Q19_Q: &str = "const Q: u32 = 19u;";

pub(super) const D3Q19_EX: &str = r#"
const EX = array<i32, 19>(
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0
);
"#;

pub(super) const D3Q19_EY: &str = r#"
const EY = array<i32, 19>(
    0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1
);
"#;

pub(super) const D3Q19_EZ: &str = r#"
const EZ = array<i32, 19>(
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1
);
"#;

pub(super) const D3Q19_WEIGHTS: &str = r#"
const WEIGHTS = array<f32, 19>(
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
);
"#;

pub(super) const D3Q19_OPP: &str = r#"
const OPP = array<u32, 19>(
    0u, 2u, 1u, 4u, 3u, 6u, 5u, 10u, 9u, 8u, 7u, 14u, 13u, 12u, 11u, 18u, 17u, 16u, 15u
);
"#;

pub(super) const D3Q19_REFLECT_X: &str = r#"
const REFLECT_X = array<u32, 19>(
    0u, 2u, 1u, 3u, 4u, 5u, 6u, 8u, 7u, 10u, 9u, 12u, 11u, 14u, 13u, 15u, 16u, 17u, 18u
);
"#;

pub(super) const D3Q19_REFLECT_Y: &str = r#"
const REFLECT_Y = array<u32, 19>(
    0u, 1u, 2u, 4u, 3u, 5u, 6u, 9u, 10u, 7u, 8u, 11u, 12u, 13u, 14u, 16u, 15u, 18u, 17u
);
"#;

pub(super) const D3Q19_REFLECT_Z: &str = r#"
const REFLECT_Z = array<u32, 19>(
    0u, 1u, 2u, 3u, 4u, 6u, 5u, 7u, 8u, 9u, 10u, 13u, 14u, 11u, 12u, 17u, 18u, 15u, 16u
);
"#;
