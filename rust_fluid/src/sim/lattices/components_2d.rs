// --------------------------------------- D2Q9 CONSTANTS ---------------------------------------
pub(super) const D2Q9_Q: &str = "const Q: u32 = 9u;";

pub(super) const D2Q9_EX: &str = r#"
const EX = array<i32, 9>(
    0, 1, 0, -1, 0, 1, -1, -1, 1
);
"#;

pub(super) const D2Q9_EY: &str = r#"
const EY = array<i32, 9>(
    0, 0, 1, 0, -1, 1, 1, -1, -1
);
"#;

pub(super) const D2Q9_WEIGHTS: &str = r#"
const WEIGHTS = array<f32, 9>(
    4.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
);
"#;

pub(super) const D2Q9_OPP: &str = r#"
const OPP = array<u32, 9>(
    0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u
);
"#;

// ═══════════════════════════════════════════════════════════════════════════
//  BOUNDARY CONDITIONS (A-A PATTERN: EVEN AND ODD VARIANTS)
// ═══════════════════════════════════════════════════════════════════════════

// --------------------------------------- FLUID BOUNDARIES ---------------------------------------
pub(super) const FLUID_PULL_STREAMING_EVEN: &str = r#"
    pulled_f = fa[neighbour_idx + i * TOTAL_CELLS];
"#;

pub(super) const FLUID_PULL_STREAMING_ODD: &str = r#"
    pulled_f = fa[neighbour_idx + OPP[i] * TOTAL_CELLS];
"#;


// --------------------------------------- SOLID BOUNDARIES ---------------------------------------
pub(super) const SOLID_BOUNCE_BACK_EVEN: &str = r#"
    // Pull from our own cell's opposite direction 
    pulled_f = fa[cell_idx + OPP[i] * TOTAL_CELLS];
"#;

pub(super) const SOLID_BOUNCE_BACK_ODD: &str = r#"
    // Pull from our own cell's opposite direction (stored non-inverted in this step)
    pulled_f = fa[cell_idx + i * TOTAL_CELLS];
"#;


// --------------------------------------- INLET BOUNDARIES ---------------------------------------
pub(super) const INLET_EQUILIBRIUM_EVEN: &str = r#"
    let eu = (f32(ex) * cfg.vel.x) + (f32(ey) * cfg.vel.y);
    let u_sq = dot(cfg.vel, cfg.vel);
    pulled_f = WEIGHTS[i] * cfg.density * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - 1.5 * u_sq);
"#;

pub(super) const INLET_EQUILIBRIUM_ODD: &str = r#"
    let eu = (f32(ex) * cfg.vel.x) + (f32(ey) * cfg.vel.y);
    let u_sq = dot(cfg.vel, cfg.vel);
    pulled_f = WEIGHTS[i] * cfg.density * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - 1.5 * u_sq);
"#;


// --------------------------------------- OUTLET BOUNDARIES ---------------------------------------
pub(super) const OUTLET_ZERO_GRADIENT_EVEN: &str = r#"
    // Pull from our own cell in the same direction (zero gradient extrapolation)
    pulled_f = fa[cell_idx + i * TOTAL_CELLS];
"#;

pub(super) const OUTLET_ZERO_GRADIENT_ODD: &str = r#"
    // Pull from our own cell in the same direction
    pulled_f = fa[cell_idx + OPP[i] * TOTAL_CELLS];
"#;


// ═══════════════════════════════════════════════════════════════════════════
//  COLLISION LOGIC (A-A Pattern compatible, operates strictly on f_local)
// ═══════════════════════════════════════════════════════════════════════════

pub(super) const BGK_COLLISION: &str = r#"
    if (rho > 0.0) {
        u = u / rho;
    }

    let u_sq = dot(u, u);
    let u_sq_term = 1.5 * u_sq;

    for (var i: u32 = 0u; i < Q; i += 1u) {
        let eu = (f32(EX[i]) * u.x) + (f32(EY[i]) * u.y);
        let feq = WEIGHTS[i] * rho * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - u_sq_term);
        f_local[i] = f_local[i] - OMEGA * (f_local[i] - feq);
    }
"#;

pub(super) const MRT_COLLISION: &str = r#"
    if (rho > 0.0) {
        u = u / rho;
    }

    let ux = u.x;
    let uy = u.y;
    let ux2 = ux * ux;
    let uy2 = uy * uy;
    let u2 = ux2 + uy2;

    let f0 = f_local[0]; let f1 = f_local[1]; let f2 = f_local[2];
    let f3 = f_local[3]; let f4 = f_local[4]; let f5 = f_local[5];
    let f6 = f_local[6]; let f7 = f_local[7]; let f8 = f_local[8];

    // 1. Compute Moments (m = M * f)
    var m: array<f32, 9>;
    m[0] = rho;
    m[1] = -4.0*f0 - (f1+f2+f3+f4) + 2.0*(f5+f6+f7+f8);
    m[2] = 4.0*f0 - 2.0*(f1+f2+f3+f4) + (f5+f6+f7+f8);
    m[3] = f1 - f3 + f5 - f6 - f7 + f8;
    m[4] = -2.0*(f1 - f3) + f5 - f6 - f7 + f8;
    m[5] = f2 - f4 + f5 + f6 - f7 - f8;
    m[6] = -2.0*(f2 - f4) + f5 + f6 - f7 - f8;
    m[7] = f1 - f2 + f3 - f4;
    m[8] = f5 - f6 + f7 - f8;

    // 2. Compute Equilibrium Moments (meq)
    var meq: array<f32, 9>;
    meq[0] = rho;
    meq[1] = rho * (-2.0 + 3.0 * u2);
    meq[2] = rho * (1.0 - 3.0 * u2);
    meq[3] = rho * ux;
    meq[4] = -rho * ux;
    meq[5] = rho * uy;
    meq[6] = -rho * uy;
    meq[7] = rho * (ux2 - uy2);
    meq[8] = rho * ux * uy;

    let m7_neq = m[7] - meq[7];
    let m8_neq = m[8] - meq[8];
    
    let OMEGA_EFF = OMEGA;

    // 3. Relax Moments
    m[1] = m[1] - 1.63 * (m[1] - meq[1]); // e
    m[2] = m[2] - 1.14 * (m[2] - meq[2]); // epsilon 
    m[4] = m[4] - 1.92 * (m[4] - meq[4]); // q_x 
    m[6] = m[6] - 1.92 * (m[6] - meq[6]); // q_y 
    m[7] = m[7] - OMEGA_EFF * m7_neq;     // p_xx
    m[8] = m[8] - OMEGA_EFF * m8_neq;     // p_xy

    // 4. Inverse Transformation -> Write to f_local
    let m0 = m[0]; let m1 = m[1]; let m2 = m[2]; let m3 = m[3]; 
    let m4 = m[4]; let m5 = m[5]; let m6 = m[6]; let m7 = m[7]; let m8 = m[8];

    f_local[0] = (1.0/9.0)  * (m0 - m1 + m2);
    f_local[1] = (1.0/36.0) * (4.0*m0 - m1 - 2.0*m2 + 6.0*m3 - 6.0*m4 + 9.0*m7);
    f_local[2] = (1.0/36.0) * (4.0*m0 - m1 - 2.0*m2 + 6.0*m5 - 6.0*m6 - 9.0*m7);
    f_local[3] = (1.0/36.0) * (4.0*m0 - m1 - 2.0*m2 - 6.0*m3 + 6.0*m4 + 9.0*m7);
    f_local[4] = (1.0/36.0) * (4.0*m0 - m1 - 2.0*m2 - 6.0*m5 + 6.0*m6 - 9.0*m7);
    f_local[5] = (1.0/36.0) * (4.0*m0 + 2.0*m1 + m2 + 6.0*m3 + 3.0*m4 + 6.0*m5 + 3.0*m6 + 9.0*m8);
    f_local[6] = (1.0/36.0) * (4.0*m0 + 2.0*m1 + m2 - 6.0*m3 - 3.0*m4 + 6.0*m5 + 3.0*m6 - 9.0*m8);
    f_local[7] = (1.0/36.0) * (4.0*m0 + 2.0*m1 + m2 - 6.0*m3 - 3.0*m4 - 6.0*m5 - 3.0*m6 + 9.0*m8);
    f_local[8] = (1.0/36.0) * (4.0*m0 + 2.0*m1 + m2 + 6.0*m3 + 3.0*m4 - 6.0*m5 - 3.0*m6 - 9.0*m8);
"#;


// ═══════════════════════════════════════════════════════════════════════════
//  BASE TEMPLATES (EVEN, ODD, INIT, EXTRACT)
// ═══════════════════════════════════════════════════════════════════════════

pub(super) const BASE_STEP_EVEN_2D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(102) override OMEGA: f32 = 1.0;
@id(103) override wgs_x: u32 = 8;
@id(104) override wgs_y: u32 = 8;

override TOTAL_CELLS = NX * NY;

//{Q}
const FLAG_TYPE_SHIFT: u32 = 24u;
const FLAG_ID_MASK: u32 = 0x00FFFFFFu;
//{EX}
//{EY}
//{WEIGHTS}
//{OPP}

struct BoundaryConfig {
    vel: vec2<f32>,
    density: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> fa: array<f32>;
@group(0) @binding(1) var<storage, read_write> fb: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> boundary_configs: array<BoundaryConfig>;

@compute @workgroup_size(wgs_x, wgs_y)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    if (x >= NX || y >= NY) { return; }

    let cell_idx = x + (y * NX);
    let cell_pos = vec2<i32>(i32(x), i32(y));
    var f_local: array<f32, Q>;
    var rho: f32 = 0.0;
    var u = vec2<f32>(0.0, 0.0);

    // --- STREAMING (PULL) ---
    for (var i: u32 = 0u; i < Q; i += 1u) {
        let ex = EX[i];
        let ey = EY[i];
        let neighbour_pos = cell_pos - vec2<i32>(ex, ey);
        var pulled_f: f32 = 0.0;

        if (neighbour_pos.x < 0 || neighbour_pos.y < 0 || neighbour_pos.x >= i32(NX) || neighbour_pos.y >= i32(NY)) {
            // Out of bounds domain limits
            //{SOLID_BOUNDARY_LOGIC_EVEN}
        } else {
            let neighbour_idx = u32(neighbour_pos.x) + (u32(neighbour_pos.y) * NX);
            let raw_flag = flags[neighbour_idx];
            let boundary_type = raw_flag >> FLAG_TYPE_SHIFT;
            let cfg_id = raw_flag & FLAG_ID_MASK;
            let cfg = boundary_configs[cfg_id];

            switch boundary_type {
                case 0u: {
                    //{FLUID_BOUNDARY_LOGIC_EVEN}
                }
                case 1u: {
                    //{SOLID_BOUNDARY_LOGIC_EVEN}
                }
                case 2u: {
                    //{INLET_BOUNDARY_LOGIC_EVEN}
                }
                case 3u: {
                    //{OUTLET_BOUNDARY_LOGIC_EVEN}
                }
                default: {
                    pulled_f = 0.0;
                }
            }
        }

        f_local[i] = pulled_f;
        rho += pulled_f;
        u += vec2<f32>(f32(ex), f32(ey)) * pulled_f;
    }

    // --- COLLISION ---
    //{COLLISION_LOGIC}

    // --- WRITE (INVERTED) ---
    for (var i: u32 = 0u; i < Q; i += 1u) {
        fb[cell_idx + OPP[i] * TOTAL_CELLS] = f_local[i];
    }
}
"#;


pub(super) const BASE_STEP_ODD_2D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(102) override OMEGA: f32 = 1.0;
@id(103) override wgs_x: u32 = 8;
@id(104) override wgs_y: u32 = 8;

override TOTAL_CELLS = NX * NY;

//{Q}
const FLAG_TYPE_SHIFT: u32 = 24u;
const FLAG_ID_MASK: u32 = 0x00FFFFFFu;
//{EX}
//{EY}
//{WEIGHTS}
//{OPP}

struct BoundaryConfig {
    vel: vec2<f32>,
    density: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> fa: array<f32>;
@group(0) @binding(1) var<storage, read_write> fb: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> boundary_configs: array<BoundaryConfig>;

@compute @workgroup_size(wgs_x, wgs_y)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    if (x >= NX || y >= NY) { return; }

    let cell_idx = x + (y * NX);
    let cell_pos = vec2<i32>(i32(x), i32(y));
    var f_local: array<f32, Q>;
    var rho: f32 = 0.0;
    var u = vec2<f32>(0.0, 0.0);

    // --- STREAMING (PULL) ---
    for (var i: u32 = 0u; i < Q; i += 1u) {
        let ex = EX[i];
        let ey = EY[i];
        let neighbour_pos = cell_pos - vec2<i32>(ex, ey);
        var pulled_f: f32 = 0.0;

        if (neighbour_pos.x < 0 || neighbour_pos.y < 0 || neighbour_pos.x >= i32(NX) || neighbour_pos.y >= i32(NY)) {
            // Out of bounds domain limits
            //{SOLID_BOUNDARY_LOGIC_ODD}
        } else {
            let neighbour_idx = u32(neighbour_pos.x) + (u32(neighbour_pos.y) * NX);
            let raw_flag = flags[neighbour_idx];
            let boundary_type = raw_flag >> FLAG_TYPE_SHIFT;
            let cfg_id = raw_flag & FLAG_ID_MASK;
            let cfg = boundary_configs[cfg_id];

            switch boundary_type {
                case 0u: {
                    //{FLUID_BOUNDARY_LOGIC_ODD}
                }
                case 1u: {
                    //{SOLID_BOUNDARY_LOGIC_ODD}
                }
                case 2u: {
                    //{INLET_BOUNDARY_LOGIC_ODD}
                }
                case 3u: {
                    //{OUTLET_BOUNDARY_LOGIC_ODD}
                }
                default: {
                    pulled_f = 0.0;
                }
            }
        }

        f_local[i] = pulled_f;
        rho += pulled_f;
        u += vec2<f32>(f32(ex), f32(ey)) * pulled_f;
    }

    // --- COLLISION ---
    //{COLLISION_LOGIC}

    // --- WRITE (NON-INVERTED) ---
    for (var i: u32 = 0u; i < Q; i += 1u) {
        fb[cell_idx + i * TOTAL_CELLS] = f_local[i];
    }
}
"#;

pub(super) const BASE_INIT_2D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;

@id(102) override RHO_INIT: f32 = 1.0;
@id(103) override U_X_INIT: f32 = 0.0;
@id(104) override U_Y_INIT: f32 = 0.0;

@id(105) override wgs_x: u32 = 8;
@id(106) override wgs_y: u32 = 8;

override TOTAL_CELLS = NX * NY;
//{Q}

const FLAG_TYPE_SHIFT: u32 = 24u;

//{EX}
//{EY}

//{WEIGHTS}

@group(0) @binding(0) var<storage, read_write> fa: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;

@compute @workgroup_size(wgs_x, wgs_y)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

    let x = id.x;
    let y = id.y;

    if ( x >= NX || y >= NY ) {
        return;
    }

    let cell_idx = x + ( y * NX );
    let raw_flag = flags[cell_idx];
    let boundary_type = raw_flag >> FLAG_TYPE_SHIFT;

    var rho = RHO_INIT;
    var u = vec2<f32>(U_X_INIT, U_Y_INIT);

    if ( boundary_type == 1u ) {
        u = vec2<f32>(0.0, 0.0);
    }

    let u_sq = dot( u, u );
    let u_sq_term = 1.5 * u_sq;
    for ( var i=0u; i < Q; i += 1u ) {
        let eu = ( f32(EX[i]) * u.x ) + ( f32(EY[i]) * u.y );
        let feq = WEIGHTS[i] * rho * ( 1.0 + 3.0 * eu + 4.5 * ( eu * eu ) - u_sq_term );
        fa[cell_idx + i * TOTAL_CELLS] = feq;
    }

}
"#;

pub(super) const BASE_EXTRACT_2D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(105) override wgs_x: u32 = 8;
@id(106) override wgs_y: u32 = 8;

override TOTAL_CELLS = NX * NY;

//{Q}
//{EX}
//{EY}

@group(0) @binding(0) var<storage, read> fa: array<f32>;
@group(0) @binding(1) var<storage, read_write> macro_data: array<vec4<f32>>;

@compute @workgroup_size(wgs_x, wgs_y)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

    let x = id.x;
    let y = id.y;

    if ( x >= NX || y >= NY ) {
        return;
    }

    let cell_idx = x + ( y * NX );
    var rho = 0.0;
    var u = vec2<f32>(0.0, 0.0);

    for (var i: u32 = 0u; i < Q; i += 1u) {

        let pop = fa[i * TOTAL_CELLS + cell_idx ];
        rho += pop;
        u += vec2<f32>(f32(EX[i]), f32(EY[i])) * pop;

    } 

    if ( rho > 0.0 ) {
        u = u / rho;
    }

    macro_data[cell_idx] = vec4<f32>(u.x, u.y, 0.0, rho);

}
"#;