//! Base D3 WGSL programs and in-place alternating-access templates.

/// Legacy initialization template retained for compiler variants.
pub const BASE_INIT_3D: &str = r#"
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

@group(0) @binding(0) var<storage, read_write> fa: array<f32>; // POP_STORAGE
@group(0) @binding(2) var<storage, read> flags: array<u32>;

//{PRECISION_HELPERS}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    if (x >= NX || y >= NY || z >= NZ) { return; }

    let cell_idx = x + y * NX + z * NX * NY;
    let flag = flags[cell_idx];
    let boundary_type = flag >> 24u;

    var rho = 1.0;
    var u = vec3<f32>(0.0, 0.0, 0.0);

    //{INIT_CUSTOM_LOGIC}

    for (var i = 0u; i < Q; i = i + 1u) {
        let ex_i = f32(EX[i]);
        let ey_i = f32(EY[i]);
        let ez_i = f32(EZ[i]);
        let w_i = WEIGHTS[i];

        let cu = ex_i * u.x + ey_i * u.y + ez_i * u.z;
        let u2 = u.x * u.x + u.y * u.y + u.z * u.z;
        let feq = w_i * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);

        let idx = cell_idx + i * TOTAL_CELLS;
        store_fa(idx, feq);
    }
}
"#;

/// Legacy macroscopic extraction template.
pub const BASE_EXTRACT_3D: &str = r#"
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
@id(126) override INVERTED: u32 = 0u;

override TOTAL_CELLS = NX * NY * NZ;

//{Q}
//{EX}
//{EY}
//{EZ}
//{WEIGHTS}
//{OPP}

@group(0) @binding(0) var<storage, read> fa: array<f32>; // POP_FA
@group(0) @binding(1) var<storage, read_write> macro_data: array<vec4<f32>>;

//{PRECISION_HELPERS}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    if (x >= NX || y >= NY || z >= NZ) { return; }

    let cell_idx = x + y * NX + z * NX * NY;

    var rho = 0.0;
    var u = vec3<f32>(0.0, 0.0, 0.0);

    for (var i = 0u; i < Q; i = i + 1u) {
        let direction = select(i, OPP[i], INVERTED != 0u);
        let idx = cell_idx + direction * TOTAL_CELLS;
        let f_val = load_fa(idx);
        rho += f_val;
        u.x += f32(EX[i]) * f_val;
        u.y += f32(EY[i]) * f_val;
        u.z += f32(EZ[i]) * f_val;
    }

    if (rho > 0.0) {
        u = (u + 0.5 * vec3<f32>(FORCE_X, FORCE_Y, FORCE_Z)) / rho;
    }

    macro_data[cell_idx] = vec4<f32>(u.x, u.y, u.z, rho);
}
"#;

/// Legacy even-phase step template.
pub const BASE_STEP_EVEN_3D: &str = r#"
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
@id(124) override PERIODIC_Y: u32 = 0u;
@id(125) override PERIODIC_Z: u32 = 0u;
@id(127) override SPONGE_LEN: u32 = 0u;
@id(128) override SPONGE_STRENGTH: f32 = 0.0;
@id(129) override SPONGE_CFG: u32 = 0u;

override TOTAL_CELLS = NX * NY * NZ;

const FLAG_TYPE_SHIFT: u32 = 24u;
const FLAG_ID_MASK: u32 = 0x00FFFFFFu;
const INTERIOR_OPT_BIT: u32 = 1u << 23u;

//{Q}
//{EX}
//{EY}
//{EZ}
//{WEIGHTS}
//{OPP}

struct BoundaryConfig {
    vel: vec3<f32>,
    density: f32,
}

@group(0) @binding(0) var<storage, read> fa: array<f32>; // POP_FA
@group(0) @binding(1) var<storage, read_write> fb: array<f32>; // POP_FB
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> boundary_configs: array<BoundaryConfig>;

//{PRECISION_HELPERS}
//{REFLECT_X}
//{REFLECT_Y}
//{REFLECT_Z}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    if (x >= NX || y >= NY || z >= NZ) { return; }
    let cell_idx = x + y * NX + z * NX * NY;
    let flag = flags[cell_idx];
    let my_flag = flag;
    let boundary_type = flag >> FLAG_TYPE_SHIFT;
    let cfg_id = flag & FLAG_ID_MASK;

    var f_local: array<f32, 19>;
    var rho = 0.0;
    var u = vec3<f32>(0.0, 0.0, 0.0);
    
    if (((my_flag >> 23u) & 1u) == 1u) {
        //{UNROLLED_FAST_PATH_EVEN}
    } else {
        for (var i = 0u; i < Q; i = i + 1u) {
            let ex = EX[i];
            let ey = EY[i];
            let ez = EZ[i];
            let dx = -ex;
            let dy = -ey;
            let dz = -ez;
            let nx = i32(x) + dx;
            let ny = i32(y) + dy;
            let nz = i32(z) + dz;

            var neighbour_idx = 0u;
            let px = select(clamp(nx, 0, i32(NX) - 1), (nx + i32(NX)) % i32(NX), PERIODIC_X != 0u);
            let py = select(clamp(ny, 0, i32(NY) - 1), (ny + i32(NY)) % i32(NY), PERIODIC_Y != 0u);
            let pz = select(clamp(nz, 0, i32(NZ) - 1), (nz + i32(NZ)) % i32(NZ), PERIODIC_Z != 0u);
            neighbour_idx = u32(px + py * i32(NX) + pz * i32(NX * NY));

            var pulled_f = 0.0;
            let neighbour_type = flags[neighbour_idx] >> FLAG_TYPE_SHIFT;
            if (boundary_type == 0u && neighbour_type == 1u) {
                // Half-way no-slip bounce-back is performed at the fluid link.
                // On the even phase the input buffer is in normal layout.
                pulled_f = load_fa(cell_idx + OPP[i] * TOTAL_CELLS);
            } else {
                switch boundary_type {
                    //{BOUNDARY_SWITCH_CASES_EVEN}
                    default: {
                        //{FLUID_PULL_LOGIC_EVEN}
                    }
                }
            }

            f_local[i] = pulled_f;
            rho += pulled_f;
            u += vec3<f32>(f32(EX[i]), f32(EY[i]), f32(EZ[i])) * pulled_f;
        }
    }

    //{POST_STREAMING_CORRECTION}

    if (boundary_type == 0u) {
        //{COLLISION_LOGIC}
        //{SPONGE_LOGIC}
    }

    //{UNROLLED_WRITE_EVEN}
}
"#;

/// Legacy odd-phase step template.
pub const BASE_STEP_ODD_3D: &str = r#"
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
@id(124) override PERIODIC_Y: u32 = 0u;
@id(125) override PERIODIC_Z: u32 = 0u;
@id(127) override SPONGE_LEN: u32 = 0u;
@id(128) override SPONGE_STRENGTH: f32 = 0.0;
@id(129) override SPONGE_CFG: u32 = 0u;

override TOTAL_CELLS = NX * NY * NZ;

const FLAG_TYPE_SHIFT: u32 = 24u;
const FLAG_ID_MASK: u32 = 0x00FFFFFFu;

//{Q}
//{EX}
//{EY}
//{EZ}
//{WEIGHTS}
//{OPP}

struct BoundaryConfig {
    vel: vec3<f32>,
    density: f32,
}

@group(0) @binding(0) var<storage, read> fa: array<f32>; // POP_FA
@group(0) @binding(1) var<storage, read_write> fb: array<f32>; // POP_FB
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> boundary_configs: array<BoundaryConfig>;

//{PRECISION_HELPERS}
//{REFLECT_X}
//{REFLECT_Y}
//{REFLECT_Z}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    if (x >= NX || y >= NY || z >= NZ) { return; }
    let cell_idx = x + y * NX + z * NX * NY;
    let flag = flags[cell_idx];
    let my_flag = flag;
    let boundary_type = flag >> FLAG_TYPE_SHIFT;
    let cfg_id = flag & FLAG_ID_MASK;

    var f_local: array<f32, 19>;
    var rho = 0.0;
    var u = vec3<f32>(0.0, 0.0, 0.0);
    
    if (((my_flag >> 23u) & 1u) == 1u) {
        //{UNROLLED_FAST_PATH_ODD}
    } else {
        for (var i = 0u; i < Q; i = i + 1u) {
            let ex = EX[i];
            let ey = EY[i];
            let ez = EZ[i];
            let dx = -ex;
            let dy = -ey;
            let dz = -ez;
            let nx = i32(x) + dx;
            let ny = i32(y) + dy;
            let nz = i32(z) + dz;

            var neighbour_idx = 0u;
            let px = select(clamp(nx, 0, i32(NX) - 1), (nx + i32(NX)) % i32(NX), PERIODIC_X != 0u);
            let py = select(clamp(ny, 0, i32(NY) - 1), (ny + i32(NY)) % i32(NY), PERIODIC_Y != 0u);
            let pz = select(clamp(nz, 0, i32(NZ) - 1), (nz + i32(NZ)) % i32(NZ), PERIODIC_Z != 0u);
            neighbour_idx = u32(px + py * i32(NX) + pz * i32(NX * NY));

            var pulled_f = 0.0;
            let neighbour_type = flags[neighbour_idx] >> FLAG_TYPE_SHIFT;
            if (boundary_type == 0u && neighbour_type == 1u) {
                // The odd-phase input buffer is direction-inverted, so the
                // physical opposite population occupies slot i.
                pulled_f = load_fa(cell_idx + i * TOTAL_CELLS);
            } else {
                switch boundary_type {
                    //{BOUNDARY_SWITCH_CASES_ODD}
                    default: {
                        //{FLUID_PULL_LOGIC_ODD}
                    }
                }
            }

            f_local[i] = pulled_f;
            rho += pulled_f;
            u += vec3<f32>(f32(EX[i]), f32(EY[i]), f32(EZ[i])) * pulled_f;
        }
    }

    //{POST_STREAMING_CORRECTION}

    if (boundary_type == 0u) {
        //{COLLISION_LOGIC}
        //{SPONGE_LOGIC}
    }

    //{UNROLLED_WRITE_ODD}
}
"#;

/// Boundary-free even-phase fast-path template.
pub const PURE_FLUID_STEP_EVEN_3D: &str = r#"
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
@id(124) override PERIODIC_Y: u32 = 0u;
@id(125) override PERIODIC_Z: u32 = 0u;

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
//{REFLECT_X}
//{REFLECT_Y}
//{REFLECT_Z}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    if (x >= NX || y >= NY || z >= NZ) { return; }
    let cell_idx = x + y * NX + z * NX * NY;

    var f_local: array<f32, 19>;
    var rho = 0.0;
    var u = vec3<f32>(0.0, 0.0, 0.0);
    
    for (var i = 0u; i < Q; i = i + 1u) {
        let nx = i32(x) - EX[i];
        let ny = i32(y) - EY[i];
        let nz = i32(z) - EZ[i];
        let px = select(clamp(nx, 0, i32(NX)-1), (nx+i32(NX))%i32(NX), PERIODIC_X != 0u);
        let py = select(clamp(ny, 0, i32(NY)-1), (ny+i32(NY))%i32(NY), PERIODIC_Y != 0u);
        let pz = select(clamp(nz, 0, i32(NZ)-1), (nz+i32(NZ))%i32(NZ), PERIODIC_Z != 0u);
        let neighbour_idx = u32(px + py*i32(NX) + pz*i32(NX*NY));
        let pulled_f = load_fa(neighbour_idx + i*TOTAL_CELLS);
        f_local[i] = pulled_f;
        rho += pulled_f;
        u += vec3<f32>(f32(EX[i]), f32(EY[i]), f32(EZ[i])) * pulled_f;
    }

    //{COLLISION_LOGIC}

    //{UNROLLED_WRITE_EVEN}
}
"#;

/// Boundary-free odd-phase fast-path template.
pub const PURE_FLUID_STEP_ODD_3D: &str = r#"
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
@id(124) override PERIODIC_Y: u32 = 0u;
@id(125) override PERIODIC_Z: u32 = 0u;

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
//{REFLECT_X}
//{REFLECT_Y}
//{REFLECT_Z}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    if (x >= NX || y >= NY || z >= NZ) { return; }
    let cell_idx = x + y * NX + z * NX * NY;

    var f_local: array<f32, 19>;
    var rho = 0.0;
    var u = vec3<f32>(0.0, 0.0, 0.0);
    
    for (var i = 0u; i < Q; i = i + 1u) {
        let nx = i32(x) - EX[i];
        let ny = i32(y) - EY[i];
        let nz = i32(z) - EZ[i];
        let px = select(clamp(nx, 0, i32(NX)-1), (nx+i32(NX))%i32(NX), PERIODIC_X != 0u);
        let py = select(clamp(ny, 0, i32(NY)-1), (ny+i32(NY))%i32(NY), PERIODIC_Y != 0u);
        let pz = select(clamp(nz, 0, i32(NZ)-1), (nz+i32(NZ))%i32(NZ), PERIODIC_Z != 0u);
        let neighbour_idx = u32(px + py*i32(NX) + pz*i32(NX*NY));
        let pulled_f = load_fa(neighbour_idx + OPP[i]*TOTAL_CELLS);
        f_local[i] = pulled_f;
        rho += pulled_f;
        u += vec3<f32>(f32(EX[i]), f32(EY[i]), f32(EZ[i])) * pulled_f;
    }

    //{COLLISION_LOGIC}

    //{UNROLLED_WRITE_ODD}
}
"#;

/// Single-buffer Esoteric-Pull initialization. The initial equilibria are
/// stored in the phase-1 layout so phase 0 can perform the first streamed load.
/// Initializes the in-place D3Q19 population buffer.
pub const IN_PLACE_INIT_3D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(110) override NZ: u32 = 1080;
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

@group(0) @binding(0) var<storage, read_write> fa: array<f32>; // POP_STORAGE
@group(0) @binding(2) var<storage, read> flags: array<u32>;

//{PRECISION_HELPERS}

fn wrapped_cell(ix: i32, iy: i32, iz: i32) -> u32 {
    let wx = (ix + i32(NX)) % i32(NX);
    let wy = (iy + i32(NY)) % i32(NY);
    let wz = (iz + i32(NZ)) % i32(NZ);
    return u32(wx + wy * i32(NX) + wz * i32(NX * NY));
}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    if (x >= NX || y >= NY || z >= NZ) { return; }

    let cell_idx = x + y * NX + z * NX * NY;
    var rho = 1.0;
    var u = vec3<f32>(0.0, 0.0, 0.0);
    //{INIT_CUSTOM_LOGIC}

    var f_local: array<f32, 19>;
    let u2 = dot(u, u);
    for (var i = 0u; i < Q; i += 1u) {
        let cu = f32(EX[i]) * u.x + f32(EY[i]) * u.y + f32(EZ[i]) * u.z;
        f_local[i] = WEIGHTS[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }

    // Esoteric-Pull phase-1 store. Every address has exactly one writer.
    store_fa(cell_idx, f_local[0]);
    for (var pair = 1u; pair < Q; pair += 1u) {
        let other = OPP[pair];
        if (pair >= other) { continue; }
        let plus = wrapped_cell(
            i32(x) + EX[pair],
            i32(y) + EY[pair],
            i32(z) + EZ[pair]
        );
        store_fa(plus + other * TOTAL_CELLS, f_local[pair]);
        store_fa(cell_idx + pair * TOTAL_CELLS, f_local[other]);
    }
}
"#;

/// One phase of race-free, single-buffer Esoteric-Pull streaming/collision.
/// `PHASE` is compiled as either zero or one; storage addresses form a
/// permutation, so no invocation reads an address written by another.
/// Advances in-place D3Q19 populations and prepares outlet state.
pub const IN_PLACE_STEP_3D: &str = r#"
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
@id(127) override SPONGE_LEN: u32 = 0u;
@id(128) override SPONGE_STRENGTH: f32 = 0.0;
@id(129) override SPONGE_CFG: u32 = 0u;
override PHASE: u32 = 0u;

override TOTAL_CELLS = NX * NY * NZ;
const FLAG_TYPE_SHIFT: u32 = 24u;
const FLAG_ID_MASK: u32 = 0x00FFFFFFu;

//{Q}
//{EX}
//{EY}
//{EZ}
//{WEIGHTS}
//{OPP}

struct BoundaryConfig {
    vel: vec3<f32>,
    density: f32,
}

@group(0) @binding(0) var<storage, read_write> fa: array<f32>; // POP_FA
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> boundary_configs: array<BoundaryConfig>;

//{PRECISION_HELPERS}
//{REFLECT_X}
//{REFLECT_Y}
//{REFLECT_Z}

fn wrapped_cell(ix: i32, iy: i32, iz: i32) -> u32 {
    let wx = (ix + i32(NX)) % i32(NX);
    let wy = (iy + i32(NY)) % i32(NY);
    let wz = (iz + i32(NZ)) % i32(NZ);
    return u32(wx + wy * i32(NX) + wz * i32(NX * NY));
}

fn load_streamed_at(ix: i32, iy: i32, iz: i32, direction: u32) -> f32 {
    let cell = wrapped_cell(ix, iy, iz);
    if (direction == 0u) { return load_fa(cell); }
    let pair = min(direction, OPP[direction]);
    let other = OPP[pair];
    let plus = wrapped_cell(ix + EX[pair], iy + EY[pair], iz + EZ[pair]);
    if (PHASE == 0u) {
        return select(
            load_fa(plus + pair * TOTAL_CELLS),
            load_fa(cell + other * TOTAL_CELLS),
            direction == pair
        );
    }
    return select(
        load_fa(plus + other * TOTAL_CELLS),
        load_fa(cell + pair * TOTAL_CELLS),
        direction == pair
    );
}

fn store_streamed_at(ix: i32, iy: i32, iz: i32, direction: u32, value: f32) {
    let cell = wrapped_cell(ix, iy, iz);
    if (direction == 0u) {
        store_fb(cell, value);
        return;
    }
    let pair = min(direction, OPP[direction]);
    let other = OPP[pair];
    let plus = wrapped_cell(ix + EX[pair], iy + EY[pair], iz + EZ[pair]);
    if (PHASE == 0u) {
        let address = select(plus + pair * TOTAL_CELLS, cell + other * TOTAL_CELLS,
            direction == pair);
        store_fb(address, value);
    } else {
        let address = select(plus + other * TOTAL_CELLS, cell + pair * TOTAL_CELLS,
            direction == pair);
        store_fb(address, value);
    }
}

fn store_collided(x: u32, y: u32, z: u32, f: ptr<function, array<f32, 19>>) {
    let cell = x + y * NX + z * NX * NY;
    store_fb(cell, (*f)[0]);
    for (var pair = 1u; pair < Q; pair += 1u) {
        let other = OPP[pair];
        if (pair >= other) { continue; }
        let plus = wrapped_cell(
            i32(x) + EX[pair],
            i32(y) + EY[pair],
            i32(z) + EZ[pair]
        );
        if (PHASE == 0u) {
            store_fb(plus + pair * TOTAL_CELLS, (*f)[pair]);
            store_fb(cell + other * TOTAL_CELLS, (*f)[other]);
        } else {
            store_fb(plus + other * TOTAL_CELLS, (*f)[pair]);
            store_fb(cell + pair * TOTAL_CELLS, (*f)[other]);
        }
    }
}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    if (x >= NX || y >= NY || z >= NZ) { return; }
    let cell_idx = x + y * NX + z * NX * NY;
    let flag = flags[cell_idx];
    let my_flag = flag;
    let boundary_type = flag >> FLAG_TYPE_SHIFT;
    let cfg_id = flag & FLAG_ID_MASK;

    // A skipped solid is part of the Esoteric-Pull bounce-back exchange: its
    // interface slots are written and subsequently read by the adjacent fluid.
    if (boundary_type == 1u) { return; }

    // Most cells in large periodic/obstacle domains are marked as having only
    // fluid neighbors. Their AA addresses are statically known, allowing the
    // compiler to issue straight-line loads/stores with no direction loop,
    // modulo, OPP lookup, or boundary switch in the hot path.
    let sponge_cells = min(SPONGE_LEN, NX);
    let in_sponge = sponge_cells > 0u && x >= NX - sponge_cells;
    let has_direct_neighbors = x > 0u && x + 1u < NX
        && y > 0u && y + 1u < NY && z > 0u && z + 1u < NZ;
    if (boundary_type == 0u && has_direct_neighbors && !in_sponge) {
        var f_local: array<f32, 19>;
        var rho = 0.0;
        var u = vec3<f32>(0.0);
        //{AA_INTERIOR_LOADS}
        //{COLLISION_LOGIC}
        //{AA_INTERIOR_STORES}
        return;
    }

    var f_local: array<f32, 19>;
    var rho = 0.0;
    var u = vec3<f32>(0.0);
    for (var i = 0u; i < Q; i += 1u) {
        let ex = EX[i];
        let ey = EY[i];
        let ez = EZ[i];
        let neighbour_idx = wrapped_cell(i32(x) - ex, i32(y) - ey, i32(z) - ez);
        var pulled_f = load_streamed_at(i32(x), i32(y), i32(z), i);
        switch boundary_type {
            //{BOUNDARY_SWITCH_CASES}
            default: {}
        }
        f_local[i] = pulled_f;
        rho += pulled_f;
        u += vec3<f32>(f32(ex), f32(ey), f32(ez)) * pulled_f;
    }

    //{POST_STREAMING_CORRECTION}
    if (boundary_type == 0u) {
        //{COLLISION_LOGIC}
        //{SPONGE_LOGIC}
    }
    store_collided(x, y, z, &f_local);
}

// This is a separate dispatch from `main`, providing a device-wide ordering
// point before collide/stream. Each outlet invocation writes only its own AA
// address set and reads only the corresponding last-interior set.
@compute @workgroup_size(8, 8, 1)
fn prepare_outlet(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let y = global_id.x;
    let z = global_id.y;
    if (y >= NY || z >= NZ || NX < 2u) { return; }
    let x = NX - 1u;
    let cell = x + y * NX + z * NX * NY;
    if ((flags[cell] >> FLAG_TYPE_SHIFT) != 3u) { return; }

    for (var i = 0u; i < Q; i += 1u) {
        if (EX[i] < 0) {
            let extrapolated = load_streamed_at(i32(x) - 1, i32(y), i32(z), i);
            store_streamed_at(i32(x), i32(y), i32(z), i, extrapolated);
        }
    }
}
"#;

/// Extracts macroscopic fields from either in-place AA phase.
pub const IN_PLACE_EXTRACT_3D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(110) override NZ: u32 = 1080;
@id(103) override wgs_x: u32 = 8;
@id(104) override wgs_y: u32 = 8;
@id(111) override wgs_z: u32 = 1;
@id(120) override FORCE_X: f32 = 0.0;
@id(121) override FORCE_Y: f32 = 0.0;
@id(122) override FORCE_Z: f32 = 0.0;
override PHASE: u32 = 0u;

override TOTAL_CELLS = NX * NY * NZ;
//{Q}
//{EX}
//{EY}
//{EZ}
//{WEIGHTS}
//{OPP}

@group(0) @binding(0) var<storage, read> fa: array<f32>; // POP_FA
@group(0) @binding(1) var<storage, read_write> macro_data: array<vec4<f32>>;
//{PRECISION_HELPERS}

fn wrapped_cell(ix: i32, iy: i32, iz: i32) -> u32 {
    let wx = (ix + i32(NX)) % i32(NX);
    let wy = (iy + i32(NY)) % i32(NY);
    let wz = (iz + i32(NZ)) % i32(NZ);
    return u32(wx + wy * i32(NX) + wz * i32(NX * NY));
}

fn load_streamed_at(ix: i32, iy: i32, iz: i32, direction: u32) -> f32 {
    let cell = wrapped_cell(ix, iy, iz);
    if (direction == 0u) { return load_fa(cell); }
    let pair = min(direction, OPP[direction]);
    let other = OPP[pair];
    let plus = wrapped_cell(ix + EX[pair], iy + EY[pair], iz + EZ[pair]);
    if (PHASE == 0u) {
        return select(
            load_fa(plus + pair * TOTAL_CELLS),
            load_fa(cell + other * TOTAL_CELLS),
            direction == pair
        );
    }
    return select(
        load_fa(plus + other * TOTAL_CELLS),
        load_fa(cell + pair * TOTAL_CELLS),
        direction == pair
    );
}

@compute @workgroup_size(wgs_x, wgs_y, wgs_z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    if (x >= NX || y >= NY || z >= NZ) { return; }
    let cell_idx = x + y * NX + z * NX * NY;
    var rho = 0.0;
    var u = vec3<f32>(0.0);
    for (var i = 0u; i < Q; i += 1u) {
        let f = load_streamed_at(i32(x), i32(y), i32(z), i);
        rho += f;
        u += vec3<f32>(f32(EX[i]), f32(EY[i]), f32(EZ[i])) * f;
    }
    if (rho > 0.0) {
        u = (u + 0.5 * vec3<f32>(FORCE_X, FORCE_Y, FORCE_Z)) / rho;
    }
    macro_data[cell_idx] = vec4<f32>(u, rho);
}
"#;
