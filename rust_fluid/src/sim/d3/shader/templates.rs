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

pub const BASE_EXTRACT_3D: &str = r#"
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
        let idx = cell_idx + i * TOTAL_CELLS;
        let f_val = load_fa(idx);
        rho += f_val;
        u.x += f32(EX[i]) * f_val;
        u.y += f32(EY[i]) * f_val;
        u.z += f32(EZ[i]) * f_val;
    }

    if (rho > 0.0) {
        u /= rho;
    }

    macro_data[cell_idx] = vec4<f32>(u.x, u.y, u.z, rho);
}
"#;

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
        if PERIODIC_X != 0u {
            let px = (nx + i32(NX)) % i32(NX);
            let py = (ny + i32(NY)) % i32(NY);
            let pz = (nz + i32(NZ)) % i32(NZ);
            neighbour_idx = u32(px + py * i32(NX) + pz * i32(NX * NY));
        } else {
            let px = clamp(nx, 0, i32(NX) - 1);
            let py = clamp(ny, 0, i32(NY) - 1);
            let pz = clamp(nz, 0, i32(NZ) - 1);
            neighbour_idx = u32(px + py * i32(NX) + pz * i32(NX * NY));
        }

        var pulled_f = 0.0;
        switch boundary_type {
            //{BOUNDARY_SWITCH_CASES_EVEN}
            default: {
                //{FLUID_PULL_LOGIC_EVEN}
            }
        }
        
        f_local[i] = pulled_f;
        rho += pulled_f;
        u += vec3<f32>(f32(EX[i]), f32(EY[i]), f32(EZ[i])) * pulled_f;
    }

    //{POST_STREAMING_CORRECTION}

    if (boundary_type == 0u) {
        //{COLLISION_LOGIC}
    }

    //{UNROLLED_WRITE_EVEN}
}
"#;

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
        if PERIODIC_X != 0u {
            let px = (nx + i32(NX)) % i32(NX);
            let py = (ny + i32(NY)) % i32(NY);
            let pz = (nz + i32(NZ)) % i32(NZ);
            neighbour_idx = u32(px + py * i32(NX) + pz * i32(NX * NY));
        } else {
            let px = clamp(nx, 0, i32(NX) - 1);
            let py = clamp(ny, 0, i32(NY) - 1);
            let pz = clamp(nz, 0, i32(NZ) - 1);
            neighbour_idx = u32(px + py * i32(NX) + pz * i32(NX * NY));
        }

        var pulled_f = 0.0;
        switch boundary_type {
            //{BOUNDARY_SWITCH_CASES_ODD}
            default: {
                //{FLUID_PULL_LOGIC_ODD}
            }
        }
        
        f_local[i] = pulled_f;
        rho += pulled_f;
        u += vec3<f32>(f32(EX[i]), f32(EY[i]), f32(EZ[i])) * pulled_f;
    }

    //{POST_STREAMING_CORRECTION}

    if (boundary_type == 0u) {
        //{COLLISION_LOGIC}
    }

    //{UNROLLED_WRITE_ODD}
}
"#;

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
    
    //{UNROLLED_FAST_PATH_EVEN}

    //{COLLISION_LOGIC}

    //{UNROLLED_WRITE_EVEN}
}
"#;

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
    
    //{UNROLLED_FAST_PATH_ODD}

    //{COLLISION_LOGIC}

    //{UNROLLED_WRITE_ODD}
}
"#;