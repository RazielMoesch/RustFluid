pub const BASE_STEP_EVEN_2D: &str = r#"
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
//{REFLECT_Y}
//{REFLECT_X}

struct BoundaryConfig {
    vel: vec2<f32>,
    density: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> fa: array<f32>; // POP_FA
@group(0) @binding(1) var<storage, read_write> fb: array<f32>; // POP_FB
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> boundary_configs: array<BoundaryConfig>;

//{PRECISION_HELPERS}

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
    let my_flag = flags[cell_idx];
    if ((my_flag & 0x00800000u) != 0u) {
        //{UNROLLED_FAST_PATH_EVEN}
    } else {
        for (var i: u32 = 0u; i < Q; i += 1u) {
            let ex = EX[i];
            let ey = EY[i];
            let neighbour_pos = cell_pos - vec2<i32>(ex, ey);
            var pulled_f: f32 = 0.0;

            if (neighbour_pos.x < 0 || neighbour_pos.y < 0 || neighbour_pos.x >= i32(NX) || neighbour_pos.y >= i32(NY)) {
                // Out of bounds domain limits
                //{BOUNCE_BACK_LOGIC_EVEN}
            } else {
                let neighbour_idx = u32(neighbour_pos.x) + (u32(neighbour_pos.y) * NX);
                let raw_flag = flags[neighbour_idx];
                let boundary_type = raw_flag >> FLAG_TYPE_SHIFT;
                let cfg_id = raw_flag & FLAG_ID_MASK;

                switch boundary_type {
                    //{BOUNDARY_SWITCH_CASES_EVEN}
                    default: {
                        pulled_f = 0.0;
                    }
                }
            }

            f_local[i] = pulled_f;
            rho += pulled_f;
            u += vec2<f32>(f32(ex), f32(ey)) * pulled_f;
        }
    }

    // --- POST-STREAMING CORRECTION ---
    //{POST_STREAMING_CORRECTION}

    // --- COLLISION ---
    //{COLLISION_LOGIC}

    // --- WRITE (INVERTED) ---
    //{UNROLLED_WRITE_EVEN}
}
"#;

pub const BASE_STEP_ODD_2D: &str = r#"
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
//{REFLECT_Y}
//{REFLECT_X}

struct BoundaryConfig {
    vel: vec2<f32>,
    density: f32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> fa: array<f32>; // POP_FA
@group(0) @binding(1) var<storage, read_write> fb: array<f32>; // POP_FB
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> boundary_configs: array<BoundaryConfig>;

//{PRECISION_HELPERS}

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
    let my_flag = flags[cell_idx];
    if ((my_flag & 0x00800000u) != 0u) {
        //{UNROLLED_FAST_PATH_ODD}
    } else {
        for (var i: u32 = 0u; i < Q; i += 1u) {
            let ex = EX[i];
            let ey = EY[i];
            let neighbour_pos = cell_pos - vec2<i32>(ex, ey);
            var pulled_f: f32 = 0.0;

            if (neighbour_pos.x < 0 || neighbour_pos.y < 0 || neighbour_pos.x >= i32(NX) || neighbour_pos.y >= i32(NY)) {
                // Out of bounds domain limits
                //{BOUNCE_BACK_LOGIC_ODD}
            } else {
                let neighbour_idx = u32(neighbour_pos.x) + (u32(neighbour_pos.y) * NX);
                let raw_flag = flags[neighbour_idx];
                let boundary_type = raw_flag >> FLAG_TYPE_SHIFT;
                let cfg_id = raw_flag & FLAG_ID_MASK;

                switch boundary_type {
                    //{BOUNDARY_SWITCH_CASES_ODD}
                    default: {
                        pulled_f = 0.0;
                    }
                }
            }

            f_local[i] = pulled_f;
            rho += pulled_f;
            u += vec2<f32>(f32(ex), f32(ey)) * pulled_f;
        }
    }

    // --- POST-STREAMING CORRECTION ---
    //{POST_STREAMING_CORRECTION}

    // --- COLLISION ---
    //{COLLISION_LOGIC}

    // --- WRITE (NON-INVERTED) ---
    //{UNROLLED_WRITE_ODD}
}
"#;

pub const BASE_INIT_2D: &str = r#"
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

@group(0) @binding(0) var<storage, read_write> fa: array<f32>; // POP_STORAGE
@group(0) @binding(2) var<storage, read> flags: array<u32>;

//{PRECISION_HELPERS}

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
        store_fa(cell_idx + i * TOTAL_CELLS, feq);
    }

}
"#;

pub const BASE_EXTRACT_2D: &str = r#"
@id(100) override NX: u32 = 1920;
@id(101) override NY: u32 = 1080;
@id(105) override wgs_x: u32 = 8;
@id(106) override wgs_y: u32 = 8;
@id(107) override is_even: u32 = 1u;

override TOTAL_CELLS = NX * NY;

//{Q}
//{EX}
//{EY}
//{OPP}
//{WEIGHTS}

@group(0) @binding(0) var<storage, read> fa: array<f32>; // POP_FA
@group(0) @binding(1) var<storage, read_write> macro_data: array<vec4<f32>>;

//{PRECISION_HELPERS}

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

        let read_idx = select(i, OPP[i], is_even != 0u);
        let pop = load_fa(read_idx * TOTAL_CELLS + cell_idx);
        rho += pop;
        u += vec2<f32>(f32(EX[i]), f32(EY[i])) * pop;

    } 

    if ( rho > 0.0 ) {
        u = u / rho;
    }

    macro_data[cell_idx] = vec4<f32>(u.x, u.y, 0.0, rho);

}
"#;
