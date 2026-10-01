use crate::sim::common::precision::PrecisionConfig;
use crate::sim::d3::lattice::Lattice3D;
use crate::sim::d3::collision::Collision3D;
use crate::sim::d3::boundary::Boundary3D;
use super::templates::*;

pub struct ShaderCompiler3D;

impl ShaderCompiler3D {
    pub fn compile(
        lattice: &dyn Lattice3D,
        collision: &dyn Collision3D,
        boundaries: &[&dyn Boundary3D],
        precision_cfg: &PrecisionConfig,
        pure_fluid: bool,
        init_type: crate::sim::d3::config::InitType,
        config: &crate::sim::d3::config::SimulationConfig3D,
    ) -> (String, String, String, String) { // (init, step_even, step_odd, extract)
        
        let q = lattice.q();
        let ex_arr = lattice.ex_array();
        let ey_arr = lattice.ey_array();
        let ez_arr = lattice.ez_array();
        let opp_arr = lattice.opp_array();

        let init_custom_logic = match init_type {
            crate::sim::d3::config::InitType::Uniform => String::new(),
            crate::sim::d3::config::InitType::TaylorGreen => r#"
    // Taylor-Green Vortex Initialization
    // A = 0.1, k = 1.0 (requires L=2pi, mapped to NX, NY, NZ)
    let px = f32(x) / f32(NX) * 2.0 * 3.14159265;
    let py = f32(y) / f32(NY) * 2.0 * 3.14159265;
    let pz = f32(z) / f32(NZ) * 2.0 * 3.14159265;
    
    u.x = 0.1 * sin(px) * cos(py) * cos(pz);
    u.y = -0.1 * cos(px) * sin(py) * cos(pz);
    u.z = 0.0;
    
    rho = 1.0; 
            "#.to_string(),
        };

        // 1. Init Shader
        let init_wgsl = format!("{}\n{}", precision_cfg.enable_directive, BASE_INIT_3D)
            .replace("override NX: u32 = 1920;", &format!("override NX: u32 = {};", config.nx))
            .replace("override NY: u32 = 1080;", &format!("override NY: u32 = {};", config.ny))
            .replace("override NZ: u32 = 1080;", &format!("override NZ: u32 = {};", config.nz))
            .replace("override wgs_x: u32 = 8;", &format!("override wgs_x: u32 = {};", config.wgs_x))
            .replace("override wgs_y: u32 = 8;", &format!("override wgs_y: u32 = {};", config.wgs_y))
            .replace("override wgs_z: u32 = 1;", &format!("override wgs_z: u32 = {};", config.wgs_z))
            .replace("override FORCE_X: f32 = 0.0;", &format!("override FORCE_X: f32 = {:.6};", config.force_x))
            .replace("override FORCE_Y: f32 = 0.0;", &format!("override FORCE_Y: f32 = {:.6};", config.force_y))
            .replace("override FORCE_Z: f32 = 0.0;", &format!("override FORCE_Z: f32 = {:.6};", config.force_z))
            .replace("override PERIODIC_X: u32 = 0u;", &format!("override PERIODIC_X: u32 = {}u;", if config.periodic_x { 1 } else { 0 }))
            .replace("override OMEGA: f32 = 1.0;", &format!("override OMEGA: f32 = {:.6};", config.omega))
            .replace("//{Q}", &lattice.wgsl_constants())
            .replace("array<f32>; // POP_STORAGE", &format!("array<{}>;", precision_cfg.pop_type))
            .replace("//{PRECISION_HELPERS}", &precision_cfg.init_helpers())
            .replace("//{INIT_CUSTOM_LOGIC}", &init_custom_logic);

        // 2. Extract Shader
        let extract_wgsl = format!("{}\n{}", precision_cfg.enable_directive, BASE_EXTRACT_3D)
            .replace("override NX: u32 = 1920;", &format!("override NX: u32 = {};", config.nx))
            .replace("override NY: u32 = 1080;", &format!("override NY: u32 = {};", config.ny))
            .replace("override NZ: u32 = 1080;", &format!("override NZ: u32 = {};", config.nz))
            .replace("override wgs_x: u32 = 8;", &format!("override wgs_x: u32 = {};", config.wgs_x))
            .replace("override wgs_y: u32 = 8;", &format!("override wgs_y: u32 = {};", config.wgs_y))
            .replace("override wgs_z: u32 = 1;", &format!("override wgs_z: u32 = {};", config.wgs_z))
            .replace("override FORCE_X: f32 = 0.0;", &format!("override FORCE_X: f32 = {:.6};", config.force_x))
            .replace("override FORCE_Y: f32 = 0.0;", &format!("override FORCE_Y: f32 = {:.6};", config.force_y))
            .replace("override FORCE_Z: f32 = 0.0;", &format!("override FORCE_Z: f32 = {:.6};", config.force_z))
            .replace("override PERIODIC_X: u32 = 0u;", &format!("override PERIODIC_X: u32 = {}u;", if config.periodic_x { 1 } else { 0 }))
            .replace("override OMEGA: f32 = 1.0;", &format!("override OMEGA: f32 = {:.6};", config.omega))
            .replace("//{Q}", &lattice.wgsl_constants())
            .replace("array<f32>; // POP_FA", &format!("array<{}>;", precision_cfg.pop_type))
            .replace("//{PRECISION_HELPERS}", &precision_cfg.extract_helpers());

        // 3. Step Shaders
        let mut unrolled_fast_even = String::new();
        let mut unrolled_fast_odd = String::new();
        let mut unrolled_write_even = String::new();
        let mut unrolled_write_odd = String::new();

        for i in 0..q as usize {
            let dx = -ex_arr[i];
            let dy = -ey_arr[i];
            let dz = -ez_arr[i];
            let opp = opp_arr[i];

            unrolled_fast_even.push_str(&format!(
                "
        {{
            let offset = {dx} + {dy} * i32(NX) + {dz} * i32(NX * NY);
            let n_idx = u32(i32(cell_idx) + offset);
            let pulled_f = load_fa(n_idx + {i}u * TOTAL_CELLS);
            f_local[{i}] = pulled_f;
            rho += pulled_f;
            u += vec3<f32>({ex}.0, {ey}.0, {ez}.0) * pulled_f;
        }}",
                dx = dx, dy = dy, dz = dz, i = i, ex = ex_arr[i], ey = ey_arr[i], ez = ez_arr[i]
            ));

            unrolled_fast_odd.push_str(&format!(
                "
        {{
            let offset = {dx} + {dy} * i32(NX) + {dz} * i32(NX * NY);
            let n_idx = u32(i32(cell_idx) + offset);
            let pulled_f = load_fa(n_idx + {opp}u * TOTAL_CELLS);
            f_local[{i}] = pulled_f;
            rho += pulled_f;
            u += vec3<f32>({ex}.0, {ey}.0, {ez}.0) * pulled_f;
        }}",
                dx = dx, dy = dy, dz = dz, i = i, opp = opp, ex = ex_arr[i], ey = ey_arr[i], ez = ez_arr[i]
            ));

            unrolled_write_even.push_str(&format!("    store_fb(cell_idx + {opp}u * TOTAL_CELLS, f_local[{i}]);\n", opp = opp, i = i));
            unrolled_write_odd.push_str(&format!("    store_fb(cell_idx + {i}u * TOTAL_CELLS, f_local[{i}]);\n", i = i));
        }

        let mut boundary_switch_even = String::new();
        let mut boundary_switch_odd = String::new();
        let mut post_streaming_logic = String::new();
        let mut fallback_fluid_even = String::new();
        let mut fallback_fluid_odd = String::new();

        for b in boundaries {
            if b.type_id() == 0 {
                if let Some(pull) = b.pull_even() { fallback_fluid_even = pull.to_string(); }
                if let Some(pull) = b.pull_odd() { fallback_fluid_odd = pull.to_string(); }
            }
            if let Some(pull_even) = b.pull_even() {
                boundary_switch_even.push_str(&format!("case {}u: {{ {} }}\n", b.type_id(), pull_even));
            }
            if let Some(pull_odd) = b.pull_odd() {
                boundary_switch_odd.push_str(&format!("case {}u: {{ {} }}\n", b.type_id(), pull_odd));
            }
            if let Some(post) = b.post_streaming() {
                post_streaming_logic.push_str(post);
                post_streaming_logic.push('\n');
            }
        }

        let step_even_template = if pure_fluid { PURE_FLUID_STEP_EVEN_3D } else { BASE_STEP_EVEN_3D };
        let step_even_wgsl = format!("{}\n{}", precision_cfg.enable_directive, step_even_template)
            .replace("override NX: u32 = 1920;", &format!("override NX: u32 = {};", config.nx))
            .replace("override NY: u32 = 1080;", &format!("override NY: u32 = {};", config.ny))
            .replace("override NZ: u32 = 1080;", &format!("override NZ: u32 = {};", config.nz))
            .replace("override wgs_x: u32 = 8;", &format!("override wgs_x: u32 = {};", config.wgs_x))
            .replace("override wgs_y: u32 = 8;", &format!("override wgs_y: u32 = {};", config.wgs_y))
            .replace("override wgs_z: u32 = 1;", &format!("override wgs_z: u32 = {};", config.wgs_z))
            .replace("override FORCE_X: f32 = 0.0;", &format!("override FORCE_X: f32 = {:.6};", config.force_x))
            .replace("override FORCE_Y: f32 = 0.0;", &format!("override FORCE_Y: f32 = {:.6};", config.force_y))
            .replace("override FORCE_Z: f32 = 0.0;", &format!("override FORCE_Z: f32 = {:.6};", config.force_z))
            .replace("override PERIODIC_X: u32 = 0u;", &format!("override PERIODIC_X: u32 = {}u;", if config.periodic_x { 1 } else { 0 }))
            .replace("override OMEGA: f32 = 1.0;", &format!("override OMEGA: f32 = {:.6};", config.omega))
            .replace("//{Q}", &lattice.wgsl_constants())
            .replace("//{UNROLLED_FAST_PATH_EVEN}", &unrolled_fast_even)
            .replace("//{UNROLLED_WRITE_EVEN}", &unrolled_write_even)
            .replace("//{FLUID_PULL_LOGIC_EVEN}", &fallback_fluid_even)
            .replace("//{BOUNDARY_SWITCH_CASES_EVEN}", &boundary_switch_even)
            .replace("//{POST_STREAMING_CORRECTION}", &post_streaming_logic)
            .replace("//{COLLISION_LOGIC}", collision.wgsl())
            .replace("array<f32>; // POP_FA", &format!("array<{}>;", precision_cfg.pop_type))
            .replace("array<f32>; // POP_FB", &format!("array<{}>;", precision_cfg.pop_type))
            .replace("//{PRECISION_HELPERS}", &precision_cfg.step_helpers())
            .replace("//{REFLECT_X}", lattice.reflect_x().unwrap_or(""))
            .replace("//{REFLECT_Y}", lattice.reflect_y().unwrap_or(""))
            .replace("//{REFLECT_Z}", lattice.reflect_z().unwrap_or(""));

        let step_odd_template = if pure_fluid { PURE_FLUID_STEP_ODD_3D } else { BASE_STEP_ODD_3D };
        let step_odd_wgsl = format!("{}\n{}", precision_cfg.enable_directive, step_odd_template)
            .replace("override NX: u32 = 1920;", &format!("override NX: u32 = {};", config.nx))
            .replace("override NY: u32 = 1080;", &format!("override NY: u32 = {};", config.ny))
            .replace("override NZ: u32 = 1080;", &format!("override NZ: u32 = {};", config.nz))
            .replace("override wgs_x: u32 = 8;", &format!("override wgs_x: u32 = {};", config.wgs_x))
            .replace("override wgs_y: u32 = 8;", &format!("override wgs_y: u32 = {};", config.wgs_y))
            .replace("override wgs_z: u32 = 1;", &format!("override wgs_z: u32 = {};", config.wgs_z))
            .replace("override FORCE_X: f32 = 0.0;", &format!("override FORCE_X: f32 = {:.6};", config.force_x))
            .replace("override FORCE_Y: f32 = 0.0;", &format!("override FORCE_Y: f32 = {:.6};", config.force_y))
            .replace("override FORCE_Z: f32 = 0.0;", &format!("override FORCE_Z: f32 = {:.6};", config.force_z))
            .replace("override PERIODIC_X: u32 = 0u;", &format!("override PERIODIC_X: u32 = {}u;", if config.periodic_x { 1 } else { 0 }))
            .replace("override OMEGA: f32 = 1.0;", &format!("override OMEGA: f32 = {:.6};", config.omega))
            .replace("//{Q}", &lattice.wgsl_constants())
            .replace("//{UNROLLED_FAST_PATH_ODD}", &unrolled_fast_odd)
            .replace("//{UNROLLED_WRITE_ODD}", &unrolled_write_odd)
            .replace("//{FLUID_PULL_LOGIC_ODD}", &fallback_fluid_odd)
            .replace("//{BOUNDARY_SWITCH_CASES_ODD}", &boundary_switch_odd)
            .replace("//{POST_STREAMING_CORRECTION}", &post_streaming_logic)
            .replace("//{COLLISION_LOGIC}", collision.wgsl())
            .replace("array<f32>; // POP_FA", &format!("array<{}>;", precision_cfg.pop_type))
            .replace("array<f32>; // POP_FB", &format!("array<{}>;", precision_cfg.pop_type))
            .replace("//{PRECISION_HELPERS}", &precision_cfg.step_helpers())
            .replace("//{REFLECT_X}", lattice.reflect_x().unwrap_or(""))
            .replace("//{REFLECT_Y}", lattice.reflect_y().unwrap_or(""))
            .replace("//{REFLECT_Z}", lattice.reflect_z().unwrap_or(""));

        (init_wgsl, step_even_wgsl, step_odd_wgsl, extract_wgsl)
    }
}
