//! Assembles complete D2 WGSL programs from solver components.

use super::templates::*;
use crate::sim::common::precision::PrecisionConfig;
use crate::sim::d2::boundary::Boundary2D;
use crate::sim::d2::collision::Collision2D;
use crate::sim::d2::lattice::Lattice2D;

/// Stateless source generator for D2 initialization, stepping, and extraction.
pub struct ShaderCompiler2D;

impl ShaderCompiler2D {
    pub fn compile(
        lattice: &dyn Lattice2D,
        collision: &dyn Collision2D,
        boundaries: &[&dyn Boundary2D],
        precision_cfg: &PrecisionConfig,
    ) -> (String, String, String, String) {
        // (init, step_even, step_odd, extract)

        let q = lattice.q();
        let ex_arr = lattice.ex_array();
        let ey_arr = lattice.ey_array();
        let opp_arr = lattice.opp_array();

        // 1. Init Shader
        let init_wgsl = format!("{}\n{}", precision_cfg.enable_directive, BASE_INIT_2D)
            .replace("//{Q}", &lattice.wgsl_constants())
            .replace(
                "array<f32>; // POP_STORAGE",
                &format!("array<{}>;", precision_cfg.pop_type),
            )
            .replace("//{PRECISION_HELPERS}", &precision_cfg.init_helpers());

        // 2. Extract Shader
        let extract_wgsl = format!("{}\n{}", precision_cfg.enable_directive, BASE_EXTRACT_2D)
            .replace("//{Q}", &lattice.wgsl_constants())
            .replace(
                "array<f32>; // POP_FA",
                &format!("array<{}>;", precision_cfg.pop_type),
            )
            .replace("//{PRECISION_HELPERS}", &precision_cfg.extract_helpers());

        // 3. Step Shaders
        let mut unrolled_fast_even = String::new();
        let mut unrolled_fast_odd = String::new();
        let mut unrolled_write_even = String::new();
        let mut unrolled_write_odd = String::new();

        for i in 0..q as usize {
            let dx = -ex_arr[i];
            let dy = -ey_arr[i];
            let opp = opp_arr[i];

            unrolled_fast_even.push_str(&format!(
                "
        {{
            let offset = {dx} + {dy} * i32(NX);
            let n_idx = u32(i32(cell_idx) + offset);
            let pulled_f = load_fa(n_idx + {i}u * TOTAL_CELLS);
            f_local[{i}] = pulled_f;
            rho += pulled_f;
            u += vec2<f32>({ex}.0, {ey}.0) * pulled_f;
        }}",
                dx = dx,
                dy = dy,
                i = i,
                ex = ex_arr[i],
                ey = ey_arr[i]
            ));

            unrolled_fast_odd.push_str(&format!(
                "
        {{
            let offset = {dx} + {dy} * i32(NX);
            let n_idx = u32(i32(cell_idx) + offset);
            let pulled_f = load_fa(n_idx + {opp}u * TOTAL_CELLS);
            f_local[{i}] = pulled_f;
            rho += pulled_f;
            u += vec2<f32>({ex}.0, {ey}.0) * pulled_f;
        }}",
                dx = dx,
                dy = dy,
                i = i,
                opp = opp,
                ex = ex_arr[i],
                ey = ey_arr[i]
            ));

            unrolled_write_even.push_str(&format!(
                "    store_fb(cell_idx + {opp}u * TOTAL_CELLS, f_local[{i}]);\n",
                opp = opp,
                i = i
            ));
            unrolled_write_odd.push_str(&format!(
                "    store_fb(cell_idx + {i}u * TOTAL_CELLS, f_local[{i}]);\n",
                i = i
            ));
        }

        let mut boundary_switch_even = String::new();
        let mut boundary_switch_odd = String::new();
        let mut post_streaming_logic = String::new();
        let mut fallback_fluid_even = String::new();
        let mut fallback_fluid_odd = String::new();

        for b in boundaries {
            if b.type_id() == 0 {
                if let Some(pull) = b.pull_even() {
                    fallback_fluid_even = pull.to_string();
                }
                if let Some(pull) = b.pull_odd() {
                    fallback_fluid_odd = pull.to_string();
                }
            }
            if let Some(pull_even) = b.pull_even() {
                boundary_switch_even.push_str(&format!(
                    "case {}u: {{ {} }}\n",
                    b.type_id(),
                    pull_even
                ));
            }
            if let Some(pull_odd) = b.pull_odd() {
                boundary_switch_odd.push_str(&format!(
                    "case {}u: {{ {} }}\n",
                    b.type_id(),
                    pull_odd
                ));
            }
            if let Some(post) = b.post_streaming() {
                post_streaming_logic.push_str(post);
                post_streaming_logic.push('\n');
            }
        }

        let step_even_wgsl = format!("{}\n{}", precision_cfg.enable_directive, BASE_STEP_EVEN_2D)
            .replace("//{Q}", &lattice.wgsl_constants())
            .replace("//{UNROLLED_FAST_PATH_EVEN}", &unrolled_fast_even)
            .replace("//{UNROLLED_WRITE_EVEN}", &unrolled_write_even)
            .replace("//{FLUID_PULL_LOGIC_EVEN}", &fallback_fluid_even)
            .replace("//{BOUNDARY_SWITCH_CASES_EVEN}", &boundary_switch_even)
            .replace("//{POST_STREAMING_CORRECTION}", &post_streaming_logic)
            .replace("//{COLLISION_LOGIC}", collision.wgsl())
            .replace(
                "array<f32>; // POP_FA",
                &format!("array<{}>;", precision_cfg.pop_type),
            )
            .replace(
                "array<f32>; // POP_FB",
                &format!("array<{}>;", precision_cfg.pop_type),
            )
            .replace("//{PRECISION_HELPERS}", &precision_cfg.step_helpers())
            .replace("//{REFLECT_X}", lattice.reflect_x().unwrap_or(""))
            .replace("//{REFLECT_Y}", lattice.reflect_y().unwrap_or(""));

        let step_odd_wgsl = format!("{}\n{}", precision_cfg.enable_directive, BASE_STEP_ODD_2D)
            .replace("//{Q}", &lattice.wgsl_constants())
            .replace("//{UNROLLED_FAST_PATH_ODD}", &unrolled_fast_odd)
            .replace("//{UNROLLED_WRITE_ODD}", &unrolled_write_odd)
            .replace("//{FLUID_PULL_LOGIC_ODD}", &fallback_fluid_odd)
            .replace("//{BOUNDARY_SWITCH_CASES_ODD}", &boundary_switch_odd)
            .replace("//{POST_STREAMING_CORRECTION}", &post_streaming_logic)
            .replace("//{COLLISION_LOGIC}", collision.wgsl())
            .replace(
                "array<f32>; // POP_FA",
                &format!("array<{}>;", precision_cfg.pop_type),
            )
            .replace(
                "array<f32>; // POP_FB",
                &format!("array<{}>;", precision_cfg.pop_type),
            )
            .replace("//{PRECISION_HELPERS}", &precision_cfg.step_helpers())
            .replace("//{REFLECT_X}", lattice.reflect_x().unwrap_or(""))
            .replace("//{REFLECT_Y}", lattice.reflect_y().unwrap_or(""));

        (init_wgsl, step_even_wgsl, step_odd_wgsl, extract_wgsl)
    }
}
