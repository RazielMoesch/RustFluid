//! Assembles specialized in-place D3 WGSL programs from solver components.

use super::templates::*;
use crate::sim::common::precision::PrecisionConfig;
use crate::sim::d3::boundary::Boundary3D;
use crate::sim::d3::collision::Collision3D;
use crate::sim::d3::lattice::Lattice3D;

/// Stateless source generator for D3 initialization, stepping, and extraction.
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
    ) -> (String, String, String, String) {
        // (init, step_even, step_odd, extract)
        // Fixed-point formatting silently rounded the FluidX3D body force
        // (1.849606e-7) to zero. Scientific notation preserves small values.
        let wgsl_f32 = |value: f32| format!("{value:.9e}");

        let q = lattice.q();
        let ex_arr = lattice.ex_array();
        let ey_arr = lattice.ey_array();
        let ez_arr = lattice.ez_array();
        let opp_arr = lattice.opp_array();

        // NVIDIA's WGSL path does not consistently unroll the direction loop.
        // BGK is the throughput reference, so emit its fixed-Q arithmetic as
        // straight-line code while leaving custom collision models untouched.
        let collision_logic = if collision.name() == "BGK" {
            let mut source = String::from(
                r#"
    let F = vec3<f32>(FORCE_X, FORCE_Y, FORCE_Z);
    u = (u + 0.5 * F) / rho;
    let u_sq_term = 1.5 * dot(u, u);
    let omega_factor = 1.0 - 0.5 * OMEGA;
    let uF = dot(u, F);
"#,
            );
            for i in 0..q as usize {
                source.push_str(&format!(
                    "    {{\n\
                         let eu = {ex}.0 * u.x + {ey}.0 * u.y + {ez}.0 * u.z;\n\
                         let eF = {ex}.0 * F.x + {ey}.0 * F.y + {ez}.0 * F.z;\n\
                         let feq = WEIGHTS[{i}] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - u_sq_term);\n\
                         let source_i = WEIGHTS[{i}] * omega_factor * (3.0 * (eF - uF) + 9.0 * eu * eF);\n\
                         f_local[{i}] = f_local[{i}] - OMEGA * (f_local[{i}] - feq) + source_i;\n\
                     }}\n",
                    ex = ex_arr[i],
                    ey = ey_arr[i],
                    ez = ez_arr[i],
                ));
            }
            source
        } else {
            collision.wgsl().to_string()
        };

        let init_custom_logic = match init_type {
            crate::sim::d3::config::InitType::Uniform => format!(
                "rho = {};\n    u = vec3<f32>({}, {}, {});",
                wgsl_f32(config.rho_init),
                wgsl_f32(config.u_x_init),
                wgsl_f32(config.u_y_init),
                wgsl_f32(config.u_z_init)
            ),
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
            "#
            .to_string(),
        };

        // 1. Init Shader
        let init_wgsl = format!("{}\n{}", precision_cfg.enable_directive, IN_PLACE_INIT_3D)
            .replace(
                "override NX: u32 = 1920;",
                &format!("override NX: u32 = {};", config.nx),
            )
            .replace(
                "override NY: u32 = 1080;",
                &format!("override NY: u32 = {};", config.ny),
            )
            .replace(
                "override NZ: u32 = 1080;",
                &format!("override NZ: u32 = {};", config.nz),
            )
            .replace(
                "override wgs_x: u32 = 8;",
                &format!("override wgs_x: u32 = {};", config.wgs_x),
            )
            .replace(
                "override wgs_y: u32 = 8;",
                &format!("override wgs_y: u32 = {};", config.wgs_y),
            )
            .replace(
                "override wgs_z: u32 = 1;",
                &format!("override wgs_z: u32 = {};", config.wgs_z),
            )
            .replace(
                "override FORCE_X: f32 = 0.0;",
                &format!("override FORCE_X: f32 = {};", wgsl_f32(config.force_x)),
            )
            .replace(
                "override FORCE_Y: f32 = 0.0;",
                &format!("override FORCE_Y: f32 = {};", wgsl_f32(config.force_y)),
            )
            .replace(
                "override FORCE_Z: f32 = 0.0;",
                &format!("override FORCE_Z: f32 = {};", wgsl_f32(config.force_z)),
            )
            .replace(
                "override PERIODIC_X: u32 = 0u;",
                &format!(
                    "override PERIODIC_X: u32 = {}u;",
                    if config.periodic_x { 1 } else { 0 }
                ),
            )
            .replace(
                "override PERIODIC_Y: u32 = 0u;",
                &format!(
                    "override PERIODIC_Y: u32 = {}u;",
                    if config.periodic_y { 1 } else { 0 }
                ),
            )
            .replace(
                "override PERIODIC_Z: u32 = 0u;",
                &format!(
                    "override PERIODIC_Z: u32 = {}u;",
                    if config.periodic_z { 1 } else { 0 }
                ),
            )
            .replace(
                "override OMEGA: f32 = 1.0;",
                &format!("override OMEGA: f32 = {};", wgsl_f32(config.omega)),
            )
            .replace("//{Q}", &lattice.wgsl_constants())
            .replace(
                "array<f32>; // POP_STORAGE",
                &format!("array<{}>;", precision_cfg.pop_type),
            )
            .replace("//{PRECISION_HELPERS}", &precision_cfg.init_helpers())
            .replace("//{INIT_CUSTOM_LOGIC}", &init_custom_logic);

        // 2. Extract Shader
        let extract_wgsl = format!(
            "{}\n{}",
            precision_cfg.enable_directive, IN_PLACE_EXTRACT_3D
        )
        .replace(
            "override NX: u32 = 1920;",
            &format!("override NX: u32 = {};", config.nx),
        )
        .replace(
            "override NY: u32 = 1080;",
            &format!("override NY: u32 = {};", config.ny),
        )
        .replace(
            "override NZ: u32 = 1080;",
            &format!("override NZ: u32 = {};", config.nz),
        )
        .replace(
            "override wgs_x: u32 = 8;",
            &format!("override wgs_x: u32 = {};", config.wgs_x),
        )
        .replace(
            "override wgs_y: u32 = 8;",
            &format!("override wgs_y: u32 = {};", config.wgs_y),
        )
        .replace(
            "override wgs_z: u32 = 1;",
            &format!("override wgs_z: u32 = {};", config.wgs_z),
        )
        .replace(
            "override FORCE_X: f32 = 0.0;",
            &format!("override FORCE_X: f32 = {};", wgsl_f32(config.force_x)),
        )
        .replace(
            "override FORCE_Y: f32 = 0.0;",
            &format!("override FORCE_Y: f32 = {};", wgsl_f32(config.force_y)),
        )
        .replace(
            "override FORCE_Z: f32 = 0.0;",
            &format!("override FORCE_Z: f32 = {};", wgsl_f32(config.force_z)),
        )
        .replace(
            "override PERIODIC_X: u32 = 0u;",
            &format!(
                "override PERIODIC_X: u32 = {}u;",
                if config.periodic_x { 1 } else { 0 }
            ),
        )
        .replace(
            "override PERIODIC_Y: u32 = 0u;",
            &format!(
                "override PERIODIC_Y: u32 = {}u;",
                if config.periodic_y { 1 } else { 0 }
            ),
        )
        .replace(
            "override PERIODIC_Z: u32 = 0u;",
            &format!(
                "override PERIODIC_Z: u32 = {}u;",
                if config.periodic_z { 1 } else { 0 }
            ),
        )
        .replace(
            "override OMEGA: f32 = 1.0;",
            &format!("override OMEGA: f32 = {};", wgsl_f32(config.omega)),
        )
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
            let opp = opp_arr[i] as usize;
            let pair = i.min(opp);
            let other = opp_arr[pair] as usize;
            let plus_offset = ex_arr[pair]
                + ey_arr[pair] * config.nx as i32
                + ez_arr[pair] * (config.nx * config.ny) as i32;

            let even_address = if i == 0 {
                "cell_idx".to_string()
            } else if i == pair {
                format!("cell_idx + {other}u * TOTAL_CELLS")
            } else {
                format!("u32(i32(cell_idx) + {plus_offset}) + {pair}u * TOTAL_CELLS")
            };
            let odd_address = if i == 0 {
                "cell_idx".to_string()
            } else if i == pair {
                format!("cell_idx + {pair}u * TOTAL_CELLS")
            } else {
                format!("u32(i32(cell_idx) + {plus_offset}) + {other}u * TOTAL_CELLS")
            };

            let mut u_updates = String::new();
            if ex_arr[i] == 1 {
                u_updates.push_str(&format!("        u.x += pulled_{i};\n"));
            }
            if ex_arr[i] == -1 {
                u_updates.push_str(&format!("        u.x -= pulled_{i};\n"));
            }
            if ey_arr[i] == 1 {
                u_updates.push_str(&format!("        u.y += pulled_{i};\n"));
            }
            if ey_arr[i] == -1 {
                u_updates.push_str(&format!("        u.y -= pulled_{i};\n"));
            }
            if ez_arr[i] == 1 {
                u_updates.push_str(&format!("        u.z += pulled_{i};\n"));
            }
            if ez_arr[i] == -1 {
                u_updates.push_str(&format!("        u.z -= pulled_{i};\n"));
            }

            unrolled_fast_even.push_str(&format!(
                "        let pulled_{i} = load_fa({even_address});\n        f_local[{i}] = pulled_{i};\n        rho += pulled_{i};\n{u_updates}",
            ));
            unrolled_fast_odd.push_str(&format!(
                "        let pulled_{i} = load_fa({odd_address});\n        f_local[{i}] = pulled_{i};\n        rho += pulled_{i};\n{u_updates}",
            ));

            if i == 0 {
                unrolled_write_even.push_str("        store_fb(cell_idx, f_local[0]);\n");
                unrolled_write_odd.push_str("        store_fb(cell_idx, f_local[0]);\n");
            } else if i < opp {
                unrolled_write_even.push_str(&format!(
                    "        store_fb(u32(i32(cell_idx) + {plus_offset}) + {i}u * TOTAL_CELLS, f_local[{i}]);\n        store_fb(cell_idx + {opp}u * TOTAL_CELLS, f_local[{opp}]);\n"
                ));
                unrolled_write_odd.push_str(&format!(
                    "        store_fb(u32(i32(cell_idx) + {plus_offset}) + {opp}u * TOTAL_CELLS, f_local[{i}]);\n        store_fb(cell_idx + {i}u * TOTAL_CELLS, f_local[{opp}]);\n"
                ));
            }
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

        let step_even_template = IN_PLACE_STEP_3D;

        // Sponge (damping) layer: blend f toward the far-field equilibrium over
        // the last `sponge_len` cells so residual wake vorticity is absorbed
        // before it reaches the outlet plane. Applied after collision, so it is
        // independent of the collision model.
        let sponge_logic = if pure_fluid {
            ""
        } else {
            r#"
        // ── Sponge (damping) layer ──
        // SPONGE_LEN == 0 disables it entirely.
        let sponge_len = min(SPONGE_LEN, NX);
        if (sponge_len > 0u) {
            let sponge_start = NX - sponge_len;
            if (x >= sponge_start) {
                // 0 at the sponge entrance, 1 at the last cell.
                let sponge_t = f32(x + 1u - sponge_start) / f32(sponge_len);
                // Quadratic ramp: zero slope at the entrance so the layer does
                // not reflect the vortices it is meant to absorb.
                let sponge_eta = clamp(SPONGE_STRENGTH * sponge_t * sponge_t, 0.0, 1.0);
                if (sponge_eta > 0.0) {
                    let sponge_target = boundary_configs[SPONGE_CFG];
                    let sponge_vel = sponge_target.vel;
                    let sponge_rho = sponge_target.density;
                    let sponge_u_sq = dot(sponge_vel, sponge_vel);
                    for (var si: u32 = 0u; si < Q; si += 1u) {
                        let e_vec = vec3<f32>(f32(EX[si]), f32(EY[si]), f32(EZ[si]));
                        let eu = dot(e_vec, sponge_vel);
                        let feq_sponge = WEIGHTS[si] * sponge_rho
                            * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - 1.5 * sponge_u_sq);
                        f_local[si] = f_local[si] + sponge_eta * (feq_sponge - f_local[si]);
                    }
                }
            }
        }
"#
        };

        let step_even_wgsl = format!("{}\n{}", precision_cfg.enable_directive, step_even_template)
            .replace(
                "override NX: u32 = 1920;",
                &format!("override NX: u32 = {};", config.nx),
            )
            .replace(
                "override NY: u32 = 1080;",
                &format!("override NY: u32 = {};", config.ny),
            )
            .replace(
                "override NZ: u32 = 1080;",
                &format!("override NZ: u32 = {};", config.nz),
            )
            .replace(
                "override wgs_x: u32 = 8;",
                &format!("override wgs_x: u32 = {};", config.wgs_x),
            )
            .replace(
                "override wgs_y: u32 = 8;",
                &format!("override wgs_y: u32 = {};", config.wgs_y),
            )
            .replace(
                "override wgs_z: u32 = 1;",
                &format!("override wgs_z: u32 = {};", config.wgs_z),
            )
            .replace(
                "override FORCE_X: f32 = 0.0;",
                &format!("override FORCE_X: f32 = {};", wgsl_f32(config.force_x)),
            )
            .replace(
                "override FORCE_Y: f32 = 0.0;",
                &format!("override FORCE_Y: f32 = {};", wgsl_f32(config.force_y)),
            )
            .replace(
                "override FORCE_Z: f32 = 0.0;",
                &format!("override FORCE_Z: f32 = {};", wgsl_f32(config.force_z)),
            )
            .replace(
                "override PERIODIC_X: u32 = 0u;",
                &format!(
                    "override PERIODIC_X: u32 = {}u;",
                    if config.periodic_x { 1 } else { 0 }
                ),
            )
            .replace(
                "override PERIODIC_Y: u32 = 0u;",
                &format!(
                    "override PERIODIC_Y: u32 = {}u;",
                    if config.periodic_y { 1 } else { 0 }
                ),
            )
            .replace(
                "override PERIODIC_Z: u32 = 0u;",
                &format!(
                    "override PERIODIC_Z: u32 = {}u;",
                    if config.periodic_z { 1 } else { 0 }
                ),
            )
            .replace(
                "override OMEGA: f32 = 1.0;",
                &format!("override OMEGA: f32 = {};", wgsl_f32(config.omega)),
            )
            .replace(
                "override SPONGE_LEN: u32 = 0u;",
                &format!("override SPONGE_LEN: u32 = {}u;", config.sponge_len),
            )
            .replace(
                "override SPONGE_STRENGTH: f32 = 0.0;",
                &format!(
                    "override SPONGE_STRENGTH: f32 = {};",
                    wgsl_f32(config.sponge_strength)
                ),
            )
            .replace(
                "override SPONGE_CFG: u32 = 0u;",
                &format!("override SPONGE_CFG: u32 = {}u;", config.sponge_cfg),
            )
            .replace("//{Q}", &lattice.wgsl_constants())
            .replace("//{UNROLLED_FAST_PATH_EVEN}", &unrolled_fast_even)
            .replace("//{UNROLLED_WRITE_EVEN}", &unrolled_write_even)
            .replace("//{AA_INTERIOR_LOADS}", &unrolled_fast_even)
            .replace("//{AA_INTERIOR_STORES}", &unrolled_write_even)
            .replace("//{FLUID_PULL_LOGIC_EVEN}", &fallback_fluid_even)
            .replace("//{BOUNDARY_SWITCH_CASES}", &boundary_switch_even)
            .replace("//{POST_STREAMING_CORRECTION}", &post_streaming_logic)
            .replace("//{COLLISION_LOGIC}", &collision_logic)
            .replace("//{SPONGE_LOGIC}", sponge_logic)
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
            .replace("//{REFLECT_Y}", lattice.reflect_y().unwrap_or(""))
            .replace("//{REFLECT_Z}", lattice.reflect_z().unwrap_or(""));

        let step_odd_template = IN_PLACE_STEP_3D;
        let step_odd_wgsl = format!("{}\n{}", precision_cfg.enable_directive, step_odd_template)
            .replace("override PHASE: u32 = 0u;", "override PHASE: u32 = 1u;")
            .replace(
                "override NX: u32 = 1920;",
                &format!("override NX: u32 = {};", config.nx),
            )
            .replace(
                "override NY: u32 = 1080;",
                &format!("override NY: u32 = {};", config.ny),
            )
            .replace(
                "override NZ: u32 = 1080;",
                &format!("override NZ: u32 = {};", config.nz),
            )
            .replace(
                "override wgs_x: u32 = 8;",
                &format!("override wgs_x: u32 = {};", config.wgs_x),
            )
            .replace(
                "override wgs_y: u32 = 8;",
                &format!("override wgs_y: u32 = {};", config.wgs_y),
            )
            .replace(
                "override wgs_z: u32 = 1;",
                &format!("override wgs_z: u32 = {};", config.wgs_z),
            )
            .replace(
                "override FORCE_X: f32 = 0.0;",
                &format!("override FORCE_X: f32 = {};", wgsl_f32(config.force_x)),
            )
            .replace(
                "override FORCE_Y: f32 = 0.0;",
                &format!("override FORCE_Y: f32 = {};", wgsl_f32(config.force_y)),
            )
            .replace(
                "override FORCE_Z: f32 = 0.0;",
                &format!("override FORCE_Z: f32 = {};", wgsl_f32(config.force_z)),
            )
            .replace(
                "override PERIODIC_X: u32 = 0u;",
                &format!(
                    "override PERIODIC_X: u32 = {}u;",
                    if config.periodic_x { 1 } else { 0 }
                ),
            )
            .replace(
                "override PERIODIC_Y: u32 = 0u;",
                &format!(
                    "override PERIODIC_Y: u32 = {}u;",
                    if config.periodic_y { 1 } else { 0 }
                ),
            )
            .replace(
                "override PERIODIC_Z: u32 = 0u;",
                &format!(
                    "override PERIODIC_Z: u32 = {}u;",
                    if config.periodic_z { 1 } else { 0 }
                ),
            )
            .replace(
                "override OMEGA: f32 = 1.0;",
                &format!("override OMEGA: f32 = {};", wgsl_f32(config.omega)),
            )
            .replace(
                "override SPONGE_LEN: u32 = 0u;",
                &format!("override SPONGE_LEN: u32 = {}u;", config.sponge_len),
            )
            .replace(
                "override SPONGE_STRENGTH: f32 = 0.0;",
                &format!(
                    "override SPONGE_STRENGTH: f32 = {};",
                    wgsl_f32(config.sponge_strength)
                ),
            )
            .replace(
                "override SPONGE_CFG: u32 = 0u;",
                &format!("override SPONGE_CFG: u32 = {}u;", config.sponge_cfg),
            )
            .replace("//{Q}", &lattice.wgsl_constants())
            .replace("//{UNROLLED_FAST_PATH_ODD}", &unrolled_fast_odd)
            .replace("//{UNROLLED_WRITE_ODD}", &unrolled_write_odd)
            .replace("//{AA_INTERIOR_LOADS}", &unrolled_fast_odd)
            .replace("//{AA_INTERIOR_STORES}", &unrolled_write_odd)
            .replace("//{FLUID_PULL_LOGIC_ODD}", &fallback_fluid_odd)
            .replace("//{BOUNDARY_SWITCH_CASES}", &boundary_switch_odd)
            .replace("//{POST_STREAMING_CORRECTION}", &post_streaming_logic)
            .replace("//{COLLISION_LOGIC}", &collision_logic)
            .replace("//{SPONGE_LOGIC}", sponge_logic)
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
            .replace("//{REFLECT_Y}", lattice.reflect_y().unwrap_or(""))
            .replace("//{REFLECT_Z}", lattice.reflect_z().unwrap_or(""));

        (init_wgsl, step_even_wgsl, step_odd_wgsl, extract_wgsl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::common::precision::{Precision, PrecisionConfig};
    use crate::sim::d3::boundary::outlet::ZeroGradientOutlet;
    use crate::sim::d3::boundary::{Boundary3D, fluid::Fluid};
    use crate::sim::d3::collision::bgk::Bgk;
    use crate::sim::d3::config::SimulationConfig3D;
    use crate::sim::d3::lattice::d3q19::D3Q19;

    #[test]
    fn generated_shader_preserves_small_force_and_uses_wrapped_in_place_topology() {
        let config = SimulationConfig3D {
            nx: 5,
            ny: 6,
            nz: 7,
            force_x: 1.8496058e-7,
            periodic_x: true,
            periodic_y: false,
            periodic_z: true,
            ..Default::default()
        };
        let boundaries: Vec<&dyn Boundary3D> = vec![&Fluid];
        let (_, even, odd, _) = ShaderCompiler3D::compile(
            &D3Q19::new(),
            &Bgk::new(),
            &boundaries,
            &PrecisionConfig::from(Precision::F32),
            false,
            config.init_type,
            &config,
        );
        for shader in [&even, &odd] {
            assert!(shader.contains("override FORCE_X: f32 = 1.849605"));
            assert!(shader.contains("e-7;"));
            // Esoteric Pull needs a bijective storage topology. Physical
            // non-periodic behavior is supplied by boundary nodes.
            assert!(shader.contains("fn wrapped_cell"));
            assert!(shader.contains("fn load_streamed_at"));
            assert!(shader.contains("fn store_collided"));
            assert!(shader.contains("fn prepare_outlet"));
            assert!(shader.contains("let has_direct_neighbors"));
            assert!(shader.contains("let pulled_18 = load_fa"));
            assert!(shader.contains("f_local[18] = f_local[18] - OMEGA"));
            assert!(!shader.contains("//{AA_INTERIOR_"));
        }
        assert!(even.contains("override PHASE: u32 = 0u;"));
        assert!(odd.contains("override PHASE: u32 = 1u;"));
    }

    #[test]
    fn both_in_place_phases_write_every_population_address_exactly_once() {
        let (nx, ny, nz) = (5i32, 6i32, 7i32);
        let cells = (nx * ny * nz) as usize;
        let lattice = D3Q19::new();
        let (ex, ey, ez) = (lattice.ex_array(), lattice.ey_array(), lattice.ez_array());
        let opp = lattice.opp_array();

        for pair in 1..19 {
            if pair >= opp[pair] as usize {
                continue;
            }
            let other = opp[pair] as usize;
            assert_eq!(
                (ex[other], ey[other], ez[other]),
                (-ex[pair], -ey[pair], -ez[pair])
            );
        }

        for phase in 0..=1 {
            let mut addresses = Vec::with_capacity(cells * 19);
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let cell = (x + y * nx + z * nx * ny) as usize;
                        let mut owned_reads = vec![cell];
                        let mut owned_writes = vec![cell];
                        addresses.push(cell);
                        for pair in 1usize..19 {
                            let other = opp[pair] as usize;
                            if pair >= other {
                                continue;
                            }
                            let px = (x + ex[pair]).rem_euclid(nx);
                            let py = (y + ey[pair]).rem_euclid(ny);
                            let pz = (z + ez[pair]).rem_euclid(nz);
                            let plus = (px + py * nx + pz * nx * ny) as usize;
                            if phase == 0 {
                                addresses.push(plus + pair * cells);
                                addresses.push(cell + other * cells);
                                owned_reads.push(cell + other * cells);
                                owned_reads.push(plus + pair * cells);
                                owned_writes.push(plus + pair * cells);
                                owned_writes.push(cell + other * cells);
                            } else {
                                addresses.push(plus + other * cells);
                                addresses.push(cell + pair * cells);
                                owned_reads.push(cell + pair * cells);
                                owned_reads.push(plus + other * cells);
                                owned_writes.push(plus + other * cells);
                                owned_writes.push(cell + pair * cells);
                            }
                        }
                        owned_reads.sort_unstable();
                        owned_writes.sort_unstable();
                        assert_eq!(owned_reads, owned_writes);
                    }
                }
            }
            addresses.sort_unstable();
            addresses.dedup();
            assert_eq!(addresses, (0..cells * 19).collect::<Vec<_>>());
        }
    }

    fn compile_with_config(config: SimulationConfig3D) -> (String, String) {
        let boundaries: Vec<&dyn Boundary3D> = vec![&Fluid, &ZeroGradientOutlet];
        let (_, even, odd, _) = ShaderCompiler3D::compile(
            &D3Q19::new(),
            &Bgk::new(),
            &boundaries,
            &PrecisionConfig::from(Precision::F32),
            config.pure_fluid,
            config.init_type,
            &config,
        );
        (even, odd)
    }

    #[test]
    fn sponge_overrides_and_damping_block_reach_both_step_phases() {
        let config = SimulationConfig3D {
            nx: 576,
            sponge_len: 57,
            sponge_strength: 0.01,
            sponge_cfg: 3,
            ..Default::default()
        };
        let (even, odd) = compile_with_config(config);

        for shader in [&even, &odd] {
            assert!(shader.contains("override SPONGE_LEN: u32 = 57u;"));
            assert!(shader.contains("override SPONGE_CFG: u32 = 3u;"));
            assert!(shader.contains(&format!("override SPONGE_STRENGTH: f32 = {:.9e};", 0.01f32)));
            // The target far-field state is read from the boundary config
            // buffer rather than hard-coded, so it stays in sync with the inlet.
            assert!(shader.contains("boundary_configs[SPONGE_CFG]"));
            // Clamped so an over-long sponge cannot underflow NX - sponge_len.
            assert!(shader.contains("min(SPONGE_LEN, NX)"));
            // Placed after collision so it is collision-model independent.
            let collision = shader.find("f_local[0] = f_local[0] - OMEGA").unwrap();
            let sponge = shader
                .find("sponge_eta * (feq_sponge - f_local[si])")
                .unwrap();
            assert!(collision < sponge);
        }
    }

    #[test]
    fn sponge_defaults_to_disabled() {
        let (even, odd) = compile_with_config(SimulationConfig3D::default());
        for shader in [&even, &odd] {
            assert!(shader.contains("override SPONGE_LEN: u32 = 0u;"));
            assert!(shader.contains("override SPONGE_STRENGTH: f32 = 0.000000000e0;"));
            // The block is emitted but short-circuits on `sponge_len > 0u`.
            assert!(shader.contains("if (sponge_len > 0u)"));
        }
    }

    #[test]
    fn pure_fluid_step_shaders_stay_free_of_sponge_code() {
        // Pure-fluid mode must not execute or emit the damping block.
        let config = SimulationConfig3D {
            pure_fluid: true,
            sponge_len: 32,
            sponge_strength: 0.05,
            ..Default::default()
        };
        let (even, odd) = compile_with_config(config);
        for shader in [&even, &odd] {
            assert!(!shader.contains("sponge_eta"));
        }
    }
}
