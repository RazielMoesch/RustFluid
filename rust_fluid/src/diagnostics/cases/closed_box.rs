//! Conservation and no-slip checks in a sealed bounce-back box.

use crate::diagnostics::metrics::MacroMetrics;
use crate::diagnostics::result::DiagnosticResult;
use crate::runtime::context::HeadlessContext;
use crate::sim::common::precision::Precision;
use crate::sim::d3::boundary::Boundary3D;
use crate::sim::d3::boundary::bounce_back::BounceBack;
use crate::sim::d3::boundary::fluid::Fluid;
use crate::sim::d3::collision::bgk::Bgk;
use crate::sim::d3::config::SimulationConfig3D;
use crate::sim::d3::lattice::d3q19::D3Q19;
use crate::sim::d3::solver::Lbm3D;

fn mass_tolerance(precision: Precision) -> f32 {
    match precision {
        // FP16S is deliberately lossy: every population is rounded when it is
        // written after each time step. A long transient therefore needs a
        // wider conservation budget than the FP32 path.
        Precision::FP16S => 5e-4,
        Precision::F32 | Precision::Auto => 1e-4,
    }
}

/// Runs a closed-box drift test for one precision and grid size.
pub fn run_closed_box(
    ctx: &HeadlessContext,
    precision: Precision,
    steps: u32,
    nx: u32,
) -> DiagnosticResult {
    let lattice = D3Q19::new();
    let collision = Bgk::new();
    let bounce_back = BounceBack;
    let fluid = Fluid;
    let boundaries: Vec<&dyn Boundary3D> = vec![&fluid, &bounce_back];

    let config = SimulationConfig3D {
        nx,
        ny: nx,
        nz: nx,
        init_type: crate::sim::d3::config::InitType::Uniform,
        rho_init: 1.0,
        u_x_init: 0.1,
        u_y_init: 0.05,
        u_z_init: 0.02,
        wgs_x: 8,
        wgs_y: 8,
        wgs_z: 1,
        omega: 1.0 / 0.55,
        force_x: 0.0,
        force_y: 0.0,
        force_z: 0.0,
        periodic_x: false,
        periodic_y: false,
        periodic_z: false,
        pure_fluid: false,
        num_boundary_configs: 1,
        sponge_len: 0,
        sponge_strength: 0.0,
        sponge_cfg: 0,
    };

    let mut lbm = Lbm3D::new(
        ctx.device,
        config,
        precision,
        &lattice,
        &collision,
        &boundaries,
    );

    let mut flags = vec![0u32; (nx * nx * nx) as usize];
    let type_id = bounce_back.type_id();

    for x in 0..nx {
        for y in 0..nx {
            for z in 0..nx {
                if x == 0 || x == nx - 1 || y == 0 || y == nx - 1 || z == 0 || z == nx - 1 {
                    let idx = (x + y * nx + z * nx * nx) as usize;
                    flags[idx] = type_id << 24;
                }
            }
        }
    }

    let bcs = vec![0.0f32; 4];
    lbm.write_buffers(ctx.queue, &flags, &bcs);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    lbm.init(&mut encoder);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    lbm.extract(&mut encoder);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let data_initial = lbm.download_macro_data(ctx.device, ctx.queue);
    let initial_metrics = MacroMetrics::compute(&data_initial);
    let expected_mass = initial_metrics.rho_mean;

    for _ in 0..steps {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        lbm.step(&mut encoder);
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    lbm.extract(&mut encoder);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let data_final = lbm.download_macro_data(ctx.device, ctx.queue);
    let final_metrics = MacroMetrics::compute(&data_final);

    let mass = final_metrics.rho_mean;
    let mut result = DiagnosticResult::pass(&format!(
        "closed_box_{:?}_{}steps_{}^3",
        precision, steps, nx
    ));

    if final_metrics.nan_count > 0 {
        result = DiagnosticResult::fail(
            &result.name,
            &format!("NaNs detected: {}", final_metrics.nan_count),
        );
    }

    let mass_error = (mass - expected_mass).abs() / expected_mass;
    let tolerance = mass_tolerance(precision);
    if mass_error > tolerance {
        result = DiagnosticResult::fail(
            &result.name,
            &format!(
                "Mass is not conserved: expected {}, got {} (relative error {:.3e}, tolerance {:.1e})",
                expected_mass, mass, mass_error, tolerance
            ),
        );
    }

    if final_metrics.max_u > 0.3 {
        result = DiagnosticResult::warn(
            &result.name,
            &format!(
                "Velocity noise in closed box: max_u = {}",
                final_metrics.max_u
            ),
        );
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16s_closed_box_has_a_precision_appropriate_mass_budget() {
        assert_eq!(mass_tolerance(Precision::F32), 1e-4);
        assert_eq!(mass_tolerance(Precision::Auto), 1e-4);
        assert_eq!(mass_tolerance(Precision::FP16S), 5e-4);
    }
}
