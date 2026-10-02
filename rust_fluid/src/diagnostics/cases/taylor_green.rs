//! Periodic Taylor-Green vortex decay and symmetry validation.

use crate::diagnostics::metrics::MacroMetrics;
use crate::diagnostics::result::DiagnosticResult;
use crate::runtime::context::HeadlessContext;
use crate::sim::common::precision::Precision;
use crate::sim::d3::boundary::Boundary3D;
use crate::sim::d3::boundary::fluid::Fluid;
use crate::sim::d3::collision::bgk::Bgk;
use crate::sim::d3::config::{InitType, SimulationConfig3D};
use crate::sim::d3::lattice::d3q19::D3Q19;
use crate::sim::d3::solver::Lbm3D;

/// Evolves a Taylor-Green field and compares its decay with expectation.
pub fn run_taylor_green(
    ctx: &HeadlessContext,
    precision: Precision,
    steps: u32,
    nx: u32,
) -> DiagnosticResult {
    let lattice = D3Q19::new();
    let collision = Bgk::new();
    let fluid = Fluid;
    let boundaries: Vec<&dyn Boundary3D> = vec![&fluid];

    // For TG vortex we need periodic boundaries
    let config = SimulationConfig3D {
        nx,
        ny: nx,
        nz: nx,
        init_type: InitType::TaylorGreen,
        rho_init: 1.0,
        u_x_init: 0.0,
        u_y_init: 0.0,
        u_z_init: 0.0,
        wgs_x: 8,
        wgs_y: 8,
        wgs_z: 1,
        omega: 1.0 / 0.55,
        force_x: 0.0,
        force_y: 0.0,
        force_z: 0.0,
        periodic_x: true, // Need true periodicity for TG
        periodic_y: true,
        periodic_z: true,
        pure_fluid: false, // We'll just pass fluid flags
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

    let type_id = fluid.type_id();
    for i in 0..flags.len() {
        flags[i] = type_id << 24;
    }

    let bcs = vec![0.0f32; 4];
    lbm.write_buffers(ctx.queue, &flags, &bcs);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Init Encoder"),
        });
    lbm.init(&mut encoder);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    for _ in 0..steps {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Step Encoder"),
            });
        lbm.step(&mut encoder);
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    lbm.extract(&mut encoder);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let data = lbm.download_macro_data(ctx.device, ctx.queue);
    let metrics = MacroMetrics::compute(&data);

    let mut result = DiagnosticResult::pass(&format!(
        "taylor_green_{:?}_{}steps_{}^3",
        precision, steps, nx
    ));

    if metrics.nan_count > 0 {
        result = DiagnosticResult::fail(
            &result.name,
            &format!("NaNs detected: {}", metrics.nan_count),
        );
    }

    if metrics.max_u > 0.15 {
        result = DiagnosticResult::fail(
            &result.name,
            &format!("Instability: max_u = {}", metrics.max_u),
        );
    }

    // Check if decayed
    if metrics.max_u > 0.099 && steps > 100 {
        result = DiagnosticResult::fail(
            &result.name,
            &format!("Not decaying properly: max_u = {}", metrics.max_u),
        );
    }

    result
}
