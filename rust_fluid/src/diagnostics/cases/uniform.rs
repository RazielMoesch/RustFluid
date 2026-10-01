use crate::diagnostics::metrics::MacroMetrics;
use crate::diagnostics::result::DiagnosticResult;
use crate::runtime::context::HeadlessContext;
use crate::sim::common::precision::Precision;
use crate::sim::d3::boundary::Boundary3D;
use crate::sim::d3::boundary::fluid::Fluid;
use crate::sim::d3::collision::bgk::Bgk;
use crate::sim::d3::config::SimulationConfig3D;
use crate::sim::d3::lattice::d3q19::D3Q19;
use crate::sim::d3::solver::Lbm3D;

pub fn run_uniform(
    ctx: &HeadlessContext,
    precision: Precision,
    steps: u32,
    nx: u32,
) -> DiagnosticResult {
    let lattice = D3Q19::new();
    let collision = Bgk::new();
    let fluid = Fluid;
    let boundaries: Vec<&dyn Boundary3D> = vec![&fluid];

    let config = SimulationConfig3D {
        nx,
        ny: nx,
        nz: nx,
        init_type: crate::sim::d3::config::InitType::Uniform,
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
        periodic_x: true,
        pure_fluid: false,
        num_boundary_configs: 1,
    };

    let mut lbm = Lbm3D::new(ctx.device, config, precision, &lattice, &collision, &boundaries);

    let flags = vec![0u32; (nx * nx * nx) as usize];
    let bcs = vec![0.0f32; 4];
    lbm.write_buffers(ctx.queue, &flags, &bcs);

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Init Encoder"),
    });
    lbm.init(&mut encoder);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    for _ in 0..steps {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Step Encoder"),
        });
        lbm.step(&mut encoder);
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    lbm.extract(&mut encoder);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let data = lbm.download_macro_data(ctx.device, ctx.queue);
    let metrics = MacroMetrics::compute(&data);

    let mut result = DiagnosticResult::pass(&format!("uniform_{:?}_{}steps_{}^3", precision, steps, nx));
    
    if metrics.nan_count > 0 {
        result = DiagnosticResult::fail(&result.name, &format!("NaNs detected: {}", metrics.nan_count));
    }
    
    if (metrics.rho_mean - 1.0).abs() > 1e-5 {
        result = DiagnosticResult::warn(&result.name, &format!("Mass drift detected: mean rho = {}", metrics.rho_mean));
    }

    if metrics.max_u > 1e-5 {
        result = DiagnosticResult::warn(&result.name, &format!("Velocity noise: max_u = {}", metrics.max_u));
    }

    result
}