use crate::diagnostics::result::DiagnosticResult;
use crate::runtime::context::HeadlessContext;
use crate::sim::common::precision::Precision;
use crate::sim::d3::boundary::Boundary3D;
use crate::sim::d3::collision::bgk::Bgk;
use crate::sim::d3::config::SimulationConfig3D;
use crate::sim::d3::lattice::d3q19::D3Q19;
use crate::sim::d3::solver::Lbm3D;

pub fn run_parity(ctx: &HeadlessContext, precision: Precision, nx: u32) -> DiagnosticResult {
    let lattice = D3Q19::new();
    let collision = Bgk::new();
    let boundaries: Vec<&dyn Boundary3D> = vec![];

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
        periodic_x: true,
        pure_fluid: true,
        num_boundary_configs: 1,
    };

    let mut lbm1 = Lbm3D::new(ctx.device, config.clone(), precision, &lattice, &collision, &boundaries);
    
    let flags = vec![0u32; (nx * nx * nx) as usize];
    let bcs = vec![0.0f32; 4];
    lbm1.write_buffers(ctx.queue, &flags, &bcs);

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    lbm1.init(&mut encoder);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    for _ in 0..2 {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        lbm1.step(&mut encoder);
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }
    
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    lbm1.extract(&mut encoder);
    ctx.queue.submit(std::iter::once(encoder.finish()));
    let data_1by1 = lbm1.download_macro_data(ctx.device, ctx.queue);

    let mut lbm2 = Lbm3D::new(ctx.device, config, precision, &lattice, &collision, &boundaries);
    lbm2.write_buffers(ctx.queue, &flags, &bcs);
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    lbm2.init(&mut encoder);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    lbm2.step_multiple(&mut encoder, 2);
    ctx.queue.submit(std::iter::once(encoder.finish()));
    
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    lbm2.extract(&mut encoder);
    ctx.queue.submit(std::iter::once(encoder.finish()));
    let data_multi = lbm2.download_macro_data(ctx.device, ctx.queue);

    let mut max_diff = 0.0_f32;
    for (c1, c2) in data_1by1.iter().zip(data_multi.iter()) {
        for i in 0..4 {
            max_diff = max_diff.max((c1[i] - c2[i]).abs());
        }
    }

    if max_diff > 1e-6 {
        DiagnosticResult::fail(
            &format!("parity_{:?}_{}^3", precision, nx),
            &format!("Mismatch between 1x2 and 2x1 steps: max_diff = {}", max_diff)
        )
    } else {
        DiagnosticResult::pass(&format!("parity_{:?}_{}^3", precision, nx))
    }
}