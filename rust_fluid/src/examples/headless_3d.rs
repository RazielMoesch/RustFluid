use crate::runtime::Headless;
use crate::sim::d3::lattice::d3q19::D3Q19;
use crate::sim::d3::collision::bgk::Bgk;
use crate::sim::common::precision::Precision;
use crate::sim::d3::solver::Lbm3D;
use crate::sim::d3::config::SimulationConfig3D;
use crate::sim::d3::boundary::bounce_back::BounceBack;
use crate::sim::d3::boundary::equilibrium_inlet::EquilibriumInlet;
use crate::sim::d3::boundary::fluid::Fluid;
use crate::sim::d3::boundary::free_slip::{FreeSlipX, FreeSlipY, FreeSlipZ};
use crate::sim::d3::boundary::outlet::ZeroGradientOutlet;
use crate::sim::d3::boundary::zou_he::ZouHeLeftVelocity;
use crate::sim::d3::boundary::Boundary3D;

const NX: u32 = 64;
const NY: u32 = 32;
const NZ: u32 = 32;

const WGS_X: u32 = 8;
const WGS_Y: u32 = 4;
const WGS_Z: u32 = 2;

pub fn run() {
    let result = Headless::builder()
        .warmup_steps(10)
        .steps(500)
        .batch_size(100)
        .simulation(|ctx| {
            let u_design = 0.1;
            let omega = 1.8;

            let lattice = D3Q19::new();
            let collision = Bgk::new();
            let boundaries: Vec<&dyn Boundary3D> = vec![
                &Fluid, &BounceBack, &EquilibriumInlet, &ZeroGradientOutlet, &FreeSlipX, &FreeSlipY, &FreeSlipZ, &ZouHeLeftVelocity
            ];

            let config = SimulationConfig3D {
                nx: NX,
                ny: NY,
                nz: NZ,
                init_type: crate::sim::d3::config::InitType::Uniform,
                rho_init: 1.0,
                u_x_init: u_design,
                u_y_init: 0.0,
                u_z_init: 0.0,
                wgs_x: WGS_X,
                wgs_y: WGS_Y,
                wgs_z: WGS_Z,
                omega,
                force_x: 0.0,
                force_y: 0.0,
                force_z: 0.0,
                periodic_x: true,
                pure_fluid: true,
                num_boundary_configs: 1,
            };

            Lbm3D::new(
                ctx.device,
                config,
                Precision::F32,
                &lattice,
                &collision,
                &boundaries,
            )
        })
        .run();
        
    println!("Headless complete: {} steps in {:.3} seconds", result.total_steps, result.elapsed_seconds);
}
