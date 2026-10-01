



use crate::runtime::{Graphics};

use crate::sim::d3::lattice::d3q19::D3Q19;

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
use crate::render::mesh::{Vertex, generate_cylinder_mesh};
use crate::render::DefaultRenderer3D;

const MULTIPLIER: f32 = 1.5;
const NX: u32 = (384.0 * MULTIPLIER) as u32;
const NY: u32 = (128.0 * MULTIPLIER) as u32;
const NZ: u32 = (128.0 * MULTIPLIER) as u32;

const WGS_X: u32 = 32;
const WGS_Y: u32 = 4;
const WGS_Z: u32 = 2;

pub fn run(stl_path: Option<&str>) {
    let reynolds: f32 = 1000.0;
    let d = 64.0;
    let u_design = 1.0 / 3.0_f32.sqrt();

    let nu = (u_design * d) / reynolds;
    let tau = 3.0 * nu + 0.5;
    let omega = 1.0 / tau;
    let force_x = 0.0;

    let stl_path = stl_path.map(|s| s.to_string());

    let mut stl_vertices = Vec::new();
    let mut stl_indices = Vec::new();

    let domain = crate::setup::SimDomain3D::new()
        .with_w(NX)
        .with_h(NY)
        .with_d(NZ)
        .with_edge_type(1, 2, 0) // Inlet on X=0 (type 2)
        .with_edge_type(4, 3, 0) // Outlet on X=W-1 (type 3)
        .with_edge_type(2, 4, 0) // Free slip on Y=0
        .with_edge_type(3, 4, 0) // Free slip on Y=H-1
        .with_edge_type(5, 7, 0) // Free slip on Z=0
        .with_edge_type(6, 7, 0) // Free slip on Z=D-1
        .with_bc(0, [u_design, 0.0, 0.0, 1.0]);

    let mut flags = domain.flags();
    let shift: u32 = 24;

    if let Some(path) = &stl_path {
        println!("Loading STL from {}", path);
        if let Ok(voxel_grid) = crate::setup::loader::Loader::load_stl(path, NX / 2, NY / 2, NZ / 2, 180.0, -90.0, 90.0) {
            let offset_x = 50; // Move closer to inlet (X=0)
            let offset_y = (NY - voxel_grid.height) / 2;
            let offset_z = (NZ - voxel_grid.depth) / 2;


            
            for z in 0..voxel_grid.depth {
                for y in 0..voxel_grid.height {
                    for x in 0..voxel_grid.width {
                        let idx = (x + y * voxel_grid.width + z * voxel_grid.width * voxel_grid.height) as usize;
                        if voxel_grid.data[idx] {
                            let global_x = x + offset_x;
                            let global_y = y + offset_y;
                            let global_z = z + offset_z;
                            let f_idx = (global_x + global_y * NX + global_z * NX * NY) as usize;
                            flags[f_idx] = 1 << shift;
                        }
                    }
                }
            }

            for mut v in voxel_grid.mesh_vertices {
                v[0] += offset_x as f32;
                v[1] += offset_y as f32;
                v[2] += offset_z as f32;
                let cx = (NX as f32 / 2.0) + offset_x as f32 - (voxel_grid.width as f32 / 2.0);
                let cy = (NY as f32 / 2.0) + offset_y as f32 - (voxel_grid.height as f32 / 2.0);
                let cz = (NZ as f32 / 2.0) + offset_z as f32 - (voxel_grid.depth as f32 / 2.0);
                let n = [v[0] - cx, v[1] - cy, v[2] - cz];
                let len = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
                let normal = if len > 0.0 { [n[0]/len, n[1]/len, n[2]/len] } else { [0.0, 1.0, 0.0] };
                stl_vertices.push(Vertex { position: v, normal });
            }
            stl_indices.extend_from_slice(&voxel_grid.mesh_indices);
        }
    } 
    
    if stl_vertices.is_empty() {
        let cx = (NX / 4) as i32;
        let cy = (NY / 2) as i32;
        let r = 12;
        let r2 = r*r;

        for z in 0..NZ {
            for y in 0..NY {
                for x in 0..NX {
                    let dx = x as i32 - cx;
                    let dy = y as i32 - cy;
                    if dx * dx + dy * dy <= r2 {
                        let idx = (x + y * NX + z * NX * NY) as usize;
                        flags[idx] = 1 << shift;
                    }
                }
            }
        }

        let (v, i) = generate_cylinder_mesh(cx as f32, cy as f32, r as f32, NZ as f32, 64);
        stl_vertices.extend(v);
        stl_indices.extend(i);
    }

    println!("Reynolds Number: {}, Kinematic Viscosity: {:.6}, Tau: {:.6}, Force: {:.3e}", reynolds, nu, tau, force_x);

    let ren_vertices = stl_vertices.clone();
    let ren_indices = stl_indices.clone();

    Graphics::builder()
        .title("RustFluid 3D — LBM Simulation")
        .size(1280, 720)
        .steps_per_frame(50)
        .extract_interval(25)
        .simulation(move |ctx| {
            let lattice = D3Q19::new();
            let collision = crate::sim::d3::collision::bgk::Bgk::new();
            let boundaries: Vec<&dyn Boundary3D> = vec![
                &Fluid, &BounceBack, &ZeroGradientOutlet, &EquilibriumInlet, &FreeSlipX, &FreeSlipY, &FreeSlipZ, &ZouHeLeftVelocity
            ];

            let config = SimulationConfig3D {
                nx: NX,
                ny: NY,
                nz: NZ,
                init_type: crate::sim::d3::config::InitType::Uniform,
                rho_init: 0.08,
                u_x_init: u_design,
                u_y_init: 0.0,
                u_z_init: 0.0,
                wgs_x: WGS_X,
                wgs_y: WGS_Y,
                wgs_z: WGS_Z,
                omega,
                force_x,
                force_y: 0.0,
                force_z: 0.0,
                periodic_x: false,
                pure_fluid: false,
                num_boundary_configs: domain.bcs.len().max(1) as u32,
            };

            let lbm = Lbm3D::new(
                ctx.device,
                config,
                Precision::F32,
                &lattice,
                &collision,
                &boundaries,
            );

            ctx.queue.write_buffer(&lbm.buffers.flags, 0, bytemuck::cast_slice(&flags));
            if !domain.bcs.is_empty() {
                ctx.queue.write_buffer(&lbm.buffers.boundary_configs, 0, bytemuck::cast_slice(&domain.bcs));
            }

            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Init Encoder"),
            });
            lbm.init(&mut encoder);
            ctx.queue.submit(std::iter::once(encoder.finish()));

            lbm
        })
        .renderer(move |ctx, lbm| {
            DefaultRenderer3D::new(
                ctx, lbm,
                NX, NY, NZ,
                Some((&ren_vertices, &ren_indices)),
                [50.0, NY as f32 * 0.4, NZ as f32 * 0.25],
                [50.0, NY as f32 * 0.55, NZ as f32 * 0.75],
                [1, 10, 10]
            )
        })
        .run()
        .unwrap();
}
