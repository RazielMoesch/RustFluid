//! Interactive D3Q19 wind tunnel around an STL mesh or fallback cylinder.

use crate::runtime::Graphics;
use crate::runtime::graphics::format_bytes;

use crate::setup::SimDomain3D;
use crate::sim::d3::buffers::SimBuffers3D;
use crate::sim::d3::lattice::Lattice3D;
use crate::sim::d3::lattice::d3q19::D3Q19;

use crate::render::DefaultRenderer3D;
use crate::render::mesh::{Vertex, generate_cylinder_mesh};
use crate::sim::common::precision::Precision;
use crate::sim::d3::boundary::Boundary3D;
use crate::sim::d3::boundary::bounce_back::BounceBack;
use crate::sim::d3::boundary::equilibrium_inlet::EquilibriumInlet;
use crate::sim::d3::boundary::fluid::Fluid;
use crate::sim::d3::boundary::free_slip::{FreeSlipX, FreeSlipY, FreeSlipZ};
use crate::sim::d3::boundary::outlet::ZeroGradientOutlet;
use crate::sim::d3::boundary::zou_he::ZouHeLeftVelocity;
use crate::sim::d3::collision::smagorinsky::Smagorinsky;
use crate::sim::d3::config::SimulationConfig3D;
use crate::sim::d3::solver::Lbm3D;

const VOXELS_PER_DISTANCE_UNIT: f32 = 1.75;

const DOMAIN_LENGTH: f32 = 384.0;
const DOMAIN_HEIGHT: f32 = 128.0;
const DOMAIN_DEPTH: f32 = 128.0;
const CHARACTERISTIC_LENGTH: f32 = 64.0;
const STL_BOX_LENGTH: f32 = DOMAIN_LENGTH * 0.5;
const STL_BOX_HEIGHT: f32 = DOMAIN_HEIGHT * 0.5;
const STL_BOX_DEPTH: f32 = DOMAIN_DEPTH * 0.5;
const STL_INLET_OFFSET: f32 = 15.0;
const FALLBACK_CYLINDER_RADIUS: f32 = 12.0;

const fn voxels_at_resolution(distance: f32, resolution: f32) -> u32 {
    (distance * resolution + 0.5) as u32
}

const fn voxels(distance: f32) -> u32 {
    voxels_at_resolution(distance, VOXELS_PER_DISTANCE_UNIT)
}

const NX: u32 = voxels(DOMAIN_LENGTH);
const NY: u32 = voxels(DOMAIN_HEIGHT);
const NZ: u32 = voxels(DOMAIN_DEPTH);

const WGS_X: u32 = 128;
const WGS_Y: u32 = 2;
const WGS_Z: u32 = 1;

const PRECISION: Precision = Precision::FP16S;

fn print_startup_banner(domain: &SimDomain3D, u_design: f32) {
    let cells = (domain.w as u64) * (domain.h as u64) * (domain.d as u64);
    let sizes = SimBuffers3D::byte_sizes(
        domain.w,
        domain.h,
        domain.d,
        D3Q19::new().q(),
        PRECISION.bytes_per_population(),
        domain.bcs.len().max(1) as u32,
    );

    println!("------------------------------------------------------------");
    println!("RustFluid 3D — STL Wind Tunnel");
    println!("------------------------------------------------------------");
    println!(
        "Domain:     {} x {} x {} = {} cells ({:.3} MCells)",
        domain.w,
        domain.h,
        domain.d,
        cells,
        cells as f64 / 1_000_000.0
    );
    println!(
        "Resolution: {:.3} voxels/distance unit | Physical domain: {:.1} x {:.1} x {:.1}",
        VOXELS_PER_DISTANCE_UNIT, DOMAIN_LENGTH, DOMAIN_HEIGHT, DOMAIN_DEPTH
    );
    println!(
        "Lattice:    D3Q19 ({} populations) | Precision: {} ({} bytes/population)",
        D3Q19::new().q(),
        PRECISION.label(),
        PRECISION.bytes_per_population()
    );
    println!(
        "Free stream: u = {:.4} | Workgroup: {}x{}x{}",
        u_design, WGS_X, WGS_Y, WGS_Z
    );
    println!("VRAM needed (solver buffers):");
    println!(
        "  populations fa    : {:>12}",
        format_bytes(sizes.population)
    );
    println!(
        "  populations fb    : {:>12}",
        format_bytes(sizes.population)
    );
    println!("  geometry flags    : {:>12}", format_bytes(sizes.flags));
    println!(
        "  boundary configs  : {:>12}",
        format_bytes(sizes.boundary_configs)
    );
    println!(
        "  macro data        : {:>12}",
        format_bytes(sizes.macro_data)
    );
    println!("  ----------------------------");
    println!("  solver total      : {:>12}", format_bytes(sizes.total));
    println!(
        "Largest single buffer: {} (must fit the device's max_buffer_size)",
        format_bytes(sizes.population)
    );
    println!(
        "Renderer additionally reserves the Q-criterion 3D texture, marching-cubes \
         vertex buffer and streamline ribbons on top of that."
    );
    println!("------------------------------------------------------------");
}

/// Builds the domain, solver, and default renderer, then enters the event loop.
pub fn run(stl_path: Option<&str>) {
    assert!(
        VOXELS_PER_DISTANCE_UNIT > 0.0,
        "VOXELS_PER_DISTANCE_UNIT must be positive"
    );

    let reynolds: f32 = 2000.0;
    let d = voxels(CHARACTERISTIC_LENGTH) as f32;
    let u_design = 0.1;

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

    print_startup_banner(&domain, u_design);

    let mut flags = domain.flags();
    let shift: u32 = 24;

    if let Some(path) = &stl_path {
        println!("Loading STL from {}", path);
        if let Ok(voxel_grid) = crate::setup::loader::Loader::load_stl(
            path,
            voxels(STL_BOX_LENGTH),
            voxels(STL_BOX_HEIGHT),
            voxels(STL_BOX_DEPTH),
            180.0,
            -90.0,
            90.0,
        ) {
            let offset_x = voxels(STL_INLET_OFFSET);
            let offset_y = (NY - voxel_grid.height) / 2;
            let offset_z = (NZ - voxel_grid.depth) / 2;

            println!(
                "Render mesh: {} vertices, {} triangles (lattice-scale welded)",
                voxel_grid.mesh_vertices.len(),
                voxel_grid.mesh_indices.len() / 3
            );

            for z in 0..voxel_grid.depth {
                for y in 0..voxel_grid.height {
                    for x in 0..voxel_grid.width {
                        let idx =
                            (x + y * voxel_grid.width + z * voxel_grid.width * voxel_grid.height)
                                as usize;
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

            for (i, mut v) in voxel_grid.mesh_vertices.into_iter().enumerate() {
                v[0] += offset_x as f32;
                v[1] += offset_y as f32;
                v[2] += offset_z as f32;
                let normal = voxel_grid.mesh_normals[i];
                stl_vertices.push(Vertex {
                    position: v,
                    normal,
                });
            }
            stl_indices.extend_from_slice(&voxel_grid.mesh_indices);
        }
    }

    if stl_vertices.is_empty() {
        let cx = (NX / 4) as i32;
        let cy = (NY / 2) as i32;
        let r = voxels(FALLBACK_CYLINDER_RADIUS) as i32;
        let r2 = r * r;

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

    println!(
        "Reynolds Number: {}, Kinematic Viscosity: {:.6}, Tau: {:.6}, Force: {:.3e}",
        reynolds, nu, tau, force_x
    );

    let ren_vertices = stl_vertices.clone();
    let ren_indices = stl_indices.clone();

    Graphics::builder()
        .title("RustFluid 3D — LBM Simulation")
        .size(1920, 1080)
        .steps_per_frame(50)
        .extract_interval(50)
        .simulation(move |ctx| {
            let lattice = D3Q19::new();
            let collision = Smagorinsky::new();
            let boundaries: Vec<&dyn Boundary3D> = vec![
                &Fluid,
                &BounceBack,
                &ZeroGradientOutlet,
                &EquilibriumInlet,
                &FreeSlipX,
                &FreeSlipY,
                &FreeSlipZ,
                &ZouHeLeftVelocity,
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
                force_x,
                force_y: 0.0,
                force_z: 0.0,
                periodic_x: false,
                periodic_y: false,
                periodic_z: false,
                pure_fluid: false,
                num_boundary_configs: domain.bcs.len().max(1) as u32,
                sponge_len: NX / 10,
                sponge_strength: 0.01,
                sponge_cfg: 0,
            };

            let lbm = Lbm3D::new(
                ctx.device,
                config,
                PRECISION,
                &lattice,
                &collision,
                &boundaries,
            );

            ctx.queue
                .write_buffer(&lbm.buffers.flags, 0, bytemuck::cast_slice(&flags));
            if !domain.bcs.is_empty() {
                ctx.queue.write_buffer(
                    &lbm.buffers.boundary_configs,
                    0,
                    bytemuck::cast_slice(&domain.bcs),
                );
            }

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Init Encoder"),
                });
            lbm.init(&mut encoder);
            ctx.queue.submit(std::iter::once(encoder.finish()));

            lbm
        })
        .renderer(move |ctx, lbm| {
            DefaultRenderer3D::new(
                ctx,
                lbm,
                NX,
                NY,
                NZ,
                Some((&ren_vertices, &ren_indices)),
                // Seed a two-dimensional plane upstream of the STL.
                [0.0, NY as f32 * 0.425, NZ as f32 * 0.3],
                [0.0, NY as f32 * 0.5, NZ as f32 * 0.7],
                [1, 3, 25],
            )
        })
        .run()
        .unwrap();
}

#[cfg(test)]
mod tests {
    use super::voxels_at_resolution;

    #[test]
    fn resolution_refines_a_fixed_physical_distance() {
        assert_eq!(voxels_at_resolution(100.0, 1.0), 100);
        assert_eq!(voxels_at_resolution(100.0, 2.0), 200);
        assert_eq!(voxels_at_resolution(10.0, 1.5), 15);
    }
}
