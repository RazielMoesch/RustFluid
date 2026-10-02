//! Headless throughput test for the FluidX3D cylinder case.
//! Call `crate::examples::headless_3d::run()` from your executable.
//!
//! FluidX3D (X,Y,Z) maps to RustFluid (Z,X,Y).
//! All three RustFluid axes are periodic and no outer-face flags are applied,
//! matching the reference topology.

use std::time::Instant;

use crate::sim::common::precision::Precision;
use crate::sim::d3::boundary::Boundary3D;
use crate::sim::d3::boundary::bounce_back::BounceBack;
use crate::sim::d3::boundary::fluid::Fluid;
use crate::sim::d3::collision::bgk::Bgk;
use crate::sim::d3::config::SimulationConfig3D;
use crate::sim::d3::lattice::d3q19::D3Q19;
use crate::sim::d3::solver::Lbm3D;
// use crate::runtime::simulation::Simulation;

const NX: u32 = 768;
const NY: u32 = 192;
const NZ: u32 = 64;

const RE: f32 = 25000.0;
const D: f32 = 64.0;
const U_DESIGN: f32 = 0.577;
// const RHO_INIT: f32 = 1.0;

const WARMUP_STEPS: u32 = 1_000;
const STEPS_PER_SAMPLE: u32 = 1_000;
const SAMPLE_COUNT: usize = 10;
const DISPATCH_BATCH: u32 = 10;

/// Adapter, device, and queue selected for the standalone comparison harness.
pub struct HeadlessGPU {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

impl HeadlessGPU {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("Failed to get headless adapter");

        let mut required = wgpu::Features::VERTEX_WRITABLE_STORAGE;
        if adapter.features().contains(wgpu::Features::SHADER_F16) {
            required |= wgpu::Features::SHADER_F16;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Headless GPU"),
                required_features: required,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("Failed to get headless device & queue");

        Self {
            device,
            queue,
            adapter,
        }
    }
}

fn run_steps(gpu: &HeadlessGPU, lbm: &mut Lbm3D, steps: u32) {
    let mut completed = 0u32;
    while completed < steps {
        let count = (steps - completed).min(DISPATCH_BATCH);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cylinder LBM steps"),
            });
        for _ in 0..count {
            lbm.step(&mut encoder);
        }
        gpu.queue.submit(std::iter::once(encoder.finish()));
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU failed while running cylinder steps");
        completed += count;
    }
}

fn print_cylinder_diagnostics(gpu: &HeadlessGPU, lbm: &mut Lbm3D, flags: &[u32]) {
    let cells = (NX as u64 * NY as u64 * NZ as u64) as usize;
    assert_eq!(flags.len(), cells);
    let size = (cells * std::mem::size_of::<[f32; 4]>()) as u64;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Cylinder macro readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Cylinder diagnostics extract and copy"),
        });
    lbm.extract(&mut encoder);
    encoder.copy_buffer_to_buffer(&lbm.buffers.macro_data, 0, &staging, 0, size);
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("readback receiver dropped");
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU failed during diagnostics readback");
    receiver
        .recv()
        .expect("readback callback missing")
        .expect("readback failed");
    let mapped = slice
        .get_mapped_range()
        .expect("mapped readback range unavailable");
    let macro_data: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
    assert_eq!(macro_data.len(), cells);

    let mut fluid_count = 0usize;
    let mut solid_count = 0usize;
    let mut invalid_count = 0usize;
    let mut negative_density_count = 0usize;
    let (mut rho_sum, mut ux_sum, mut uy_sum, mut uz_sum) = (0.0f64, 0.0, 0.0, 0.0);
    let (mut ux2_sum, mut uy2_sum, mut uz2_sum) = (0.0f64, 0.0f64, 0.0f64);
    let (mut fluid_px, mut fluid_py, mut fluid_pz) = (0.0f64, 0.0f64, 0.0f64);
    let (mut mass_all, mut px_all, mut py_all, mut pz_all) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut rho_min, mut rho_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut speed_max = 0.0f64;

    for (n, &raw_flag) in flags.iter().enumerate() {
        let [all_ux, all_uy, all_uz, all_rho] = macro_data[n];
        if [all_ux, all_uy, all_uz, all_rho]
            .iter()
            .all(|v| v.is_finite())
        {
            let all_rho = all_rho as f64;
            mass_all += all_rho;
            px_all += all_rho * all_ux as f64;
            py_all += all_rho * all_uy as f64;
            pz_all += all_rho * all_uz as f64;
        }
        if raw_flag >> 24 == 1 {
            solid_count += 1;
            continue;
        }
        fluid_count += 1;
        let [ux, uy, uz, rho] = macro_data[n];
        if ![ux, uy, uz, rho].iter().all(|v| v.is_finite()) {
            invalid_count += 1;
            continue;
        }
        let (ux, uy, uz, rho) = (ux as f64, uy as f64, uz as f64, rho as f64);
        if rho < 0.0 {
            negative_density_count += 1;
        }
        rho_sum += rho;
        ux_sum += ux;
        uy_sum += uy;
        uz_sum += uz;
        fluid_px += rho * ux;
        fluid_py += rho * uy;
        fluid_pz += rho * uz;
        ux2_sum += ux * ux;
        uy2_sum += uy * uy;
        uz2_sum += uz * uz;
        rho_min = rho_min.min(rho);
        rho_max = rho_max.max(rho);
        speed_max = speed_max.max((ux * ux + uy * uy + uz * uz).sqrt());
    }

    let valid_count = fluid_count - invalid_count;
    println!(
        "\nCORRECTNESS step={} solid={} fluid={} nonfinite_fluid={} negative_rho={}",
        lbm.step_count, solid_count, fluid_count, invalid_count, negative_density_count
    );
    if valid_count != 0 {
        let inv = 1.0 / valid_count as f64;
        println!(
            "  rho mean={:.9} min={:.9} max={:.9}",
            rho_sum * inv,
            rho_min,
            rho_max
        );
        println!("  mass fluid={:.6} all={:.6}", rho_sum, mass_all);
        println!("  P all=({:.6}, {:.6}, {:.6})", px_all, py_all, pz_all);
        println!(
            "  P fluid=({:.6}, {:.6}, {:.6})",
            fluid_px, fluid_py, fluid_pz
        );
        println!(
            "  mean_u (FluidX x,y,z)=({:.9}, {:.9}, {:.9})",
            uz_sum * inv,
            ux_sum * inv,
            uy_sum * inv
        );
        println!(
            "  mean_u (Rust X,Y,Z)=({:.9}, {:.9}, {:.9})",
            ux_sum * inv,
            uy_sum * inv,
            uz_sum * inv
        );
        println!(
            "  rms_u (Rust X,Y,Z)=({:.9}, {:.9}, {:.9}) max_speed={:.9}",
            (ux2_sum * inv).sqrt(),
            (uy2_sum * inv).sqrt(),
            (uz2_sum * inv).sqrt(),
            speed_max
        );

        let (mut sx2, mut sy2, mut sz2) = (0.0f64, 0.0f64, 0.0f64);
        let (mut sx_max, mut sy_max, mut sz_max) = (0.0f64, 0.0f64, 0.0f64);
        let mut symmetry_count = 0usize;
        for z in 0..NZ {
            for y in 1..=96 {
                let mirror_y = 192 - y;
                for x in 0..NX {
                    let a = (x + y * NX + z * NX * NY) as usize;
                    let b = (x + mirror_y * NX + z * NX * NY) as usize;
                    if flags[a] >> 24 == 1 || flags[b] >> 24 == 1 {
                        continue;
                    }
                    let va = macro_data[a];
                    let vb = macro_data[b];
                    if !va.iter().chain(vb.iter()).all(|v| v.is_finite()) {
                        continue;
                    }
                    let ex = (va[0] - vb[0]).abs() as f64;
                    let ey = (va[1] + vb[1]).abs() as f64;
                    let ez = (va[2] - vb[2]).abs() as f64;
                    sx2 += ex * ex;
                    sy2 += ey * ey;
                    sz2 += ez * ez;
                    sx_max = sx_max.max(ex);
                    sy_max = sy_max.max(ey);
                    sz_max = sz_max.max(ez);
                    symmetry_count += 1;
                }
            }
        }
        let sinv = 1.0 / symmetry_count.max(1) as f64;
        println!(
            "  symmetry rms=({:.3e}, {:.3e}, {:.3e}) max=({:.3e}, {:.3e}, {:.3e}) pairs={}",
            (sx2 * sinv).sqrt(),
            (sy2 * sinv).sqrt(),
            (sz2 * sinv).sqrt(),
            sx_max,
            sy_max,
            sz_max,
            symmetry_count
        );
    }

    const PROBES: [(&str, u32, u32, u32); 6] = [
        ("upstream", 32, 64, 96),
        ("near_upstream", 32, 96, 96),
        ("near_wake", 32, 192, 96),
        ("wake", 32, 256, 96),
        ("far_wake", 32, 400, 96),
        ("cylinder_side", 32, 128, 136),
    ];
    for &(name, fx, fy, fz) in &PROBES {
        let (rx, ry, rz) = (fy, fz, fx);
        let n = (rx + ry * NX + rz * NX * NY) as usize;
        print!(
            "  probe {} FluidX({},{},{}) Rust({},{},{})",
            name, fx, fy, fz, rx, ry, rz
        );
        if flags[n] >> 24 == 1 {
            println!(" SOLID");
        } else {
            let [ux, uy, uz, rho] = macro_data[n];
            println!(
                " rho={:.9} u_Rust(X,Y,Z)=({:.9}, {:.9}, {:.9})",
                rho, ux, uy, uz
            );
        }
    }

    drop(mapped);
    staging.unmap();
}

fn print_box_diagnostics(
    gpu: &HeadlessGPU,
    lbm: &mut Lbm3D,
    nx: u32,
    ny: u32,
    nz: u32,
    force_label: &str,
) {
    let cells = (nx * ny * nz) as usize;
    let size = (cells * std::mem::size_of::<[f32; 4]>()) as u64;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Box macro readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    lbm.extract(&mut encoder);
    encoder.copy_buffer_to_buffer(&lbm.buffers.macro_data, 0, &staging, 0, size);
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let macro_data: &[[f32; 4]] = bytemuck::cast_slice(&mapped);

    let mut rho_sum = 0.0f64;
    let mut ux_sum = 0.0f64;
    for &[ux, _, _, rho] in macro_data {
        rho_sum += rho as f64;
        ux_sum += ux as f64;
    }
    let inv = 1.0 / cells as f64;
    println!(
        "[Box {}] step {}: rho mean={:.9}, ux mean={:.9}",
        force_label,
        lbm.step_count,
        rho_sum * inv,
        ux_sum * inv
    );
}

async fn run_empty_box_tests() {
    println!("\n--- Running Empty Box Tests ---");
    let gpu = HeadlessGPU::new().await;
    let nx = 32;
    let ny = 32;
    let nz = 32;
    let cells = nx * ny * nz;
    let flags = vec![0u32; cells as usize];
    let nu = U_DESIGN * D / RE;
    let omega = 1.0 / (3.0 * nu + 0.5);
    let reference_force = 8.0 * nu * U_DESIGN / (NY as f32 * NY as f32);

    let lattice = D3Q19::new();
    let collision = Bgk::new();
    let boundaries: Vec<&dyn Boundary3D> = vec![&Fluid];

    for precision in [Precision::F32] {
        println!("Precision: {:?}", precision);
        for &(force_label, force_x) in &[
            ("Zero Force", 0.0),
            ("Positive Force", reference_force),
            ("Negative Force", -reference_force),
        ] {
            let config = SimulationConfig3D {
                nx,
                ny,
                nz,
                wgs_x: 32,
                wgs_y: 1,
                wgs_z: 1,
                omega,
                force_x,
                u_x_init: 0.1 * U_DESIGN,
                periodic_x: true,
                periodic_y: true,
                periodic_z: true,
                ..Default::default()
            };

            let mut lbm = Lbm3D::new(
                &gpu.device,
                config,
                precision,
                &lattice,
                &collision,
                &boundaries,
            );
            lbm.write_buffers(&gpu.queue, &flags, &[]);

            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            lbm.init(&mut encoder);
            gpu.queue.submit(std::iter::once(encoder.finish()));
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();

            print_box_diagnostics(&gpu, &mut lbm, nx, ny, nz, force_label);
            run_steps(&gpu, &mut lbm, 11_000);
            print_box_diagnostics(&gpu, &mut lbm, nx, ny, nz, force_label);
        }
    }
    println!("-------------------------------\n");
}

/// Selects an environment-controlled audit or benchmark and runs it to completion.
pub fn run() {
    if std::env::var_os("RUSTFLUID_BOX_AUDIT").is_some() {
        pollster::block_on(run_empty_box_tests());
        return;
    }
    pollster::block_on(run_async());
}

async fn run_async() {
    let quick_bench = std::env::var_os("RUSTFLUID_QUICK_BENCH").is_some();
    let audit_mode = std::env::var_os("RUSTFLUID_AUDIT").is_some()
        || std::env::var_os("RUSTFLUID_QUICK_AUDIT").is_some();
    let gpu = HeadlessGPU::new().await;
    let adapter_info = gpu.adapter.get_info();
    println!("Selected adapter: {}", adapter_info.name);
    println!("  Type: {:?}", adapter_info.device_type);
    println!("  Backend: {:?}", adapter_info.backend);
    println!(
        "  Vendor ID: 0x{:04x}; Device ID: 0x{:04x}",
        adapter_info.vendor, adapter_info.device
    );
    println!(
        "  Driver: {} ({})",
        adapter_info.driver, adapter_info.driver_info
    );

    let nu = U_DESIGN * D / RE;
    let tau = 3.0 * nu + 0.5;
    let omega = 1.0 / tau;
    let u_init = 0.1 * U_DESIGN;
    let force_x = if std::env::var_os("RUSTFLUID_ZERO_FORCE").is_some() {
        0.0
    } else {
        8.0 * nu * U_DESIGN / (NY as f32 * NY as f32)
    };
    let cells = NX as u64 * NY as u64 * NZ as u64;

    let mut flags = vec![0u32; cells as usize];

    for z in 0..NZ {
        for y in 0..NY {
            for x in 0..NX {
                let dx = x as i32 - 128;
                let dy = y as i32 - 96;
                if dx * dx + dy * dy <= 32 * 32 {
                    let index = (x + y * NX + z * NX * NY) as usize;
                    flags[index] = 1u32 << 24; // BounceBack
                }
            }
        }
    }
    assert_eq!(
        flags.iter().filter(|&&flag| flag >> 24 == 1).count(),
        205_376,
        "cylinder voxel count differs from the FluidX3D setup"
    );

    let ex = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0];
    let ey = [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1];
    let ez = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1];
    let mut optimized_flags = flags.clone();
    let disable_interior_optimization = std::env::var_os("RUSTFLUID_NO_INTERIOR_OPT").is_some();
    for z in 1..(NZ - 1) {
        for y in 1..(NY - 1) {
            for x in 1..(NX - 1) {
                let cell_idx = (x + y * NX + z * NX * NY) as usize;
                if flags[cell_idx] >> 24 != 0 {
                    continue;
                }
                let mut all_fluid = true;
                for i in 0..19 {
                    let nx_pos = x as i32 - ex[i];
                    let ny_pos = y as i32 - ey[i];
                    let nz_pos = z as i32 - ez[i];
                    let n_idx =
                        (nx_pos + ny_pos * NX as i32 + nz_pos * NX as i32 * NY as i32) as usize;
                    if flags[n_idx] >> 24 != 0 {
                        all_fluid = false;
                        break;
                    }
                }
                if all_fluid && !disable_interior_optimization {
                    optimized_flags[cell_idx] |= 1 << 23;
                }
            }
        }
    }
    flags = optimized_flags;

    let wgs_candidates = [
        (256, 1, 1),
        (128, 2, 1),
        (64, 4, 1),
        (64, 2, 2),
        (32, 4, 2),
        (16, 16, 1),
        (8, 8, 4),
        (128, 1, 1),
    ];
    let requested_wgs = std::env::var("RUSTFLUID_WGS").ok().and_then(|value| {
        let values: Vec<u32> = value
            .split('x')
            .filter_map(|part| part.parse().ok())
            .collect();
        (values.len() == 3).then(|| (values[0], values[1], values[2]))
    });
    let mut best_wgs = requested_wgs.unwrap_or((64, 2, 2));
    let mut best_mlups = 0.0;
    println!("--- Auto-tuning Workgroup Dimensions ---");

    let lattice = D3Q19::new();
    let collision = Bgk::new();
    let boundaries: Vec<&dyn Boundary3D> = vec![&Fluid, &BounceBack];

    for &(wgs_x, wgs_y, wgs_z) in if audit_mode || quick_bench || requested_wgs.is_some() {
        &wgs_candidates[..0]
    } else {
        &wgs_candidates[..]
    } {
        let config = SimulationConfig3D {
            nx: NX,
            ny: NY,
            nz: NZ,
            wgs_x,
            wgs_y,
            wgs_z,
            omega,
            force_x,
            u_x_init: u_init,
            periodic_x: true,
            periodic_y: true,
            periodic_z: true,
            num_boundary_configs: 2,
            ..Default::default()
        };

        let mut lbm = Lbm3D::new(
            &gpu.device,
            config,
            Precision::FP16S,
            &lattice,
            &collision,
            &boundaries,
        );
        lbm.write_buffers(&gpu.queue, &flags, &[0.0]); // dummy BCS

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cylinder tune initialization"),
            });
        lbm.init(&mut encoder);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        run_steps(&gpu, &mut lbm, 100);
        let start = Instant::now();
        run_steps(&gpu, &mut lbm, 200);
        let seconds = start.elapsed().as_secs_f64();
        let mlups = cells as f64 * 200.0 / seconds / 1.0e6;
        println!("Tested {}x{}x{}: {:.2} MLUPS", wgs_x, wgs_y, wgs_z, mlups);
        if mlups > best_mlups {
            best_mlups = mlups;
            best_wgs = (wgs_x, wgs_y, wgs_z);
        }
    }
    let (wgs_x, wgs_y, wgs_z) = best_wgs;
    println!(
        "--- Auto-tuning Complete. Optimal WGS: {}x{}x{} ---",
        wgs_x, wgs_y, wgs_z
    );

    let config = SimulationConfig3D {
        nx: NX,
        ny: NY,
        nz: NZ,
        wgs_x,
        wgs_y,
        wgs_z,
        omega,
        force_x,
        u_x_init: u_init,
        periodic_x: true,
        periodic_y: true,
        periodic_z: true,
        num_boundary_configs: 2,
        ..Default::default()
    };
    println!("Fully resolved cylinder config: {config:#?}");

    let mut lbm = Lbm3D::new(
        &gpu.device,
        config,
        Precision::FP16S,
        &lattice,
        &collision,
        &boundaries,
    );
    lbm.write_buffers(&gpu.queue, &flags, &[0.0]);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Cylinder LBM initialization"),
        });
    lbm.init(&mut encoder);
    gpu.queue.submit(std::iter::once(encoder.finish()));
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU failed during initialization");

    print_cylinder_diagnostics(&gpu, &mut lbm, &flags);

    if std::env::var_os("RUSTFLUID_QUICK_AUDIT").is_some() {
        run_steps(&gpu, &mut lbm, 100);
        print_cylinder_diagnostics(&gpu, &mut lbm, &flags);
        return;
    }

    if quick_bench {
        const QUICK_WARMUP: u32 = 100;
        const QUICK_SAMPLES: usize = 5;
        const QUICK_STEPS: u32 = 200;
        run_steps(&gpu, &mut lbm, QUICK_WARMUP);
        let mut samples = Vec::with_capacity(QUICK_SAMPLES);
        for sample in 0..QUICK_SAMPLES {
            let start = Instant::now();
            run_steps(&gpu, &mut lbm, QUICK_STEPS);
            let seconds = start.elapsed().as_secs_f64();
            let mlups = cells as f64 * QUICK_STEPS as f64 / seconds / 1.0e6;
            println!(
                "Quick sample {:>2}: {:.3} s, {:.2} MLUPS",
                sample + 1,
                seconds,
                mlups
            );
            samples.push(mlups);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "QUICK_BENCH median={:.2} MLUPS min={:.2} max={:.2} WGS={}x{}x{}",
            samples[QUICK_SAMPLES / 2],
            samples[0],
            samples[QUICK_SAMPLES - 1],
            wgs_x,
            wgs_y,
            wgs_z
        );
        return;
    }

    if audit_mode {
        const CHECKPOINTS: [u32; 15] = [
            1, 2, 5, 10, 20, 50, 100, 250, 500, 1_000, 2_000, 4_000, 6_000, 8_000, 11_000,
        ];
        for checkpoint in CHECKPOINTS {
            let steps = checkpoint - lbm.step_count;
            run_steps(&gpu, &mut lbm, steps);
            print_cylinder_diagnostics(&gpu, &mut lbm, &flags);
        }
        return;
    }

    println!(
        "D3Q19 BGK FP16S; grid={}x{}x{} ({} cells); Re={}; nu={:.6}; tau={:.6}; u_init={:.4}; Fx={:.6e}",
        NX, NY, NZ, cells, RE, nu, tau, u_init, force_x
    );
    println!(
        "Workgroup={}x{}x{}; dispatch={} steps; warmup={} steps; {} samples x {} steps",
        wgs_x, wgs_y, wgs_z, DISPATCH_BATCH, WARMUP_STEPS, SAMPLE_COUNT, STEPS_PER_SAMPLE
    );

    run_steps(&gpu, &mut lbm, WARMUP_STEPS);
    print_cylinder_diagnostics(&gpu, &mut lbm, &flags);

    let mut samples = Vec::with_capacity(SAMPLE_COUNT);
    for sample in 0..SAMPLE_COUNT {
        let start = Instant::now();
        run_steps(&gpu, &mut lbm, STEPS_PER_SAMPLE);
        let seconds = start.elapsed().as_secs_f64();
        let mlups = cells as f64 * STEPS_PER_SAMPLE as f64 / seconds / 1.0e6;
        println!(
            "Sample {:>2}: {:.3} s, {:.2} steps/s, {:.2} MLUPS",
            sample + 1,
            seconds,
            STEPS_PER_SAMPLE as f64 / seconds,
            mlups
        );
        samples.push(mlups);
    }

    samples.sort_by(f64::total_cmp);
    let median = (samples[4] + samples[5]) * 0.5;
    println!(
        "Headless cylinder MLUPS: min={:.2}, median={:.2}, max={:.2}",
        samples[0],
        median,
        samples[SAMPLE_COUNT - 1]
    );
    println!("Final step: {}", lbm.step_count);
    print_cylinder_diagnostics(&gpu, &mut lbm, &flags);
}
