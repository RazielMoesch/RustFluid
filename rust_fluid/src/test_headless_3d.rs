//! Headless throughput test for the FluidX3D cylinder case.
//! Call `crate::test_headless_3d_cylinder::run()` from your executable.
//!
//! FluidX3D (X,Y,Z) maps to RustFluid (Z,X,Y).
//! In the supplied WGSL, PERIODIC_X=1 wraps X, Y, and Z, despite its name.
//! With no outer edge flags, this matches the reference's periodic faces.

use crate::gpu::HeadlessGPU;
use crate::sim::lattices::{CollisionLogic, D3Q19, Lattice3D, Precision};
use crate::sim::lbm::LBM3D;
use std::time::Instant;

const NX: u32 = 768;
const NY: u32 = 192;
const NZ: u32 = 64;

const RE: f32 = 200.0;
const D: f32 = 64.0;
const U_DESIGN: f32 = 0.577;
const RHO_INIT: f32 = 1.0;

const WARMUP_STEPS: u32 = 1_000;
const STEPS_PER_SAMPLE: u32 = 1_000;
const SAMPLE_COUNT: usize = 10;

// Match RustFluid's existing 10-step dispatch cadence. Wait once per batch,
// as in the earlier working headless benchmark, to avoid leaving 100 command
// buffers queued before polling. Do not wait after each individual step.
const DISPATCH_BATCH: u32 = 10;

fn run_steps(gpu: &HeadlessGPU, lbm: &mut LBM3D, steps: u32) {
    let mut completed = 0u32;
    while completed < steps {
        let count = (steps - completed).min(DISPATCH_BATCH);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cylinder LBM steps"),
            });
        lbm.step_multiple(&mut encoder, count);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU failed while running cylinder steps");
        completed += count;
    }
}

fn print_cylinder_diagnostics(gpu: &HeadlessGPU, lbm: &mut LBM3D, flags: &[u32]) {
    let cells = (NX as u64 * NY as u64 * NZ as u64) as usize;
    assert_eq!(flags.len(), cells);
    let size = (cells * std::mem::size_of::<[f32; 4]>()) as u64;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Cylinder macro readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // macro_data is only refreshed by extract(); it is not updated each step.
    // The same parity-sensitive extract pipeline is used by the GUI code.
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
    receiver.recv().expect("readback callback missing").expect("readback failed");
    let mapped = slice.get_mapped_range().expect("mapped readback range unavailable");
    let macro_data: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
    assert_eq!(macro_data.len(), cells);

    let mut fluid_count = 0usize;
    let mut solid_count = 0usize;
    let mut invalid_count = 0usize;
    let (mut rho_sum, mut ux_sum, mut uy_sum, mut uz_sum) = (0.0f64, 0.0, 0.0, 0.0);
    let (mut ux2_sum, mut uy2_sum) = (0.0f64, 0.0f64);
    let (mut rho_min, mut rho_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut speed_max = 0.0f64;

    // The DEEP_FLUID optimization sets bit 23; solid type occupies bits 24..31.
    for (n, &raw_flag) in flags.iter().enumerate() {
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
        rho_sum += rho;
        ux_sum += ux;
        uy_sum += uy;
        uz_sum += uz;
        ux2_sum += ux * ux;
        uy2_sum += uy * uy;
        rho_min = rho_min.min(rho);
        rho_max = rho_max.max(rho);
        speed_max = speed_max.max((ux * ux + uy * uy + uz * uz).sqrt());
    }

    let valid_count = fluid_count - invalid_count;
    println!(
        "\nCORRECTNESS step={} solid={} fluid={} nonfinite_fluid={}",
        lbm.step_count, solid_count, fluid_count, invalid_count
    );
    if valid_count != 0 {
        let inv = 1.0 / valid_count as f64;
        println!("  rho mean={:.9} min={:.9} max={:.9}", rho_sum * inv, rho_min, rho_max);
        println!(
            "  mean_u (FluidX x,y,z)=({:.9}, {:.9}, {:.9})",
            uz_sum * inv, ux_sum * inv, uy_sum * inv
        );
        println!(
            "  mean_u (Rust X,Y,Z)=({:.9}, {:.9}, {:.9})",
            ux_sum * inv, uy_sum * inv, uz_sum * inv
        );
        println!(
            "  rms_u (Rust X,Y)=({:.9}, {:.9}) max_speed={:.9}",
            (ux2_sum * inv).sqrt(), (uy2_sum * inv).sqrt(), speed_max
        );
    }

    // FluidX coordinates (x,y,z) map to Rust coordinates (z,x,y).
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

fn print_box_diagnostics(gpu: &HeadlessGPU, lbm: &mut LBM3D, nx: u32, ny: u32, nz: u32, force_label: &str) {
    let cells = (nx * ny * nz) as usize;
    let size = (cells * std::mem::size_of::<[f32; 4]>()) as u64;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Box macro readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    lbm.extract(&mut encoder);
    encoder.copy_buffer_to_buffer(&lbm.buffers.macro_data, 0, &staging, 0, size);
    gpu.queue.submit(std::iter::once(encoder.finish()));
    
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
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
    println!("[Box {}] step {}: rho mean={:.9}, ux mean={:.9}", force_label, lbm.step_count, rho_sum * inv, ux_sum * inv);
}

async fn run_empty_box_tests() {
    println!("\n--- Running Empty Box Tests ---");
    let gpu = HeadlessGPU::new().await;
    let nx = 32; let ny = 32; let nz = 32;
    let cells = nx * ny * nz;
    let flags = vec![0u32; cells as usize];
    let wgs_x = 32; let wgs_y = 1; let wgs_z = 1;
    let omega = 1.0;
    
    // Testing FP32, then FP16S
    for precision in [Precision::F32, Precision::FP16S] {
        println!("Precision: {:?}", precision);
        for &(force_label, force_x) in &[("Zero Force", 0.0), ("Positive Force", 1e-4), ("Negative Force", -1e-4)] {
            let lattice = Lattice3D::D3Q19(
                D3Q19::new()
                    .with_collision_logic(CollisionLogic::BGK)
                    .with_precision(precision),
            );
            let mut lbm = LBM3D::new(
                &gpu.device, nx, ny, nz, 1.0, 0.0, 0.0, 0.0,
                wgs_x, wgs_y, wgs_z, omega, force_x, 0.0, 0.0, 1,
                lattice, 1,
            );
            gpu.queue.write_buffer(&lbm.buffers.flags, 0, bytemuck::cast_slice(&flags));
            
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            lbm.init(&mut encoder);
            gpu.queue.submit(std::iter::once(encoder.finish()));
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            
            print_box_diagnostics(&gpu, &mut lbm, nx, ny, nz, force_label);
            run_steps(&gpu, &mut lbm, 1000);
            print_box_diagnostics(&gpu, &mut lbm, nx, ny, nz, force_label);
        }
    }
    println!("-------------------------------\n");
}

pub fn run() {
    pollster::block_on(run_empty_box_tests());
    pollster::block_on(run_async());
}

async fn run_async() {
    let gpu = HeadlessGPU::new().await;
    let adapter_info = gpu.adapter.get_info();
    println!("Selected adapter: {}", adapter_info.name);
    println!("  Type: {:?}", adapter_info.device_type);
    println!("  Backend: {:?}", adapter_info.backend);
    println!(
        "  Vendor ID: 0x{:04x}; Device ID: 0x{:04x}",
        adapter_info.vendor, adapter_info.device
    );
    println!("  Driver: {} ({})", adapter_info.driver, adapter_info.driver_info);
    assert!(
        gpu.capabilities().shader_f16,
        "FP16S requires SHADER_F16 on this device"
    );

    let nu = U_DESIGN * D / RE;
    let tau = 3.0 * nu + 0.5;
    let omega = 1.0 / tau;
    let u_init = 0.1 * U_DESIGN;
    let force_x = 8.0 * nu * U_DESIGN / (NY as f32 * NY as f32);
    let cells = NX as u64 * NY as u64 * NZ as u64;

    let domain = crate::setup::SimDomain3D::new()
        .with_w(NX)
        .with_h(NY)
        .with_d(NZ);
    // SimDomain3D::flags() also marks all six outer faces solid. FluidX3D's
    // reference case leaves them periodic, so start with fluid everywhere.
    let mut flags = vec![0u32; cells as usize];

    // Cylinder at RustFluid (X=128,Y=96), axis Z, radius 32.
    // FluidX3D's position is (X=32,Y=128,Z=96), axis X.
    for z in 0..NZ {
        for y in 0..NY {
            for x in 0..NX {
                let dx = x as i32 - 128;
                let dy = y as i32 - 96;
                if dx * dx + dy * dy <= 32 * 32 {
                    let index = (x + y * NX + z * NX * NY) as usize;
                    flags[index] = 1u32 << 24; // TYPE_S
                }
            }
        }
    }
    assert_eq!(
        flags.iter().filter(|&&flag| flag >> 24 == 1).count(),
        205_376,
        "cylinder voxel count differs from the FluidX3D setup"
    );

    // Keep the same DEEP_FLUID flag preprocessing as the tuned benchmark.
    let ex = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0];
    let ey = [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1];
    let ez = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1];
    let mut optimized_flags = flags.clone();
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
                    let n_idx = (nx_pos + ny_pos * NX as i32 + nz_pos * NX as i32 * NY as i32) as usize;
                    if flags[n_idx] >> 24 != 0 {
                        all_fluid = false;
                        break;
                    }
                }
                if all_fluid {
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
        (32, 4, 2),
        (16, 16, 1),
        (8, 8, 4),
        (128, 1, 1),
    ];
    let mut best_wgs = (256, 1, 1);
    let mut best_mlups = 0.0;
    println!("--- Auto-tuning Workgroup Dimensions ---");
    for &(wgs_x, wgs_y, wgs_z) in &wgs_candidates {
        let lattice = Lattice3D::D3Q19(
            D3Q19::new()
                .with_collision_logic(CollisionLogic::BGK)
                .with_precision(Precision::FP16S),
        );
        let mut lbm = LBM3D::new(
            &gpu.device, NX, NY, NZ, RHO_INIT, u_init, 0.0, 0.0,
            wgs_x, wgs_y, wgs_z, omega, force_x, 0.0, 0.0, 1,
            lattice, domain.bcs.len().max(1) as u32,
        );
        gpu.queue.write_buffer(&lbm.buffers.flags, 0, bytemuck::cast_slice(&flags));
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Cylinder tune initialization"),
        });
        lbm.init(&mut encoder);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

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
    println!("--- Auto-tuning Complete. Optimal WGS: {}x{}x{} ---", wgs_x, wgs_y, wgs_z);

    let lattice = Lattice3D::D3Q19(
        D3Q19::new()
            .with_collision_logic(CollisionLogic::BGK) // SRT/BGK
            .with_precision(Precision::FP16S),
    );
    let mut lbm = LBM3D::new(
        &gpu.device,
        NX,
        NY,
        NZ,
        RHO_INIT,
        u_init,
        0.0,
        0.0,
        wgs_x,
        wgs_y,
        wgs_z,
        omega,
        force_x,
        0.0,
        0.0,
        1, // shader wraps all three axes when PERIODIC_X == 1
        lattice,
        domain.bcs.len().max(1) as u32,
    );

    gpu.queue
        .write_buffer(&lbm.buffers.flags, 0, bytemuck::cast_slice(&flags));

    // Match FluidX3D's run(0): initialize before the warm-up and timer.
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

    // Check the sign of the initialized flow before any collisions or streaming.
    // This distinguishes an extraction/axis error from later solver drift.
    print_cylinder_diagnostics(&gpu, &mut lbm, &flags);

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
