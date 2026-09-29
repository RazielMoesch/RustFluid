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
const WGS_X: u32 = 32;
const WGS_Y: u32 = 4;
const WGS_Z: u32 = 2;

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

pub fn run() {
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
        "F16Storage requires SHADER_F16 on this device"
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
    let mut flags = domain.flags();

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

    let lattice = Lattice3D::D3Q19(
        D3Q19::new()
            .with_collision_logic(CollisionLogic::BGK) // SRT/BGK
            .with_precision(Precision::F16Storage),
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
        WGS_X,
        WGS_Y,
        WGS_Z,
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

    println!(
        "D3Q19 BGK F16Storage; grid={}x{}x{} ({} cells); Re={}; nu={:.6}; tau={:.6}; u_init={:.4}; Fx={:.6e}",
        NX, NY, NZ, cells, RE, nu, tau, u_init, force_x
    );
    println!(
        "Workgroup={}x{}x{}; dispatch={} steps; warmup={} steps; {} samples x {} steps",
        WGS_X, WGS_Y, WGS_Z, DISPATCH_BATCH, WARMUP_STEPS, SAMPLE_COUNT, STEPS_PER_SAMPLE
    );

    run_steps(&gpu, &mut lbm, WARMUP_STEPS);

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
}
