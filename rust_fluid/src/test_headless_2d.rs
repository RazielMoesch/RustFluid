use crate::gpu::HeadlessGPU;
use crate::sim::benchmark::{
    export_csv_header, export_csv_row, print_comparison, print_sustained_stats,
    print_workgroup_results, benchmark_workgroup_sizes, BatchTimer, BenchmarkMetadata,
    BenchmarkResult,
};
use crate::sim::lattices::{CollisionLogic, D2Q9, Lattice2D, Precision};
use crate::sim::lbm::LBM2D;
use crate::sim::validation::{
    compare_fields, compute_mass_diagnostics, compute_total_mass, readback_macro_data,
    scan_stability,
};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

// ============================================================
// SIMULATION HYPERPARAMETERS
// ============================================================

const NX: u32 = 1920;
const NY: u32 = 1080;

const WGS_X: u32 = 16;
const WGS_Y: u32 = 16;

const RE: f32 = 250.0;
const INLET_VELOCITY: f32 = 0.02;
const OBSTACLE_BOX_SIZE: u32 = 700;

// Benchmark specific constants
const WARMUP_STEPS: u32 = 100;
const NUM_STEPS: u32 = 10_000;
const BATCH_SIZE: u32 = 1_000;

// ============================================================
// HEADLESS BENCHMARK
// ============================================================

pub fn run() {
    pollster::block_on(run_async());
}

struct RunOutputs {
    result: BenchmarkResult,
    metadata: BenchmarkMetadata,
    macro_data: Vec<f32>,
}

fn run_benchmark(
    gpu: &HeadlessGPU,
    precision: Precision,
) -> RunOutputs {
    let nu_lbm = (INLET_VELOCITY * OBSTACLE_BOX_SIZE as f32) / RE;
    let tau = 3.0 * nu_lbm + 0.5;
    let omega = 1.0 / tau;

    let d2q9 = D2Q9::new()
        .with_collision_logic(CollisionLogic::MRT)
        .with_precision(precision);

    let lattice = Lattice2D::D2Q9(d2q9);
    let compiled = lattice.compile(&gpu.device);

    let mut lbm = LBM2D::new(
        &gpu.device,
        NX,
        NY,
        1.0,
        INLET_VELOCITY,
        0.0,
        WGS_X,
        WGS_Y,
        omega,
        lattice,
        256, // num_boundary_configs
    );

    let domain = crate::setup::SimDomain2D::new()
        .with_w(NX)
        .with_h(NY)
        .with_bc(0, [0.0, 0.0, 1.0, 0.0])
        .with_bc(1, [INLET_VELOCITY, 0.0, 1.0, 0.0])
        .with_edge_type(1, 2, 1)
        .with_edge_type(2, 1, 0)
        .with_edge_type(3, 1, 0)
        .with_edge_type(4, 3, 0);

    gpu.queue.write_buffer(&lbm.buffers.flags, 0, bytemuck::cast_slice(&domain.flags()));
    gpu.queue.write_buffer(
        &lbm.buffers.boundary_configs,
        0,
        bytemuck::cast_slice(&domain.bcs),
    );

    {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Init Encoder"),
        });
        lbm.init(&mut encoder);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    // Warmup
    {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Warmup Encoder"),
        });
        lbm.step_multiple(&mut encoder, WARMUP_STEPS);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    // Benchmark
    let compute_start = Instant::now();
    let chunks = NUM_STEPS / BATCH_SIZE;
    let remainder = NUM_STEPS % BATCH_SIZE;

    let mut timer = BatchTimer::new();

    for _ in 0..chunks {
        timer.start_batch();
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        lbm.step_multiple(&mut encoder, BATCH_SIZE);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        timer.end_batch();
    }

    if remainder > 0 {
        timer.start_batch();
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        lbm.step_multiple(&mut encoder, remainder);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        timer.end_batch();
    }

    let compute_time = compute_start.elapsed();

    // Extract
    {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Extract Encoder"),
        });
        lbm.extract(&mut encoder);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    let total_cells = (NX * NY) as usize;
    let macro_floats = readback_macro_data(&gpu.device, &gpu.queue, &lbm.buffers.macro_data, total_cells);

    let capabilities = gpu.capabilities();

    let meta = BenchmarkMetadata {
        adapter_name: gpu.adapter_name(),
        backend: gpu.backend_name(),
        lattice_name: compiled.lattice_name.to_string(),
        q: compiled.q,
        collision: compiled.collision_name.to_string(),
        precision,
        nx: NX,
        ny: NY,
        wgs_x: WGS_X,
        wgs_y: WGS_Y,
        omega,
        warmup_steps: WARMUP_STEPS,
        benchmark_steps: NUM_STEPS,
        shader_f16: capabilities.shader_f16,
    };

    let res = BenchmarkResult::compute(
        precision,
        compiled.q,
        NX,
        NY,
        NUM_STEPS,
        compute_time.as_secs_f64(),
        timer.durations,
    );

    RunOutputs {
        result: res,
        metadata: meta,
        macro_data: macro_floats,
    }
}

async fn run_async() {
    println!("=== RustFluid — HEADLESS GPU Benchmark ===");

    let gpu = HeadlessGPU::new().await;
    let capabilities = gpu.capabilities();
    
    println!("GPU device acquired (headless): {}", gpu.adapter_name());
    println!("Backend: {}", gpu.backend_name());
    println!("SHADER_F16 support: {}", if capabilities.shader_f16 { "YES" } else { "NO" });

    // --- FP32 benchmark ---
    println!("\n>>> Running FP32 benchmark...");
    let fp32_out = run_benchmark(&gpu, Precision::F32);
    
    println!("{}", fp32_out.metadata);
    println!("{}", fp32_out.result);
    print_sustained_stats(&fp32_out.result, BATCH_SIZE);

    let total_cells = (NX * NY) as usize;

    let fp32_invalid = scan_stability(&fp32_out.macro_data, NX, NY, 1.0);
    if !fp32_invalid.is_empty() {
        println!("  WARNING: FP32 SIMULATION INSTABILITY DETECTED:");
        for inv in fp32_invalid.iter().take(5) {
            println!("{}", inv);
        }
    } else {
        println!("  FP32 Stability scan: OK");
    }

    let initial_mass = (NX * NY) as f64 * 1.0;
    let fp32_final_mass = compute_total_mass(&fp32_out.macro_data, total_cells);
    let fp32_mass_diag = compute_mass_diagnostics(initial_mass, fp32_final_mass);
    println!("\n--- FP32 Mass Conservation ---");
    fp32_mass_diag.print();

    // Export CSV
    let csv_path = "benchmark_results.csv";
    let is_new = !std::path::Path::new(csv_path).exists();
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(csv_path) {
        if is_new {
            let _ = f.write_all(export_csv_header().as_bytes());
        }
        let _ = f.write_all(export_csv_row(&fp32_out.metadata, &fp32_out.result).as_bytes());
    }

    // --- FP16Storage benchmark ---
    if capabilities.shader_f16 {
        println!("\n>>> Running FP16Storage benchmark...");
        let fp16_out = run_benchmark(&gpu, Precision::F16Storage);
        
        println!("{}", fp16_out.result);
        print_sustained_stats(&fp16_out.result, BATCH_SIZE);

        let fp16_invalid = scan_stability(&fp16_out.macro_data, NX, NY, 1.0);
        if !fp16_invalid.is_empty() {
            println!("  WARNING: FP16S SIMULATION INSTABILITY DETECTED:");
            for inv in fp16_invalid.iter().take(5) {
                println!("{}", inv);
            }
        } else {
            println!("  FP16S Stability scan: OK");
        }

        let fp16_final_mass = compute_total_mass(&fp16_out.macro_data, total_cells);
        let fp16_mass_diag = compute_mass_diagnostics(initial_mass, fp16_final_mass);
        println!("\n--- FP16S Mass Conservation ---");
        fp16_mass_diag.print();

        // Compare full fields
        println!("\n============================================");
        println!("           FULL-FIELD COMPARISON");
        println!("============================================");
        let comparison = compare_fields(&fp32_out.macro_data, &fp16_out.macro_data, total_cells);
        comparison.print();
        
        print_comparison(&fp32_out.result, &fp16_out.result);

        if let Ok(mut f) = OpenOptions::new().append(true).open(csv_path) {
            let _ = f.write_all(export_csv_row(&fp16_out.metadata, &fp16_out.result).as_bytes());
        }

    } else {
        println!("\nSkipping FP16Storage benchmark (SHADER_F16 not supported on this device).");
    }

    // --- Workgroup Benchmark ---
    println!("\n>>> Running workgroup size benchmark (FP32, MRT)...");
    let wg_results = benchmark_workgroup_sizes(
        &gpu.device,
        &gpu.queue,
        NX,
        NY,
        Precision::F32,
        CollisionLogic::MRT,
        1.0 / (3.0 * ((INLET_VELOCITY * OBSTACLE_BOX_SIZE as f32) / RE) + 0.5),
        100, // warmup
        1_000, // bench
    );
    print_workgroup_results(&wg_results);
}
