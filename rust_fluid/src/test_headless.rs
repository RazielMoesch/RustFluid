use crate::gpu::HeadlessGPU;
use crate::sim::lattices::{D2Q9, Lattice2D};
use crate::sim::lbm::LBM2D;
use std::time::Instant;

// ============================================================
// SIMULATION HYPERPARAMETERS
// ============================================================

// Grid dimensions
const NX: u32 = 1920;
const NY: u32 = 1080;

// Initial fluid conditions
const RHO_INIT: f32 = 1.0;
const U_X_INIT: f32 = 0.0;
const U_Y_INIT: f32 = 0.0;

// GPU workgroup dimensions
const WGS_X: u32 = 16;
const WGS_Y: u32 = 16;

// LBM parameters
const OMEGA: f32 = 1.7;
const LID_VELOCITY: f32 = 0.1;

// Number of simulation steps
const NUM_STEPS: u32 = 1000;

// ============================================================
// HEADLESS LID-DRIVEN CAVITY SIMULATION
// ============================================================

/// Hardcoded lid-driven cavity demo — runs entirely headless.
///
/// Grid layout (NX x NY):
///   - Top row (y == NY-1): inlet with rightward velocity
///   - Bottom row (y == 0): solid wall
///   - Left column (x == 0): solid wall
///   - Right column (x == NX-1): solid wall
///   - Interior: fluid
///
pub fn run() {
    pollster::block_on(run_async());
}

async fn run_async() {
    // Start total execution timer
    let total_start = Instant::now();

    println!("=== RustFluid — Lid-Driven Cavity (headless) ===");

    println!("\n--- Simulation Hyperparameters ---");
    println!("Grid:             {} x {}", NX, NY);
    println!("Initial density:  {}", RHO_INIT);
    println!("Initial velocity: ({}, {})", U_X_INIT, U_Y_INIT);
    println!("Workgroup size:   {} x {}", WGS_X, WGS_Y);
    println!("Omega:            {}", OMEGA);
    println!("Lid velocity:     {}", LID_VELOCITY);
    println!("Simulation steps: {}", NUM_STEPS);

    // ============================================================
    // GPU SETUP
    // ============================================================

    let gpu_start = Instant::now();

    let gpu = HeadlessGPU::new().await;
    let device = &gpu.device;
    let queue = &gpu.queue;

    let gpu_setup_time = gpu_start.elapsed();

    println!("\nGPU device acquired (headless).");

    // ============================================================
    // LATTICE AND SOLVER
    // ============================================================

    let setup_start = Instant::now();

    let lattice = Lattice2D::D2Q9(D2Q9::new());

    let mut lbm = LBM2D::new(
        device,
        NX,
        NY,
        RHO_INIT,
        U_X_INIT,
        U_Y_INIT,
        WGS_X,
        WGS_Y,
        OMEGA,
        lattice,
    );

    // ============================================================
    // GEOMETRY AND BOUNDARY CONDITIONS
    // ============================================================

    let total_cells = (NX * NY) as usize;

    let mut flags = vec![0u32; total_cells];

    let flag_type_shift: u32 = 24;

    for y in 0..NY {
        for x in 0..NX {
            let idx = (x + y * NX) as usize;

            if y == NY - 1 {
                // Top lid — inlet (type 2), config ID 1
                flags[idx] = (2u32 << flag_type_shift) | 1u32;
            } else if y == 0 || x == 0 || x == NX - 1 {
                // Walls — solid (type 1), config ID 0
                flags[idx] = 1u32 << flag_type_shift;
            }
        }
    }

    // Boundary configurations:
    // Each configuration contains 4 floats:
    // [vel.x, vel.y, density, padding]

    let boundary_configs: Vec<f32> = {
        let mut cfgs = vec![0.0f32; 256 * 4];

        // Config 0: solid walls
        cfgs[0] = 0.0;
        cfgs[1] = 0.0;
        cfgs[2] = 1.0;
        cfgs[3] = 0.0;

        // Config 1: moving lid
        cfgs[4] = LID_VELOCITY;
        cfgs[5] = 0.0;
        cfgs[6] = 1.0;
        cfgs[7] = 0.0;

        cfgs
    };

    // Upload geometry flags
    queue.write_buffer(
        &lbm.buffers.flags,
        0,
        bytemuck::cast_slice(&flags),
    );

    // Upload boundary configurations
    queue.write_buffer(
        &lbm.buffers.boundary_configs,
        0,
        bytemuck::cast_slice(&boundary_configs),
    );

    // ============================================================
    // INITIALIZE EQUILIBRIUM POPULATIONS
    // ============================================================

    {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Init Encoder"),
            });

        lbm.init(&mut encoder);

        queue.submit(std::iter::once(encoder.finish()));

        // Wait for GPU initialization to finish
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }

    let setup_time = setup_start.elapsed();

    println!("Initialization complete.");

    // ============================================================
    // RUN SIMULATION — COMPUTE TIMING
    // ============================================================

    println!("\nRunning {} LBM steps...", NUM_STEPS);

    let compute_start = Instant::now();

    {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Step Encoder"),
            });

        lbm.step_multiple(&mut encoder, NUM_STEPS);

        queue.submit(std::iter::once(encoder.finish()));

        // IMPORTANT:
        // Wait for all submitted GPU commands to complete.
        // Without this synchronization, the timer would mainly
        // measure CPU command submission rather than GPU execution.
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }

    let compute_time = compute_start.elapsed();

    println!("Ran {} LBM steps.", NUM_STEPS);

    // ============================================================
    // EXTRACT MACROSCOPIC QUANTITIES
    // ============================================================

    let extract_start = Instant::now();

    {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Extract Encoder"),
            });

        lbm.extract(&mut encoder);

        queue.submit(std::iter::once(encoder.finish()));

        device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }

    // ============================================================
    // GPU READBACK
    // ============================================================

    let macro_data_size = (total_cells * 16) as u64;

    let readback_buffer =
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: macro_data_size,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

    {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &lbm.buffers.macro_data,
            0,
            &readback_buffer,
            0,
            macro_data_size,
        );

        queue.submit(std::iter::once(encoder.finish()));
    }

    // Map GPU buffer to CPU memory
    let buffer_slice = readback_buffer.slice(..);

    let (sender, receiver) = std::sync::mpsc::channel();

    buffer_slice.map_async(
        wgpu::MapMode::Read,
        move |result| {
            sender.send(result).unwrap();
        },
    );

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    receiver
        .recv()
        .unwrap()
        .expect("Failed to map readback buffer");

    let data = buffer_slice.get_mapped_range().unwrap();

    let macro_floats: &[f32] = bytemuck::cast_slice(&data);

    let extract_time = extract_start.elapsed();

    // ============================================================
    // PRINT VELOCITY PROBES
    // ============================================================

    println!("\n--- Velocity probes (u_x, u_y, speed, rho) ---");

    let probes = [
        ("Center", NX / 2, NY / 2),
        ("Top-center (lid)", NX / 2, NY - 2),
        ("Bottom-center", NX / 2, 1),
        ("Left-center", 1, NY / 2),
        ("Right-center", NX - 2, NY / 2),
        ("Quarter", NX / 4, NY / 4),
        ("Three-quarter", 3 * NX / 4, 3 * NY / 4),
    ];

    for (name, px, py) in &probes {
        let idx = (*px + *py * NX) as usize;

        let base = idx * 4;

        let u_x = macro_floats[base];
        let u_y = macro_floats[base + 1];
        let rho = macro_floats[base + 3];

        let speed = (u_x * u_x + u_y * u_y).sqrt();

        println!(
            "  {:<20} ({:3},{:3}): u=({:+.6}, {:+.6}), |u|={:.6}, rho={:.6}",
            name,
            px,
            py,
            u_x,
            u_y,
            speed,
            rho
        );
    }

    // ============================================================
    // COMPUTE MAXIMUM VELOCITY
    // ============================================================

    let mut max_speed: f32 = 0.0;
    let mut max_pos = (0u32, 0u32);

    for y in 0..NY {
        for x in 0..NX {
            let idx = (x + y * NX) as usize;
            let base = idx * 4;

            let u_x = macro_floats[base];
            let u_y = macro_floats[base + 1];

            let speed = (u_x * u_x + u_y * u_y).sqrt();

            if speed > max_speed {
                max_speed = speed;
                max_pos = (x, y);
            }
        }
    }

    println!(
        "\n  Max speed: {:.6} at ({}, {})",
        max_speed,
        max_pos.0,
        max_pos.1
    );

    drop(data);
    readback_buffer.unmap();

    // ============================================================
    // PERFORMANCE SUMMARY
    // ============================================================

    let total_time = total_start.elapsed();

    let seconds_per_step = compute_time.as_secs_f64()
        / NUM_STEPS as f64;

    let steps_per_second = NUM_STEPS as f64
        / compute_time.as_secs_f64();

    let mlups = (NX as f64 * NY as f64 * NUM_STEPS as f64)
        / compute_time.as_secs_f64()
        / 1_000_000.0;

    println!("\n============================================");
    println!("            PERFORMANCE SUMMARY");
    println!("============================================");

    println!(
        "GPU setup time:        {:.6} s",
        gpu_setup_time.as_secs_f64()
    );

    println!(
        "Initialization time:   {:.6} s",
        setup_time.as_secs_f64()
    );

    println!(
        "Simulation time:       {:.6} s",
        compute_time.as_secs_f64()
    );

    println!(
        "Extraction/readback:   {:.6} s",
        extract_time.as_secs_f64()
    );

    println!("--------------------------------------------");

    println!(
        "Total execution time:  {:.6} s",
        total_time.as_secs_f64()
    );

    println!("--------------------------------------------");

    println!(
        "Average time per step: {:.6} ms",
        seconds_per_step * 1000.0
    );

    println!(
        "Simulation throughput: {:.2} steps/s",
        steps_per_second
    );

    println!(
        "LBM throughput:        {:.2} MLUPS",
        mlups
    );

    println!("============================================");

    println!("\n=== Done ===");
}