use crate::sim::lattices::Precision;
use std::fmt;
use std::time::Instant;

#[derive(Clone)]
pub struct BenchmarkResult {
    pub precision: Precision,
    pub q: u32,
    pub bytes_per_population: u64,
    pub bytes_per_lup: u64,
    pub total_cells: u64,
    pub total_steps: u32,
    pub elapsed_seconds: f64,
    pub steps_per_second: f64,
    pub mlups: f64,
    pub effective_ddf_bandwidth_gbps: f64,
    pub batch_durations: Vec<f64>,
}

impl BenchmarkResult {
    pub fn compute(
        precision: Precision,
        q: u32,
        nx: u32,
        ny: u32,
        total_steps: u32,
        elapsed_seconds: f64,
        batch_durations: Vec<f64>,
    ) -> Self {
        let bytes_per_population = precision.bytes_per_population();
        let bytes_per_lup = 2 * q as u64 * bytes_per_population;
        let total_cells = nx as u64 * ny as u64;
        let steps_per_second = total_steps as f64 / elapsed_seconds;
        let mlups = (total_cells as f64 * total_steps as f64) / elapsed_seconds / 1_000_000.0;
        let effective_ddf_bandwidth_gbps =
            (mlups * 1_000_000.0 * bytes_per_lup as f64) / 1_000_000_000.0;

        Self {
            precision,
            q,
            bytes_per_population,
            bytes_per_lup,
            total_cells,
            total_steps,
            elapsed_seconds,
            steps_per_second,
            mlups,
            effective_ddf_bandwidth_gbps,
            batch_durations,
        }
    }

    pub fn peak_batch_mlups(&self, batch_size: u32) -> Option<f64> {
        self.batch_durations
            .iter()
            .filter(|d| **d > 0.0)
            .map(|d| (self.total_cells as f64 * batch_size as f64) / d / 1_000_000.0)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn median_batch_mlups(&self, batch_size: u32) -> Option<f64> {
        if self.batch_durations.is_empty() {
            return None;
        }
        let mut mlups_vals: Vec<f64> = self
            .batch_durations
            .iter()
            .filter(|d| **d > 0.0)
            .map(|d| (self.total_cells as f64 * batch_size as f64) / d / 1_000_000.0)
            .collect();
        mlups_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = mlups_vals.len() / 2;
        Some(if mlups_vals.len() % 2 == 0 && mlups_vals.len() > 1 {
            (mlups_vals[mid - 1] + mlups_vals[mid]) / 2.0
        } else {
            mlups_vals[mid]
        })
    }

    pub fn sustained_mlups(&self, batch_size: u32) -> Option<f64> {
        if self.batch_durations.is_empty() {
            return None;
        }
        let n = self.batch_durations.len();
        let window_start = n - (n / 5).max(1);
        let window = &self.batch_durations[window_start..];
        let total_time: f64 = window.iter().sum();
        let total_steps = window.len() as f64 * batch_size as f64;
        if total_time > 0.0 {
            Some((self.total_cells as f64 * total_steps) / total_time / 1_000_000.0)
        } else {
            None
        }
    }
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "============================================")?;
        writeln!(f, "  Precision:             {}", self.precision.label())?;
        writeln!(f, "  Population storage:    {} bytes", self.bytes_per_population)?;
        writeln!(f, "  DDF bytes/LUP:         {}", self.bytes_per_lup)?;
        writeln!(f, "--------------------------------------------")?;
        writeln!(f, "  Simulation time:       {:.6} s", self.elapsed_seconds)?;
        writeln!(f, "  Simulation throughput: {:.2} steps/s", self.steps_per_second)?;
        writeln!(f, "  LBM throughput:        {:.2} MLUPS", self.mlups)?;
        writeln!(
            f,
            "  Effective DDF BW:      {:.2} GB/s",
            self.effective_ddf_bandwidth_gbps
        )?;
        write!(f, "============================================")
    }
}

pub struct BenchmarkMetadata {
    pub adapter_name: String,
    pub backend: String,
    pub lattice_name: String,
    pub q: u32,
    pub collision: String,
    pub precision: Precision,
    pub nx: u32,
    pub ny: u32,
    pub wgs_x: u32,
    pub wgs_y: u32,
    pub omega: f32,
    pub warmup_steps: u32,
    pub benchmark_steps: u32,
    pub shader_f16: bool,
}

impl fmt::Display for BenchmarkMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "--- Benchmark Metadata ---")?;
        writeln!(f, "  GPU:               {}", self.adapter_name)?;
        writeln!(f, "  Backend:           {}", self.backend)?;
        writeln!(f, "  Lattice:           {}", self.lattice_name)?;
        writeln!(f, "  Q:                 {}", self.q)?;
        writeln!(f, "  Collision:         {}", self.collision)?;
        writeln!(f, "  Precision:         {}", self.precision.label())?;
        writeln!(f, "  Grid:              {} x {} ({} cells)", self.nx, self.ny, self.nx as u64 * self.ny as u64)?;
        writeln!(f, "  Workgroup:         {} x {}", self.wgs_x, self.wgs_y)?;
        writeln!(f, "  Omega:             {:.6}", self.omega)?;
        writeln!(f, "  Warmup steps:      {}", self.warmup_steps)?;
        writeln!(f, "  Benchmark steps:   {}", self.benchmark_steps)?;
        write!(f, "  SHADER_F16:        {}", if self.shader_f16 { "YES" } else { "NO" })
    }
}

pub fn print_comparison(fp32: &BenchmarkResult, fp16: &BenchmarkResult) {
    println!("\n============================================");
    println!("           PERFORMANCE COMPARISON");
    println!("============================================");
    println!("  FP32 MLUPS:        {:.2}", fp32.mlups);
    println!("  FP16S MLUPS:       {:.2}", fp16.mlups);
    println!("  Speedup:           {:.2}x", fp16.mlups / fp32.mlups);
    println!(
        "  DDF compression:   {:.1}x ({} -> {} bytes/pop)",
        fp32.bytes_per_population as f64 / fp16.bytes_per_population as f64,
        fp32.bytes_per_population,
        fp16.bytes_per_population
    );
    println!("============================================");
}

pub struct BatchTimer {
    batch_start: Instant,
    pub durations: Vec<f64>,
}

impl BatchTimer {
    pub fn new() -> Self {
        Self {
            batch_start: Instant::now(),
            durations: Vec::new(),
        }
    }

    pub fn start_batch(&mut self) {
        self.batch_start = Instant::now();
    }

    pub fn end_batch(&mut self) {
        self.durations.push(self.batch_start.elapsed().as_secs_f64());
    }
}

pub fn print_sustained_stats(result: &BenchmarkResult, batch_size: u32) {
    if let Some(peak) = result.peak_batch_mlups(batch_size) {
        println!("  Peak batch MLUPS:    {:.2}", peak);
    }
    if let Some(median) = result.median_batch_mlups(batch_size) {
        println!("  Median batch MLUPS:  {:.2}", median);
    }
    if let Some(sustained) = result.sustained_mlups(batch_size) {
        println!("  Sustained MLUPS:     {:.2} (last 20%)", sustained);
    }
}

pub fn export_csv_header() -> String {
    "gpu,backend,lattice,q,collision,precision,nx,ny,wg_x,wg_y,steps,runtime,mlups,steps_per_second,bytes_per_population,bytes_per_lup,effective_ddf_bandwidth_gbps\n".to_string()
}

pub fn export_csv_row(meta: &BenchmarkMetadata, result: &BenchmarkResult) -> String {
    format!(
        "{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.2},{:.2},{},{},{:.2}\n",
        meta.adapter_name,
        meta.backend,
        meta.lattice_name,
        result.q,
        meta.collision,
        meta.precision.label(),
        meta.nx,
        meta.ny,
        meta.wgs_x,
        meta.wgs_y,
        result.total_steps,
        result.elapsed_seconds,
        result.mlups,
        result.steps_per_second,
        result.bytes_per_population,
        result.bytes_per_lup,
        result.effective_ddf_bandwidth_gbps,
    )
}

pub struct WorkgroupResult {
    pub wgs_x: u32,
    pub wgs_y: u32,
    pub mlups: f64,
}

pub fn benchmark_workgroup_sizes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    nx: u32,
    ny: u32,
    precision: Precision,
    collision: crate::sim::lattices::CollisionLogic,
    omega: f32,
    warmup_steps: u32,
    bench_steps: u32,
) -> Vec<WorkgroupResult> {
    use crate::sim::lattices::{D2Q9, Lattice2D};
    use crate::sim::lbm::LBM2D;

    let candidates: &[(u32, u32)] = &[
        (8, 8),
        (16, 8),
        (8, 16),
        (16, 16),
        (32, 4),
        (32, 8),
    ];

    let device_limits = device.limits();
    let max_wg = device_limits.max_compute_workgroup_size_x
        .min(device_limits.max_compute_workgroup_size_y);
    let max_invocations = device_limits.max_compute_invocations_per_workgroup;

    let mut results = Vec::new();

    for &(wx, wy) in candidates {
        if wx > max_wg || wy > max_wg || wx * wy > max_invocations {
            continue;
        }

        let lattice = Lattice2D::D2Q9(
            D2Q9::new()
                .with_collision_logic(collision)
                .with_precision(precision),
        );

        let mut lbm = LBM2D::new(device, nx, ny, 1.0, 0.02, 0.0, wx, wy, omega, lattice, 256);

        let domain = crate::setup::SimDomain2D::new()
            .with_w(nx)
            .with_h(ny)
            .with_bc(0, [0.0, 0.0, 1.0, 0.0])
            .with_bc(1, [0.02, 0.0, 1.0, 0.0])
            .with_edge_type(1, 2, 1)
            .with_edge_type(2, 1, 0)
            .with_edge_type(3, 1, 0)
            .with_edge_type(4, 3, 0);

        queue.write_buffer(&lbm.buffers.flags, 0, bytemuck::cast_slice(&domain.flags()));
        queue.write_buffer(
            &lbm.buffers.boundary_configs,
            0,
            bytemuck::cast_slice(&domain.bcs),
        );

        {
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            lbm.init(&mut enc);
            queue.submit(std::iter::once(enc.finish()));
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }

        {
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            lbm.step_multiple(&mut enc, warmup_steps);
            queue.submit(std::iter::once(enc.finish()));
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }

        let start = Instant::now();
        {
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            lbm.step_multiple(&mut enc, bench_steps);
            queue.submit(std::iter::once(enc.finish()));
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64();

        let mlups = (nx as f64 * ny as f64 * bench_steps as f64) / elapsed / 1_000_000.0;
        results.push(WorkgroupResult {
            wgs_x: wx,
            wgs_y: wy,
            mlups,
        });
    }

    results
}

pub fn print_workgroup_results(results: &[WorkgroupResult]) {
    println!("\n--- Workgroup Size Benchmark ---");
    println!("  {:>6} {:>6}  {:>10}", "wgs_x", "wgs_y", "MLUPS");
    let mut best_idx = 0;
    let mut best_mlups = 0.0;
    for (i, r) in results.iter().enumerate() {
        println!("  {:>6} {:>6}  {:>10.2}", r.wgs_x, r.wgs_y, r.mlups);
        if r.mlups > best_mlups {
            best_mlups = r.mlups;
            best_idx = i;
        }
    }
    if !results.is_empty() {
        println!(
            "  Best: {}x{} @ {:.2} MLUPS",
            results[best_idx].wgs_x, results[best_idx].wgs_y, best_mlups
        );
    }
}
