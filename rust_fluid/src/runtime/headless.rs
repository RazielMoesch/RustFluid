//! Builder-driven execution without a window or presentation surface.

use std::time::Instant;
// use crate::gpu::GPU;
use crate::runtime::context::HeadlessContext;
use crate::runtime::simulation::Simulation;

/// Warmup and timed-step counts for a headless run.
pub struct HeadlessConfig {
    pub warmup_steps: u32,
    pub steps: u32,
    pub batch_size: u32,
}

impl Default for HeadlessConfig {
    fn default() -> Self {
        Self {
            warmup_steps: 0,
            steps: 1000,
            batch_size: 100,
        }
    }
}

/// Fluent headless runner parameterized by a simulation factory.
pub struct HeadlessBuilder<S, F>
where
    S: Simulation,
    F: FnOnce(&HeadlessContext) -> S,
{
    config: HeadlessConfig,
    simulation_factory: Option<F>,
}

impl<S, F> HeadlessBuilder<S, F>
where
    S: Simulation,
    F: FnOnce(&HeadlessContext) -> S,
{
    pub fn new() -> Self {
        Self {
            config: HeadlessConfig::default(),
            simulation_factory: None,
        }
    }

    pub fn warmup_steps(mut self, steps: u32) -> Self {
        self.config.warmup_steps = steps;
        self
    }

    pub fn steps(mut self, steps: u32) -> Self {
        self.config.steps = steps;
        self
    }

    pub fn batch_size(mut self, batch_size: u32) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    pub fn simulation(mut self, factory: F) -> Self {
        self.simulation_factory = Some(factory);
        self
    }

    pub fn run(self) -> HeadlessResult {
        let factory = self
            .simulation_factory
            .expect("HeadlessBuilder requires a simulation factory");

        Headless::run_internal(self.config, factory)
    }
}

/// Wall-clock measurements returned after a headless execution.
pub struct HeadlessResult {
    pub total_steps: u32,
    pub elapsed_seconds: f64,
}

/// Entry point for constructing a `HeadlessBuilder`.
pub struct Headless;

impl Headless {
    pub fn builder<S, F>() -> HeadlessBuilder<S, F>
    where
        S: Simulation,
        F: FnOnce(&HeadlessContext) -> S,
    {
        HeadlessBuilder::new()
    }

    fn run_internal<S, F>(config: HeadlessConfig, factory: F) -> HeadlessResult
    where
        S: Simulation,
        F: FnOnce(&HeadlessContext) -> S,
    {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .expect("Failed to get headless adapter");

        let mut required = wgpu::Features::VERTEX_WRITABLE_STORAGE;
        if adapter.features().contains(wgpu::Features::SHADER_F16) {
            required |= wgpu::Features::SHADER_F16;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Headless GPU"),
            required_features: required,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("Failed to get headless device & queue");

        let ctx = HeadlessContext {
            device: &device,
            queue: &queue,
        };

        let mut simulation = factory(&ctx);
        let device = ctx.device;
        let queue = ctx.queue;

        // Warmup
        if config.warmup_steps > 0 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Headless Warmup Encoder"),
            });
            simulation.step(&mut encoder, config.warmup_steps);
            queue.submit(std::iter::once(encoder.finish()));
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }

        let start_time = Instant::now();

        // Main Loop
        let mut steps_remaining = config.steps;
        while steps_remaining > 0 {
            let steps = steps_remaining.min(config.batch_size);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Headless Batch Encoder"),
            });
            simulation.step(&mut encoder, steps);

            queue.submit(std::iter::once(encoder.finish()));
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

            steps_remaining -= steps;
        }

        let elapsed = start_time.elapsed().as_secs_f64();

        HeadlessResult {
            total_steps: config.steps,
            elapsed_seconds: elapsed,
        }
    }
}
