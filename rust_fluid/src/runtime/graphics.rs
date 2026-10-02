//! Window, surface, event-loop, simulation, and renderer orchestration.

use std::io::{self, IsTerminal, Write};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::gpu::GPU;
use crate::runtime::context::GraphicsContext;
use crate::runtime::renderer::Renderer;
use crate::runtime::simulation::Simulation;

/// Window dimensions, title, and simulation cadence.
pub struct GraphicsConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub steps_per_frame: u32,
    pub extract_interval: u32,
    pub paused: bool,
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            title: "RustFluid".to_string(),
            width: 1280,
            height: 720,
            steps_per_frame: 1,
            extract_interval: 1,
            paused: false,
        }
    }
}

/// Fluent windowed runtime parameterized by simulation and renderer factories.
pub struct GraphicsBuilder<S, R, FS, FR>
where
    S: Simulation + Send + 'static,
    R: Renderer<S> + 'static,
    FS: FnOnce(&GraphicsContext) -> S + Send + 'static,
    FR: FnOnce(&GraphicsContext, &mut S) -> R + 'static,
{
    config: GraphicsConfig,
    simulation_factory: Option<FS>,
    renderer_factory: Option<FR>,
}

impl<S, R, FS, FR> GraphicsBuilder<S, R, FS, FR>
where
    S: Simulation + Send + 'static,
    R: Renderer<S> + 'static,
    FS: FnOnce(&GraphicsContext) -> S + Send + 'static,
    FR: FnOnce(&GraphicsContext, &mut S) -> R + 'static,
{
    pub fn new() -> Self {
        Self {
            config: GraphicsConfig::default(),
            simulation_factory: None,
            renderer_factory: None,
        }
    }

    pub fn title(mut self, title: &str) -> Self {
        self.config.title = title.to_string();
        self
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }

    pub fn steps_per_frame(mut self, steps: u32) -> Self {
        self.config.steps_per_frame = steps;
        self
    }

    pub fn extract_interval(mut self, interval: u32) -> Self {
        self.config.extract_interval = interval;
        self
    }

    pub fn paused(mut self, paused: bool) -> Self {
        self.config.paused = paused;
        self
    }

    pub fn simulation(mut self, factory: FS) -> Self {
        self.simulation_factory = Some(factory);
        self
    }

    pub fn renderer(mut self, factory: FR) -> Self {
        self.renderer_factory = Some(factory);
        self
    }

    pub fn run(self) -> Result<(), winit::error::EventLoopError> {
        let sim_factory = self
            .simulation_factory
            .expect("GraphicsBuilder requires a simulation factory");
        let ren_factory = self
            .renderer_factory
            .expect("GraphicsBuilder requires a renderer factory");

        Graphics::run_internal(self.config, sim_factory, ren_factory)
    }
}

/// Entry point for constructing a `GraphicsBuilder`.
pub struct Graphics;

impl Graphics {
    pub fn builder<S, R, FS, FR>() -> GraphicsBuilder<S, R, FS, FR>
    where
        S: Simulation + Send + 'static,
        R: Renderer<S> + 'static,
        FS: FnOnce(&GraphicsContext) -> S + Send + 'static,
        FR: FnOnce(&GraphicsContext, &mut S) -> R + 'static,
    {
        GraphicsBuilder::new()
    }

    fn run_internal<S, R, FS, FR>(
        config: GraphicsConfig,
        sim_factory: FS,
        ren_factory: FR,
    ) -> Result<(), winit::error::EventLoopError>
    where
        S: Simulation + Send + 'static,
        R: Renderer<S> + 'static,
        FS: FnOnce(&GraphicsContext) -> S + Send + 'static,
        FR: FnOnce(&GraphicsContext, &mut S) -> R + 'static,
    {
        let event_loop = EventLoop::new().expect("Failed to create event loop");

        let mut app = GraphicsApp {
            config,
            sim_factory: Some(sim_factory),
            ren_factory: Some(ren_factory),
            state: None,
        };

        event_loop.run_app(&mut app)
    }
}

struct GraphicsState<S, R> {
    window: Arc<Window>,
    gpu: GPU,
    simulation: Arc<std::sync::Mutex<S>>,
    renderer: R,
    needs_extract: Arc<std::sync::atomic::AtomicBool>,
    paused_flag: Arc<std::sync::atomic::AtomicBool>,
    last_print_time: std::time::Instant,
    last_frame_time: std::time::Instant,
    frames_since_print: u32,
    steps_at_last_print: u64,
    min_frame_time_ms: f32,
    max_frame_time_ms: f32,
    total_frame_time_ms: f32,
    cell_count: u64,
    metrics_in_place: bool,
}

struct GraphicsApp<S, R, FS, FR> {
    config: GraphicsConfig,
    sim_factory: Option<FS>,
    ren_factory: Option<FR>,
    state: Option<GraphicsState<S, R>>,
}

impl<S, R, FS, FR> ApplicationHandler for GraphicsApp<S, R, FS, FR>
where
    S: Simulation + Send + 'static,
    R: Renderer<S> + 'static,
    FS: FnOnce(&GraphicsContext) -> S + Send + 'static,
    FR: FnOnce(&GraphicsContext, &mut S) -> R + 'static,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title(&self.config.title)
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.config.width,
                self.config.height,
            ));

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );

        let gpu = pollster::block_on(GPU::new(window.clone()));

        // Identify the device before anything is allocated, then report the
        // measured VRAM once the simulation and renderer have reserved theirs.
        print_gpu_info(&gpu);

        let mut simulation = {
            let ctx = GraphicsContext {
                device: &gpu.device,
                queue: &gpu.queue,
                surface_format: gpu.config.format,
                window: &window,
            };
            let sim_factory = self.sim_factory.take().unwrap();
            sim_factory(&ctx)
        };

        let renderer = {
            let ctx = GraphicsContext {
                device: &gpu.device,
                queue: &gpu.queue,
                surface_format: gpu.config.format,
                window: &window,
            };
            let ren_factory = self.ren_factory.take().unwrap();
            ren_factory(&ctx, &mut simulation)
        };

        print_vram_summary(&gpu);
        let cell_count = simulation.cell_count();
        println!(
            "Simulation: {} cells ({:.3} MCells) | Steps/frame: {} | Extract interval: {}",
            cell_count,
            cell_count as f64 / 1_000_000.0,
            self.config.steps_per_frame,
            self.config.extract_interval,
        );

        let simulation = Arc::new(std::sync::Mutex::new(simulation));
        let paused_flag = Arc::new(std::sync::atomic::AtomicBool::new(self.config.paused));
        let needs_extract = Arc::new(std::sync::atomic::AtomicBool::new(true));

        let sim_clone = simulation.clone();
        let device_clone = gpu.device.clone();
        let queue_clone = gpu.queue.clone();
        let paused_clone = paused_flag.clone();
        let needs_extract_clone = needs_extract.clone();
        let steps = self.config.steps_per_frame;
        let extract_interval = self.config.extract_interval;

        std::thread::spawn(move || {
            let chunk_size = 4;
            loop {
                if !paused_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    let mut remaining = steps;
                    while remaining > 0 {
                        let current_steps = std::cmp::min(remaining, chunk_size);
                        let mut sim = sim_clone.lock().unwrap();
                        let mut encoder =
                            device_clone.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Sim Encoder Chunk"),
                            });
                        sim.step(&mut encoder, current_steps);
                        queue_clone.submit(std::iter::once(encoder.finish()));
                        drop(sim);

                        // Wait for this chunk to finish so we don't flood the GPU queue
                        // and so the render thread can interleave its submissions.
                        let _ = device_clone.poll(wgpu::PollType::wait_indefinitely());

                        remaining -= current_steps;
                    }

                    let mut sim = sim_clone.lock().unwrap();
                    if sim.step_count() % extract_interval as u64 == 0
                        || sim.step_count() <= steps as u64
                    {
                        let mut encoder =
                            device_clone.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Extract Encoder"),
                            });
                        sim.extract(&mut encoder);
                        needs_extract_clone.store(true, std::sync::atomic::Ordering::Relaxed);
                        queue_clone.submit(std::iter::once(encoder.finish()));
                        drop(sim);
                        let _ = device_clone.poll(wgpu::PollType::wait_indefinitely());
                    }
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(16));
                }
            }
        });

        self.state = Some(GraphicsState {
            window,
            gpu,
            simulation,
            renderer,
            needs_extract,
            paused_flag,
            last_print_time: std::time::Instant::now(),
            last_frame_time: std::time::Instant::now(),
            frames_since_print: 0,
            steps_at_last_print: 0,
            min_frame_time_ms: f32::MAX,
            max_frame_time_ms: 0.0,
            total_frame_time_ms: 0.0,
            cell_count,
            metrics_in_place: io::stdout().is_terminal(),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };

        // Let renderer handle input first
        if state.renderer.input(&event) {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                if state.metrics_in_place {
                    println!();
                }
                self.state = None;
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event: kb, .. }
                if kb.state == winit::event::ElementState::Pressed =>
            {
                if let winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space) =
                    kb.physical_key
                {
                    self.config.paused = !self.config.paused;
                    state
                        .paused_flag
                        .store(self.config.paused, std::sync::atomic::Ordering::Relaxed);
                    println!("Simulation Paused: {}", self.config.paused);
                }
            }
            WindowEvent::Resized(size) => {
                let _lock = state.simulation.lock().unwrap();
                state.gpu.resize(size);
                state
                    .renderer
                    .resize(&state.gpu.device, &state.gpu.queue, size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                let frame = match state.gpu.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        let _lock = state.simulation.lock().unwrap();
                        let sz = state.window.inner_size();
                        state.gpu.resize(sz);
                        state.renderer.resize(
                            &state.gpu.device,
                            &state.gpu.queue,
                            sz.width,
                            sz.height,
                        );
                        return;
                    }
                    _ => return,
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder =
                    state
                        .gpu
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Frame Encoder"),
                        });

                if state
                    .needs_extract
                    .swap(false, std::sync::atomic::Ordering::Relaxed)
                {
                    state.renderer.on_extract();
                }

                let step_count = {
                    let mut sim_lock = state.simulation.lock().unwrap();
                    state.renderer.prepare(
                        &state.gpu.device,
                        &state.gpu.queue,
                        &mut *sim_lock,
                        &mut encoder,
                    );
                    state.renderer.render(&*sim_lock, &mut encoder, &view);
                    sim_lock.step_count()
                };

                state.gpu.queue.submit(std::iter::once(encoder.finish()));
                state.gpu.queue.present(frame);

                state
                    .window
                    .set_title(&format!("{} — Step {}", self.config.title, step_count));

                let now = std::time::Instant::now();
                let frame_time_ms =
                    now.duration_since(state.last_frame_time).as_secs_f32() * 1000.0;
                state.last_frame_time = now;

                state.min_frame_time_ms = state.min_frame_time_ms.min(frame_time_ms);
                state.max_frame_time_ms = state.max_frame_time_ms.max(frame_time_ms);
                state.total_frame_time_ms += frame_time_ms;
                state.frames_since_print += 1;

                let elapsed = now.duration_since(state.last_print_time).as_secs_f32();
                if elapsed >= 1.0 {
                    let fps = state.frames_since_print as f32 / elapsed;
                    let avg_frame_time =
                        state.total_frame_time_ms / state.frames_since_print as f32;
                    let steps_per_sec =
                        (step_count.saturating_sub(state.steps_at_last_print)) as f32 / elapsed;
                    let mlups = steps_per_sec * state.cell_count as f32 / 1_000_000.0;
                    let status = format!(
                        "[Metrics] FPS: {:>5.1} | Frame: {:>5.1}ms ({:>5.1}-{:>5.1}) | TPS: {:>6.1} | MLUPS: {:>8.1} | Step: {}",
                        fps,
                        avg_frame_time,
                        state.min_frame_time_ms,
                        state.max_frame_time_ms,
                        steps_per_sec,
                        mlups,
                        step_count
                    );
                    if state.metrics_in_place {
                        print!("\r\x1b[2K{status}");
                        let _ = io::stdout().flush();
                    } else {
                        println!("{status}");
                    }

                    state.last_print_time = now;
                    state.frames_since_print = 0;
                    state.steps_at_last_print = step_count;
                    state.min_frame_time_ms = f32::MAX;
                    state.max_frame_time_ms = 0.0;
                    state.total_frame_time_ms = 0.0;
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

/// Device and surface identity. Printed before the simulation and renderer
/// allocate anything, so the GPU is identified before the VRAM report below.
fn print_gpu_info(gpu: &GPU) {
    let info = gpu.adapter.get_info();
    println!("GPU: {} ({:?})", info.name, info.device_type);
    println!(
        "Graphics API: {:?} | Driver: {} {}",
        info.backend, info.driver, info.driver_info
    );
    println!(
        "Device IDs: vendor 0x{:04X}, device 0x{:04X} | Shader f16: {}",
        info.vendor,
        info.device,
        gpu.device.features().contains(wgpu::Features::SHADER_F16)
    );
    println!(
        "Surface: {}x{} {:?} | Present mode: {:?}",
        gpu.config.width, gpu.config.height, gpu.config.format, gpu.config.present_mode
    );
    println!(
        "Device limits: max buffer {} | max storage buffer binding {}",
        format_bytes(gpu.adapter.limits().max_buffer_size as u64),
        format_bytes(gpu.adapter.limits().max_storage_buffer_binding_size as u64)
    );
}

/// Measured VRAM. Printed after allocation, so it reflects the real total
/// rather than a prediction.
fn print_vram_summary(gpu: &GPU) {
    if let Some(report) = gpu.device.generate_allocator_report() {
        println!(
            "VRAM (wgpu): {} allocated, {} reserved | {} allocations in {} blocks",
            format_bytes(report.total_allocated_bytes),
            format_bytes(report.total_reserved_bytes),
            report.allocations.len(),
            report.blocks.len()
        );
    } else {
        println!("VRAM (wgpu): allocation reporting unavailable on this graphics backend");
    }
}

pub fn format_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * MIB;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else {
        format!("{} B", bytes)
    }
}
