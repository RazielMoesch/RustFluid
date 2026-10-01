use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::gpu::GPU;
use crate::runtime::context::GraphicsContext;
use crate::runtime::renderer::Renderer;
use crate::runtime::simulation::Simulation;

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

pub struct GraphicsBuilder<S, R, FS, FR>
where
    S: Simulation + 'static,
    R: Renderer<S> + 'static,
    FS: FnOnce(&GraphicsContext) -> S + 'static,
    FR: FnOnce(&GraphicsContext, &mut S) -> R + 'static,
{
    config: GraphicsConfig,
    simulation_factory: Option<FS>,
    renderer_factory: Option<FR>,
}

impl<S, R, FS, FR> GraphicsBuilder<S, R, FS, FR>
where
    S: Simulation + 'static,
    R: Renderer<S> + 'static,
    FS: FnOnce(&GraphicsContext) -> S + 'static,
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

pub struct Graphics;

impl Graphics {
    pub fn builder<S, R, FS, FR>() -> GraphicsBuilder<S, R, FS, FR>
    where
        S: Simulation + 'static,
        R: Renderer<S> + 'static,
        FS: FnOnce(&GraphicsContext) -> S + 'static,
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
        S: Simulation + 'static,
        R: Renderer<S> + 'static,
        FS: FnOnce(&GraphicsContext) -> S + 'static,
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
    simulation: S,
    renderer: R,
}

struct GraphicsApp<S, R, FS, FR> {
    config: GraphicsConfig,
    sim_factory: Option<FS>,
    ren_factory: Option<FR>,
    state: Option<GraphicsState<S, R>>,
}

impl<S, R, FS, FR> ApplicationHandler for GraphicsApp<S, R, FS, FR>
where
    S: Simulation + 'static,
    R: Renderer<S> + 'static,
    FS: FnOnce(&GraphicsContext) -> S + 'static,
    FR: FnOnce(&GraphicsContext, &mut S) -> R + 'static,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title(&self.config.title)
            .with_inner_size(winit::dpi::LogicalSize::new(self.config.width, self.config.height));

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );

        let gpu = pollster::block_on(GPU::new(window.clone()));
        
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

        self.state = Some(GraphicsState {
            window,
            gpu,
            simulation,
            renderer,
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
                self.state = None;
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event: kb, .. } if kb.state == winit::event::ElementState::Pressed => {
                if let winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space) = kb.physical_key {
                    self.config.paused = !self.config.paused;
                    println!("Simulation Paused: {}", self.config.paused);
                }
            }
            WindowEvent::Resized(size) => {
                state.gpu.resize(size);
                state.renderer.resize(&state.gpu.device, &state.gpu.queue, size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                let frame = match state.gpu.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        let sz = state.window.inner_size();
                        state.gpu.resize(sz);
                        state.renderer.resize(&state.gpu.device, &state.gpu.queue, sz.width, sz.height);
                        return;
                    }
                    _ => return,
                };
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder = state.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Frame Encoder"),
                });

                if !self.config.paused {
                    state.simulation.step(&mut encoder, self.config.steps_per_frame);
                    
                    if state.simulation.step_count() % self.config.extract_interval as u64 == 0 || state.simulation.step_count() <= self.config.steps_per_frame as u64 {
                        state.simulation.extract(&mut encoder);
                        state.renderer.on_extract();
                    }
                }

                state.renderer.prepare(&state.gpu.device, &state.gpu.queue, &mut state.simulation, &mut encoder);
                state.renderer.render(&state.simulation, &mut encoder, &view);

                state.gpu.queue.submit(std::iter::once(encoder.finish()));
                state.gpu.queue.present(frame);
                
                state.window.set_title(&format!("{} — Step {}", self.config.title, state.simulation.step_count()));
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
