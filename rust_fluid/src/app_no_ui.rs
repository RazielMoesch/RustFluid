use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::gpu::GPU;
use crate::render::{Render2D, RenderMode2D};
use crate::setup::loader::Loader;
use crate::sim::lattices::{D2Q9, Lattice2D};
use crate::sim::lbm::LBM2D;


const RE: f32 = 250.0;              
const STEPS_PER_FRAME: u32 = 50;   

// Lattice resolution (defines the grid size)
const NX: u32 = 1920;
const NY: u32 = 1080;
const WGS_X: u32 = 24;
const WGS_Y: u32 = 24;

// Initial obstacle render box size
const OBSTACLE_BOX_SIZE: u32 = 700;

const X_OFFSET: i32 = -250;
const Y_OFFSET: i32 = 0;

const INLET_VELOCITY: f32 = 0.02;

const RENDER_MAX_VELOCITY: f32 = 0.125;
const RENDER_MIN_VELOCITY: f32 = 0.0;

const RENDER_MAX_CURL: f32 = 0.0125;
const RENDER_MIN_CURL: f32 = 0.00005;

const RENDER_MODE: RenderModeChoice = RenderModeChoice::Curl;

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum RenderModeChoice {
    Velocity,
    Curl,
}

impl RenderModeChoice {
    fn into_mode(self) -> RenderMode2D {
        match self {
            Self::Velocity => RenderMode2D::Velocity,
            Self::Curl => RenderMode2D::Curl,
        }
    }
}


pub fn run(svg_path: Option<&str>) {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App {
        svg_path: svg_path.map(String::from),
        state: None,
    };
    event_loop.run_app(&mut app).expect("Event loop failed");
}

struct App {
    svg_path: Option<String>,
    state: Option<SimState>,
}

struct SimState {
    window: Arc<Window>,
    gpu: GPU,
    lbm: LBM2D,
    renderer: Render2D,
    camera: crate::camera::Camera2D,
    mouse_pressed: bool,
    last_cursor_pos: Option<winit::dpi::PhysicalPosition<f64>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("RustFluid — LBM Simulation")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 640u32));

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );

        let gpu = pollster::block_on(GPU::new(window.clone()));
        let device = &gpu.device;
        let queue = &gpu.queue;

        self.svg_path = Some(String::from("assets/jet.svg"));

        let mut voxel_grid = match &self.svg_path {
            Some(path) => match Loader::load_svg(path, OBSTACLE_BOX_SIZE, OBSTACLE_BOX_SIZE) {
                Ok(grid) => {
                    println!("Loaded SVG '{}' → {}×{} obstacle", path, grid.width, grid.height);
                    grid
                }
                Err(e) => {
                    eprintln!("SVG load failed ({}). Falling back to circle.", e);
                    crate::setup::loader::VoxelGrid2D::new(100, 100, generate_circle(100, 100), std::path::PathBuf::new())
                }
            },
            None => {
                println!("No SVG provided, using circle obstacle.");
                crate::setup::loader::VoxelGrid2D::new(100, 100, generate_circle(100, 100), std::path::PathBuf::new())
            }
        };

        // By default, center it (or place it wherever you'd like using .translate)
        let ox = ((NX as i32) - (voxel_grid.width as i32)) / 2;
        let oy = ((NY as i32) - (voxel_grid.height as i32)) / 2;
        voxel_grid.translate(ox, oy);

        voxel_grid.translate(X_OFFSET, Y_OFFSET);

        let domain = crate::setup::SimDomain2D::new()
            .with_w(NX)
            .with_h(NY)
            .with_bc(0, [0.0, 0.0, 1.0, 0.0])
            .with_bc(1, [INLET_VELOCITY, 0.0, 1.0, 0.0])
            .with_edge_type(1, 2, 1) // left: inlet, bc=1
            .with_edge_type(2, 1, 0) // top: solid, bc=0
            .with_edge_type(3, 1, 0) // bottom: solid, bc=0
            .with_edge_type(4, 3, 0); // right: outlet, bc=0

        let mut flags = domain.flags();
        let shift: u32 = 24;

        for y in 0..NY {
            for x in 0..NX {
                let idx = (x + y * NX) as usize;
                
                let lx = x as i32 - voxel_grid.xpos;
                let ly = y as i32 - voxel_grid.ypos;
                if lx >= 0 && lx < voxel_grid.width as i32 && ly >= 0 && ly < voxel_grid.height as i32 {
                    if voxel_grid.is_solid(lx as u32, ly as u32) {
                        flags[idx] = 1 << shift;
                    }
                }
            }
        }



        let nu_lbm = (INLET_VELOCITY * voxel_grid.width as f32) / RE;
        let tau = 3.0 * nu_lbm + 0.5;
        let omega = 1.0 / tau;

        let lattice = Lattice2D::D2Q9(D2Q9::new()
                                        .with_collision_logic(crate::sim::lattices::CollisionLogic::MRT)
                                        );
        let lbm = LBM2D::new(
            device, NX, NY, 1.0, INLET_VELOCITY, 0.0, WGS_X, WGS_Y, omega, lattice,
        );

        queue.write_buffer(&lbm.buffers.flags, 0, bytemuck::cast_slice(&flags));
        queue.write_buffer(&lbm.buffers.boundary_configs, 0, bytemuck::cast_slice(&domain.bcs));

        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Init Encoder"),
            });
            lbm.init(&mut enc);
            queue.submit(std::iter::once(enc.finish()));
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }

        let (scale_limit, min_threshold) = match RENDER_MODE {
            RenderModeChoice::Velocity => (RENDER_MAX_VELOCITY as f64, RENDER_MIN_VELOCITY as f64),
            RenderModeChoice::Curl => (RENDER_MAX_CURL as f64, RENDER_MIN_CURL as f64),
        };

        let renderer = Render2D::new(
            device,
            gpu.config.format,
            &lbm.buffers.macro_data,
            &lbm.buffers.flags,
            NX,
            NY,
            RENDER_MODE.into_mode(),
            scale_limit,
            min_threshold,
        );

        let camera = crate::camera::Camera2D::new(window.inner_size());

        println!(
            "Simulation ready — {}×{}, ω={:.4}, u_in={}, {} steps/frame",
            NX, NY, omega, INLET_VELOCITY, STEPS_PER_FRAME
        );
        println!(
            "Physical setup — Re: {:.1}, Obstacle: {}×{} lattice units",
            RE, voxel_grid.width, voxel_grid.height
        );

        self.state = Some(SimState {
            window,
            gpu,
            lbm,
            renderer,
            camera,
            mouse_pressed: false,
            last_cursor_pos: None,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                self.state = None;
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                state.gpu.resize(size);
                state.camera.resize(size);
            }

            WindowEvent::MouseInput { state: element_state, button: winit::event::MouseButton::Left, .. } => {
                state.mouse_pressed = element_state == winit::event::ElementState::Pressed;
            }

            WindowEvent::CursorMoved { position, .. } => {
                if state.mouse_pressed {
                    if let Some(last_pos) = state.last_cursor_pos {
                        let dx = (position.x - last_pos.x) as f32;
                        let dy = (position.y - last_pos.y) as f32;
                        state.camera.pan(dx, dy);
                    }
                }
                state.last_cursor_pos = Some(position);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let zoom_amount = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                state.camera.zoom(zoom_amount);
            }

            WindowEvent::RedrawRequested => {
                let device = &state.gpu.device;
                let queue = &state.gpu.queue;

                let frame = match state.gpu.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        let sz = state.window.inner_size();
                        state.gpu.resize(sz);
                        return;
                    }
                    _other => {
                        // eprintln!("Surface error: {_other:?}");
                        return;
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Frame Encoder"),
                    });

                // update camera uniform buffer
                state.renderer.update_camera(queue, state.camera.matrix());

                // compute: step + extract
                state.lbm.step_multiple(&mut encoder, STEPS_PER_FRAME);
                state.lbm.extract(&mut encoder);

                // render pass
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    state.renderer.render(&mut pass);
                }

                queue.submit(std::iter::once(encoder.finish()));
                queue.present(frame);

                state.window.set_title(&format!(
                    "RustFluid — Step {}",
                    state.lbm.step_count
                ));
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


fn generate_circle(w: u32, h: u32) -> Vec<bool> {
    let cx = w / 2;
    let cy = h / 2;
    let r = w.min(h) / 2;
    let r2 = (r * r) as i32;
    let mut data = vec![false; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let dx = x as i32 - cx as i32;
            let dy = y as i32 - cy as i32;
            if dx * dx + dy * dy <= r2 {
                data[(y * w + x) as usize] = true;
            }
        }
    }
    data
}
