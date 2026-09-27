use std::path::PathBuf;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::gpu::GPU;
use crate::render::{Render2D, RenderMode2D};
use crate::setup::loader::{Loader, VoxelGrid2D};
use crate::sim::lattices::{D2Q9, Lattice2D};
use crate::sim::lbm::LBM2D;

const RE: f32 = 150.0;
const STEPS_PER_FRAME: u32 = 100;

// Lattice resolution (defines the grid size)
const NX: u32 = 1920;
const NY: u32 = 1080;
const WGS_X: u32 = 16;
const WGS_Y: u32 = 16;

// Initial obstacle render box size
const OBSTACLE_BOX_SIZE: u32 = 500;

const X_OFFSET: i32 = -300;
const Y_OFFSET: i32 = 0;
const ROTATION: f32 = 10.0;

const INLET_VELOCITY: f32 = 0.08;

const RENDER_MAX_VELOCITY: f32 = 0.125;
const RENDER_MIN_VELOCITY: f32 = 0.0;

const RENDER_MAX_CURL: f32 = 0.0125;
const RENDER_MIN_CURL: f32 = 0.0003;

const RENDER_MODE: RenderModeChoice = RenderModeChoice::Velocity;

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
    gpu_setup_time: std::time::Duration,
    setup_time: std::time::Duration,
    sim_start: std::time::Instant,
    printed_stats: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("RustFluid — LBM Simulation")
            .with_inner_size(winit::dpi::LogicalSize::new(NX, NY));

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );

        let gpu_start = std::time::Instant::now();
        let gpu = pollster::block_on(GPU::new(window.clone()));
        let gpu_setup_time = gpu_start.elapsed();

        let device = &gpu.device;
        let queue = &gpu.queue;

        self.svg_path = Some(String::from("assets/jet.svg"));

        let setup_start = std::time::Instant::now();

        let mut voxel_grid = match &self.svg_path {
            Some(path) => match Loader::load_svg(path, OBSTACLE_BOX_SIZE, OBSTACLE_BOX_SIZE, ROTATION) {
                Ok(grid) => {
                    // Example: Rotate the object by 90 degrees
                    // grid.rotate(90.0).unwrap();
                    println!(
                        "Loaded SVG '{}' → {}×{} obstacle",
                        path, grid.width, grid.height
                    );
                    grid
                }
                Err(e) => {
                    eprintln!("SVG load failed ({}). Falling back to circle.", e);
                    // crate::setup::loader::VoxelGrid2D::new(
                    //     100,
                    //     100,
                    //     generate_circle(100, 100),
                    //     std::path::PathBuf::new(),
                    // )
                    VoxelGrid2D::new(0, 0, vec![], PathBuf::new())
                }
            },
            None => {
                println!("No SVG provided, using circle obstacle.");
                crate::setup::loader::VoxelGrid2D::new(
                    100,
                    100,
                    generate_circle(100, 100),
                    std::path::PathBuf::new(),
                )
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
            .with_bc(1, [INLET_VELOCITY, 0.0, 1.0, 0.0]) // bc 1: rightward inlet velocity
            .with_edge_type(1, 2, 1) // left: inlet, bc=1
            .with_edge_type(2, 4, 0) // top: free slip y, bc=0
            .with_edge_type(3, 4, 0) // bottom: free slip y, bc=0
            .with_edge_type(4, 3, 0); // right: outlet, bc=0

        let mut flags = domain.flags();
        let shift: u32 = 24;

        for y in 0..NY {
            for x in 0..NX {
                let idx = (x + y * NX) as usize;

                let lx = x as i32 - voxel_grid.xpos;
                let ly = y as i32 - voxel_grid.ypos;
                if lx >= 0
                    && lx < voxel_grid.width as i32
                    && ly >= 0
                    && ly < voxel_grid.height as i32
                {
                    if voxel_grid.is_solid(lx as u32, ly as u32) {
                        flags[idx] = 1 << shift;
                    }
                }
            }
        }

        let nu_lbm = (INLET_VELOCITY * voxel_grid.width as f32) / RE;
        let tau = 3.0 * nu_lbm + 0.5;
        let omega = 1.0 / tau;

        let lattice = Lattice2D::D2Q9(
            D2Q9::new()
            .with_collision_logic(crate::sim::lattices::CollisionLogic::MRT)
            .with_precision(crate::sim::lattices::Precision::F16Storage)
            ,
        );
        let lbm = LBM2D::new(
            device,
            NX,
            NY,
            1.0,
            INLET_VELOCITY,
            0.0,
            WGS_X,
            WGS_Y,
            omega,
            lattice,
            256,
        );

        queue.write_buffer(&lbm.buffers.flags, 0, bytemuck::cast_slice(&flags));
        queue.write_buffer(
            &lbm.buffers.boundary_configs,
            0,
            bytemuck::cast_slice(&domain.bcs),
        );

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

        let setup_time = setup_start.elapsed();

        self.state = Some(SimState {
            window,
            gpu,
            lbm,
            renderer,
            camera,
            mouse_pressed: false,
            last_cursor_pos: None,
            gpu_setup_time,
            setup_time,
            sim_start: std::time::Instant::now(),
            printed_stats: false,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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

            WindowEvent::MouseInput {
                state: element_state,
                button: winit::event::MouseButton::Left,
                ..
            } => {
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

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

                state
                    .window
                    .set_title(&format!("RustFluid — Step {}", state.lbm.step_count));

                if state.lbm.step_count >= 1000 && !state.printed_stats {
                    state.printed_stats = true;

                    let compute_time = state.sim_start.elapsed();
                    let extract_start = std::time::Instant::now();

                    let total_cells = (NX * NY) as usize;
                    let macro_data_size = (total_cells * 16) as u64;

                    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Readback Buffer"),
                        size: macro_data_size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });

                    let mut readback_encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Readback Encoder"),
                        });

                    readback_encoder.copy_buffer_to_buffer(
                        &state.lbm.buffers.macro_data,
                        0,
                        &readback_buffer,
                        0,
                        macro_data_size,
                    );

                    queue.submit(std::iter::once(readback_encoder.finish()));

                    let buffer_slice = readback_buffer.slice(..);
                    let (sender, receiver) = std::sync::mpsc::channel();
                    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                        sender.send(result).unwrap();
                    });

                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

                    receiver
                        .recv()
                        .unwrap()
                        .expect("Failed to map readback buffer");

                    let data = buffer_slice.get_mapped_range().unwrap();
                    let macro_floats: &[f32] = bytemuck::cast_slice(&data);

                    let extract_time = extract_start.elapsed();

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
                            name, px, py, u_x, u_y, speed, rho
                        );
                    }

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
                        max_speed, max_pos.0, max_pos.1
                    );

                    drop(data);
                    readback_buffer.unmap();

                    let total_time =
                        state.gpu_setup_time + state.setup_time + compute_time + extract_time;
                    let num_steps = state.lbm.step_count;
                    let seconds_per_step = compute_time.as_secs_f64() / num_steps as f64;
                    let steps_per_second = num_steps as f64 / compute_time.as_secs_f64();
                    let mlups = (NX as f64 * NY as f64 * num_steps as f64)
                        / compute_time.as_secs_f64()
                        / 1_000_000.0;

                    println!("\n============================================");
                    println!("            PERFORMANCE SUMMARY WITH RENDERING");
                    println!("============================================");
                    println!(
                        "GPU setup time:        {:.6} s",
                        state.gpu_setup_time.as_secs_f64()
                    );
                    println!(
                        "Initialization time:   {:.6} s",
                        state.setup_time.as_secs_f64()
                    );
                    println!("Simulation time:       {:.6} s", compute_time.as_secs_f64());
                    println!("Extraction/readback:   {:.6} s", extract_time.as_secs_f64());
                    println!("--------------------------------------------");
                    println!("Total execution time:  {:.6} s", total_time.as_secs_f64());
                    println!("--------------------------------------------");
                    println!("Average time per step: {:.6} ms", seconds_per_step * 1000.0);
                    println!("Simulation throughput: {:.2} steps/s", steps_per_second);
                    println!("LBM throughput:        {:.2} MLUPS", mlups);
                    println!("============================================\n");
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
