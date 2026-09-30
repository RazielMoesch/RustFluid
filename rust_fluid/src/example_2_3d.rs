use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::gpu::GPU;
use crate::render::{Render3D, RenderMode3D};
use crate::sim::lattices::{CollisionLogic, D3Q19, Lattice3D, Precision};
use crate::sim::lbm::LBM3D;

const STEPS_PER_FRAME: u32 = 2; // Number of LBM steps to execute per rendered frame
const EXTRACT_INTERVAL: u32 = 50; // Only extract macro_data and trace streamlines every N simulation steps

// Lattice resolution (defines the grid size)
const NX: u32 = 384;
const NY: u32 = 64;
const NZ: u32 = 64;

const WGS_X: u32 = 32;
const WGS_Y: u32 = 4;
const WGS_Z: u32 = 2;



const RENDER_MAX_VELOCITY: f32 = 1.0;
const RENDER_MIN_VELOCITY: f32 = 0.00000;

pub fn run(stl_path: Option<&str>) {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App {
        state: None,
        stl_path: stl_path.map(|s| s.to_string()),
    };
    event_loop.run_app(&mut app).expect("Event loop failed");
}

struct App {
    state: Option<SimState>,
    stl_path: Option<String>,
}

struct SimState {
    window: Arc<Window>,
    gpu: GPU,
    lbm: LBM3D,
    renderer: Render3D,
    mesh_renderer: crate::render::mesh::MeshRenderer,
    camera: crate::camera::Camera3D,
    mouse_pressed: bool,
    right_mouse_pressed: bool,
    last_cursor_pos: Option<winit::dpi::PhysicalPosition<f64>>,
    iso_q: f32,
    max_speed: f32,
    streamline_renderer: crate::render::streamlines::StreamlineRenderer,
    render_mode: crate::render::RenderMode3D,
    gpu_setup_time: std::time::Duration,
    setup_time: std::time::Duration,
    sim_start: std::time::Instant,
    printed_stats: bool,
    paused: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("RustFluid 3D — LBM Simulation")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );

        let t0 = std::time::Instant::now();
        let gpu = pollster::block_on(GPU::new(window.clone()));
        let gpu_setup_time = t0.elapsed();
        let adapter_info = gpu.adapter.get_info();
        println!("Selected adapter: {}", adapter_info.name);
        println!("  Type: {:?}", adapter_info.device_type);
        println!("  Backend: {:?}", adapter_info.backend);

        let t1 = std::time::Instant::now();
        let device = &gpu.device;
        let queue = &gpu.queue;

        let reynolds: f32 = 100.0;
        let d = 16.0;
        let u_design = 0.577;

        // 3D Domain Setup: Wind Tunnel
        let domain = crate::setup::SimDomain3D::new()
            .with_w(NX)
            .with_h(NY)
            .with_d(NZ)
            .with_edge_type(1, 2, 0) // Inlet on X=0 (type 2)
            .with_edge_type(4, 3, 0) // Outlet on X=W-1 (type 3)
            .with_edge_type(2, 4, 0) // Free slip on Y=0
            .with_edge_type(3, 4, 0) // Free slip on Y=H-1
            .with_edge_type(5, 7, 0) // Free slip on Z=0
            .with_edge_type(6, 7, 0) // Free slip on Z=D-1
            .with_bc(0, [u_design, 0.0, 0.0, 1.0]);

        let mut flags = domain.flags();
        let shift: u32 = 24;

        let mut stl_vertices = Vec::new();
        let mut stl_indices = Vec::new();

        if let Some(path) = &self.stl_path {
            println!("Loading STL from {}", path);
            match crate::setup::loader::Loader::load_stl(path, NX / 2, NY / 2, NZ / 2, 180.0, 90.0, -90.0) {
                Ok(voxel_grid) => {
                    let offset_x = (NX - voxel_grid.width) - 25; // Move closer to inlet
                    let offset_y = (NY - voxel_grid.height) / 2;
                    let offset_z = (NZ - voxel_grid.depth) / 2;
                    
                    for z in 0..voxel_grid.depth {
                        for y in 0..voxel_grid.height {
                            for x in 0..voxel_grid.width {
                                let idx = (x + y * voxel_grid.width + z * voxel_grid.width * voxel_grid.height) as usize;
                                if voxel_grid.data[idx] {
                                    let global_x = x + offset_x;
                                    let global_y = y + offset_y;
                                    let global_z = z + offset_z;
                                    let f_idx = (global_x + global_y * NX + global_z * NX * NY) as usize;
                                    flags[f_idx] = 1 << shift;
                                }
                            }
                        }
                    }

                    // For the mesh, calculate vertex normals because STL might not have smooth ones, or just use the generated ones.
                    // Let's just use what Loader returned, but translate it to the LBM domain!
                    for mut v in voxel_grid.mesh_vertices {
                        v[0] += offset_x as f32;
                        v[1] += offset_y as f32;
                        v[2] += offset_z as f32;
                        // Calculate a basic outward normal relative to center
                        let cx = (NX as f32 / 2.0) + offset_x as f32 - (voxel_grid.width as f32 / 2.0);
                        let cy = (NY as f32 / 2.0) + offset_y as f32 - (voxel_grid.height as f32 / 2.0);
                        let cz = (NZ as f32 / 2.0) + offset_z as f32 - (voxel_grid.depth as f32 / 2.0);
                        let n = [v[0] - cx, v[1] - cy, v[2] - cz];
                        let len = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
                        let normal = if len > 0.0 { [n[0]/len, n[1]/len, n[2]/len] } else { [0.0, 1.0, 0.0] };
                        stl_vertices.push(crate::render::mesh::Vertex { position: v, normal });
                    }
                    stl_indices = voxel_grid.mesh_indices;
                }
                Err(e) => {
                    println!("Failed to load STL: {}", e);
                }
            }
        } 
        
        if stl_vertices.is_empty() {
            // Fallback: Generate a cylinder fitted to the new domain
            let cx = (NX / 4) as i32; // Move to the left side so wake flows to the right
            let cy = (NY / 2) as i32; // middle of the Y height
            let r = 12;               // D = 24
            let r2 = r*r;

            for z in 0..NZ {
                for y in 0..NY {
                    for x in 0..NX {
                        let dx = x as i32 - cx;
                        let dy = y as i32 - cy;
                        // Cylinder along Z
                        if dx * dx + dy * dy <= r2 {
                            let idx = (x + y * NX + z * NX * NY) as usize;
                            flags[idx] = 1 << shift;
                        }
                    }
                }
            }

            let (v, i) = crate::render::mesh::generate_cylinder_mesh(cx as f32, cy as f32, r as f32, NZ as f32, 64);
            stl_vertices = v;
            stl_indices = i;
        }

        // FluidX3D uses a Smagorinsky subgrid-scale model to stabilize at Re=25000.
        // As per instructions, calculate viscosity from design speed 0.577.
        let nu = (u_design * d) / reynolds;
        let tau = 3.0 * nu + 0.5;
        let omega = 1.0 / tau;
        
        let u_init = u_design;
        let force_x = 0.0;

        println!("Reynolds Number: {}, Kinematic Viscosity: {:.6}, Tau: {:.6}, Force: {:.3e}", reynolds, nu, tau, force_x);

        let lattice = Lattice3D::D3Q19(
            D3Q19::new()
                .with_collision_logic(CollisionLogic::BGK)
                .with_precision(Precision::F32),
        );
        
        let boundary_config_count = domain.bcs.len().max(1) as u32;

        let lbm = LBM3D::new(
            device,
            NX,
            NY,
            NZ,
            1.0,
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
            0, // periodic_x
            lattice,
            boundary_config_count,
        );

        queue.write_buffer(&lbm.buffers.flags, 0, bytemuck::cast_slice(&flags));
        
        if !domain.bcs.is_empty() {
            queue.write_buffer(
                &lbm.buffers.boundary_configs,
                0,
                bytemuck::cast_slice(&domain.bcs),
            );
        }

        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Init Encoder"),
            });
            lbm.init(&mut enc);
            queue.submit(std::iter::once(enc.finish()));
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }

        let mut renderer = Render3D::new(
            device,
            queue,
            gpu.config.format,
            &lbm.buffers.macro_data,
            &lbm.buffers.flags,
            NX,
            NY,
            NZ,
            RenderMode3D::FlowStreams,
            RENDER_MAX_VELOCITY as f64,
            RENDER_MIN_VELOCITY as f64,
            &gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }).create_view(&wgpu::TextureViewDescriptor::default())
        );

        let mesh_renderer = crate::render::mesh::MeshRenderer::new(
            device,
            &renderer.camera_buffer,
            gpu.config.format,
            gpu.config.width,
            gpu.config.height,
            &stl_vertices,
            &stl_indices,
        );

        let streamline_renderer = crate::render::streamlines::StreamlineRenderer::new(
            device,
            &renderer.camera_buffer,
            &lbm.buffers.macro_data,
            &lbm.buffers.flags,
            gpu.config.format,
            wgpu::TextureFormat::Depth32Float,
            NX, NY, NZ,
            // --- F-35 Bounds ---
            // [50.0, NY as f32 * 0.4, NZ as f32 * 0.25], // seed_bounds_min
            // [50.0, NY as f32 * 0.55, NZ as f32 * 0.75], // seed_bounds_max
            // [1, 10, 10], // grid_size [X, Y, Z]
            // --- Sphere/Cylinder Bounds (Commented out) ---
            [50.0, NY as f32 * 0.25, NZ as f32 * 0.25], // seed_bounds_min
            [50.0, NY as f32 * 0.75, NZ as f32 * 0.75], // seed_bounds_max
            [1, 10, 10], // grid_size [X, Y, Z]
            256, // 256 points max
            &renderer.colormap_view,
            &renderer.sampler,
        );

        renderer.update_depth_view(device, &mesh_renderer.depth_view);

        let mut camera = crate::camera::Camera3D::new(window.inner_size());
        
        camera.target = glam::Vec3::new(NX as f32 * 0.5, NY as f32 * 0.5, NZ as f32 * 0.5);
        camera.distance = 350.0;
        camera.yaw = -0.3; // look upstream towards the sphere
        camera.pitch = std::f32::consts::PI / 6.0;
        camera.update();

        println!(
            "Simulation ready — {}×{}×{}, ω={:.4}, u_in={}, {} steps/frame",
            NX, NY, NZ, omega, u_design, STEPS_PER_FRAME
        );

        self.state = Some(SimState {
            window,
            gpu,
            lbm,
            renderer,
            mesh_renderer,
            camera,
            mouse_pressed: false,
            right_mouse_pressed: false,
            last_cursor_pos: None,
            iso_q: 0.000003,
            max_speed: 0.6,
            streamline_renderer,
            render_mode: crate::render::RenderMode3D::FlowStreams,
            gpu_setup_time,
            setup_time: t1.elapsed(),
            sim_start: std::time::Instant::now(),
            printed_stats: false,
            paused: false,
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
                state.mesh_renderer.resize(&state.gpu.device, size.width, size.height);
                state.renderer.update_depth_view(&state.gpu.device, &state.mesh_renderer.depth_view);
            }

            WindowEvent::MouseInput {
                state: element_state,
                button,
                ..
            } => {
                if button == winit::event::MouseButton::Left {
                    state.mouse_pressed = element_state == winit::event::ElementState::Pressed;
                } else if button == winit::event::MouseButton::Right {
                    state.right_mouse_pressed = element_state == winit::event::ElementState::Pressed;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if let Some(last_pos) = state.last_cursor_pos {
                    let dx = (position.x - last_pos.x) as f32;
                    let dy = (position.y - last_pos.y) as f32;
                    if state.mouse_pressed {
                        state.camera.rotate(dx, dy);
                    } else if state.right_mouse_pressed {
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

            WindowEvent::KeyboardInput { event: kb_event, .. } => {
                if kb_event.state == winit::event::ElementState::Pressed && !kb_event.repeat {
                    match kb_event.physical_key {
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowUp) => {
                            state.iso_q *= 1.5;
                            println!("ISO_Q: {}", state.iso_q);
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowDown) => {
                            state.iso_q /= 1.5;
                            println!("ISO_Q: {}", state.iso_q);
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowRight) => {
                            state.max_speed *= 1.5;
                            println!("MAX_SPEED: {}", state.max_speed);
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowLeft) => {
                            state.max_speed /= 1.5;
                            println!("MAX_SPEED: {}", state.max_speed);
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Digit1) => {
                            state.render_mode = crate::render::RenderMode3D::QCriterion;
                            println!("Render Mode: Q-Criterion");
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Digit2) => {
                            state.render_mode = crate::render::RenderMode3D::FlowStreams;
                            println!("Render Mode: Flow Streams");
                        }
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space) => {
                            state.paused = !state.paused;
                            println!("Simulation {}", if state.paused { "paused" } else { "resumed" });
                        }
                        _ => {}
                    }
                }
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
                let inv_view_proj = state.camera.matrix().inverse();
                let view_proj = state.camera.matrix();
                state.renderer.update_camera(queue, inv_view_proj, view_proj, state.camera.eye, state.max_speed, state.iso_q, state.render_mode as u32);

                if !state.paused {
                    // compute: step
                    state.lbm.step_multiple(&mut encoder, STEPS_PER_FRAME);
                    
                    // Only extract and trace streamlines periodically to save compute
                    if state.lbm.step_count % EXTRACT_INTERVAL == 0 || state.lbm.step_count <= STEPS_PER_FRAME {
                        state.lbm.extract(&mut encoder);
                        
                        if state.render_mode == crate::render::RenderMode3D::QCriterion {
                            state.renderer.compute_vorticity(&mut encoder);
                        } else if state.render_mode == crate::render::RenderMode3D::FlowStreams {
                            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("Streamline Tracing Pass"),
                                timestamp_writes: None,
                            });
                            state.streamline_renderer.compute(&mut pass);
                        }
                    }
                }

                // render pass
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Mesh Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &state.mesh_renderer.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    state.mesh_renderer.render(&mut pass);
                }

                {
                    let depth_stencil = if state.render_mode == crate::render::RenderMode3D::FlowStreams {
                        Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &state.mesh_renderer.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        })
                    } else { None };

                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass 3D"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: depth_stencil,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    
                    if state.render_mode == crate::render::RenderMode3D::QCriterion {
                        state.renderer.render(&mut pass);
                    } else if state.render_mode == crate::render::RenderMode3D::FlowStreams {
                        state.streamline_renderer.render(&mut pass);
                    }
                }

                queue.submit(std::iter::once(encoder.finish()));
                queue.present(frame);

                state
                    .window
                    .set_title(&format!("RustFluid 3D — Step {}", state.lbm.step_count));

                if state.lbm.step_count >= 1000 && !state.printed_stats {
                    state.printed_stats = true;

                    state.gpu.device
                                        .poll(wgpu::PollType::wait_indefinitely())
                                        .unwrap();

                    let compute_time = state.sim_start.elapsed();

                    
                    let total_time = state.gpu_setup_time + state.setup_time + compute_time;
                    let num_steps = state.lbm.step_count;
                    let seconds_per_step = compute_time.as_secs_f64() / num_steps as f64;
                    let steps_per_second = num_steps as f64 / compute_time.as_secs_f64();
                    
                    let nx = NX as f64;
                    let ny = NY as f64;
                    let nz = NZ as f64;
                    let mlups = (nx * ny * nz * num_steps as f64)
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
