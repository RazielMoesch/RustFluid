use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::gpu::GPU;
use crate::render::{Render3D, RenderMode3D};
use crate::sim::lattices::{CollisionLogic, D3Q19, Lattice3D, Precision};
use crate::sim::lbm::LBM3D;

const STEPS_PER_FRAME: u32 = 10; // Increased to let flow develop faster

// Lattice resolution (defines the grid size)
const NX: u32 = 768;
const NY: u32 = 192;
const NZ: u32 = 64;

const WGS_X: u32 = 32;
const WGS_Y: u32 = 4;
const WGS_Z: u32 = 2;

const INLET_VELOCITY: f32 = 0.0577;

const RENDER_MAX_VELOCITY: f32 = 0.06;
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
    particle_renderer: crate::render::particles::ParticleRenderer,
    render_mode: crate::render::RenderMode3D,
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

        // 3D Domain Setup: Channel flow with an obstacle
        let domain = crate::setup::SimDomain3D::new()
            .with_w(NX)
            .with_h(NY)
            .with_d(NZ)
            ;

        let mut flags = domain.flags();
        let shift: u32 = 24;

        let mut stl_vertices = Vec::new();
        let mut stl_indices = Vec::new();

        if let Some(path) = &self.stl_path {
            println!("Loading STL from {}", path);
            match crate::setup::loader::Loader::load_stl(path, NX / 2, NY / 2, NZ / 2) {
                Ok(voxel_grid) => {
                    let offset_x = (NX - voxel_grid.width) / 4;
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
            // Replicate FluidX3D setup (Cylinder)
            let cx = 640; // 768 - 128 (Move to the right side so wake flows to the left)
            let cy = 96;  // middle of the 192-cell height
            let r = 32;   // D = 64
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
        let reynolds: f32 = 200.0;
        let d = 64.0;
        let u_design = 0.577;
        let nu = (u_design * d) / reynolds;
        let tau = 3.0 * nu + 0.5;
        let omega = 1.0 / tau;
        
        let u_init = -u_design * 0.1;
        let force_x = -8.0 * nu * u_design / ((NY as f32) * (NY as f32));

        println!("Reynolds Number: {}, Kinematic Viscosity: {:.6}, Tau: {:.6}, Force: {:.3e}", reynolds, nu, tau, force_x);

        let lattice = Lattice3D::D3Q19(
            D3Q19::new()
                .with_collision_logic(CollisionLogic::BGK)
                .with_precision(Precision::FP16S),
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
            1, // periodic_x
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

        let particle_renderer = crate::render::particles::ParticleRenderer::new(
            device,
            &renderer.camera_buffer,
            &lbm.buffers.macro_data,
            &lbm.buffers.flags,
            gpu.config.format,
            NX,
            NY,
            NZ,
            50000, // 50,000 streamlines
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
            NX, NY, NZ, omega, INLET_VELOCITY, STEPS_PER_FRAME
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
            max_speed: 0.086,
            particle_renderer,
            render_mode: crate::render::RenderMode3D::FlowStreams,
            gpu_setup_time,
            setup_time: t1.elapsed(),
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

                if state.lbm.step_count == 100 {
                    let nx = NX;
                    let ny = NY;
                    let nz = NZ;
                    let size = (nx * ny * nz * 16) as wgpu::BufferAddress;
                    let staging_buffer = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let mut encoder = state.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    encoder.copy_buffer_to_buffer(&state.lbm.buffers.macro_data, 0, &staging_buffer, 0, size);
                    state.gpu.queue.submit(Some(encoder.finish()));

                    let buffer_slice = staging_buffer.slice(..);
                    let (tx, rx) = std::sync::mpsc::channel();
                    buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
                    state.gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    rx.recv().unwrap().unwrap();

                    let data = buffer_slice.get_mapped_range().unwrap();
                    let macro_data: &[[f32; 4]] = bytemuck::cast_slice(&data);

                    let mut max_q = 0.0f32;
                    let mut count = 0;
                    let mut valid_count = 0;
                    for z in 1..(nz-1) {
                        for y in 1..(ny-1) {
                            for x in 1..(nx-1) {
                                // Basic validation: skip solid boundaries (if any)
                                // We don't have flags here easily, but we can just check if speed is 0 maybe?
                                
                                valid_count += 1;
                                
                                let get_u = |xi, yi, zi| {
                                    let i = (xi + yi * nx + zi * nx * ny) as usize;
                                    let m = macro_data[i];
                                    glam::Vec3::new(m[0], m[1], m[2])
                                };

                                let u_r = get_u(x+1, y, z);
                                let u_l = get_u(x-1, y, z);
                                let u_t = get_u(x, y+1, z);
                                let u_b = get_u(x, y-1, z);
                                let u_f = get_u(x, y, z+1);
                                let u_k = get_u(x, y, z-1);

                                let ux_x = (u_r.x - u_l.x) * 0.5;
                                let ux_y = (u_t.x - u_b.x) * 0.5;
                                let ux_z = (u_f.x - u_k.x) * 0.5;

                                let uy_x = (u_r.y - u_l.y) * 0.5;
                                let uy_y = (u_t.y - u_b.y) * 0.5;
                                let uy_z = (u_f.y - u_k.y) * 0.5;

                                let uz_x = (u_r.z - u_l.z) * 0.5;
                                let uz_y = (u_t.z - u_b.z) * 0.5;
                                let uz_z = (u_f.z - u_k.z) * 0.5;

                                let s_xx = ux_x;
                                let s_yy = uy_y;
                                let s_zz = uz_z;
                                let s_xy = 0.5 * (ux_y + uy_x);
                                let s_xz = 0.5 * (ux_z + uz_x);
                                let s_yz = 0.5 * (uy_z + uz_y);

                                let norm_s2 = s_xx*s_xx + s_yy*s_yy + s_zz*s_zz + 2.0*(s_xy*s_xy + s_xz*s_xz + s_yz*s_yz);

                                let o_xy = 0.5 * (ux_y - uy_x);
                                let o_xz = 0.5 * (ux_z - uz_x);
                                let o_yz = 0.5 * (uy_z - uz_y);

                                let norm_omega2 = 2.0*(o_xy*o_xy + o_xz*o_xz + o_yz*o_yz);

                                let q_crit = 0.5 * (norm_omega2 - norm_s2);
                                
                                if q_crit > 0.0 {
                                    if q_crit > max_q { max_q = q_crit; }
                                    if q_crit >= state.iso_q {
                                        count += 1;
                                    }
                                }
                            }
                        }
                    }
                    println!("DIAGNOSTIC (Step {}): Max Q = {:.8}, Count above iso_q ({}) = {} / {}", state.lbm.step_count, max_q, state.iso_q, count, valid_count);
                }


                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Frame Encoder"),
                });

                // update camera uniform buffer
                let inv_view_proj = state.camera.matrix().inverse();
                let view_proj = state.camera.matrix();
                state.renderer.update_camera(queue, inv_view_proj, view_proj, state.camera.eye, state.max_speed, state.iso_q, state.render_mode as u32);

                // compute: step + extract + vorticity
                state.lbm.step_multiple(&mut encoder, STEPS_PER_FRAME);
                state.lbm.extract(&mut encoder);
                
                if state.render_mode == crate::render::RenderMode3D::QCriterion {
                    state.renderer.compute_vorticity(&mut encoder);
                } else if state.render_mode == crate::render::RenderMode3D::FlowStreams {
                    // We need a compute pass to advect the particles
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Particle Advection Pass"),
                        timestamp_writes: None,
                    });
                    state.particle_renderer.advect(&mut pass);
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
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    
                    if state.render_mode == crate::render::RenderMode3D::QCriterion {
                        state.renderer.render(&mut pass);
                    } else if state.render_mode == crate::render::RenderMode3D::FlowStreams {
                        state.particle_renderer.render(&mut pass);
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
