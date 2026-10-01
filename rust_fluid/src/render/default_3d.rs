use crate::render::{
    mesh::{MeshRenderer, Vertex},
    streamlines::StreamlineRenderer,
    particles::ParticleRenderer,
    qcriterion::QCriterionRenderer3D,
    Render3D, RenderMode3D,
};
use crate::camera::Camera3D;
use crate::runtime::{GraphicsContext, Renderer};
use crate::sim::d3::solver::Lbm3D;
use winit::event::{WindowEvent, ElementState, MouseButton};
use winit::keyboard::{KeyCode, PhysicalKey};

pub struct DefaultRenderer3D {
    pub flow_field: Render3D,
    pub particles: ParticleRenderer,
    pub streamlines: StreamlineRenderer,
    pub q_surface: QCriterionRenderer3D,
    pub mesh: Option<MeshRenderer>,
    pub camera: Camera3D,

    pub show_flow_streams: bool,
    pub show_streamlines: bool,
    pub show_q_surface: bool,

    pub needs_q_update: bool,

    pub iso_q: f32,
    pub max_speed: f32,

    /// Fallback depth texture when no mesh is present
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    mouse_pressed: bool,
    right_mouse_pressed: bool,
    last_cursor_pos: Option<winit::dpi::PhysicalPosition<f64>>,
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("DefaultRenderer3D Depth"),
        size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

impl DefaultRenderer3D {
    pub fn new(
        ctx: &GraphicsContext,
        lbm: &Lbm3D,
        nx: u32,
        ny: u32,
        nz: u32,
        mesh_data: Option<(&[Vertex], &[u32])>,
        seed_bounds_min: [f32; 3],
        seed_bounds_max: [f32; 3],
        seed_grid_size: [u32; 3],
    ) -> Self {
        let window_size = ctx.window.inner_size();
        let (depth_texture, depth_view) = create_depth_texture(ctx.device, window_size.width, window_size.height);

        let dummy_depth_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Depth"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_depth_view = dummy_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut flow_field = Render3D::new(
            ctx.device,
            ctx.queue,
            ctx.surface_format,
            &lbm.buffers.macro_data,
            &lbm.buffers.flags,
            nx,
            ny,
            nz,
            RenderMode3D::FlowStreams,
            1.0,
            0.0,
            &dummy_depth_view,
        );

        let streamlines = StreamlineRenderer::new(
            ctx.device,
            &flow_field.camera_buffer, // Streamlines share the same camera buffer as flow_field
            &lbm.buffers.macro_data,
            &lbm.buffers.flags,
            ctx.surface_format,
            wgpu::TextureFormat::Depth32Float,
            nx, ny, nz,
            seed_bounds_min,
            seed_bounds_max,
            seed_grid_size,
            256, // max_points
            0.5, // thickness
            &flow_field.colormap_view, // Share colormap
            &flow_field.sampler,
        );

        let particles = ParticleRenderer::new(
            ctx.device,
            &flow_field.camera_buffer,
            &lbm.buffers.macro_data,
            &lbm.buffers.flags,
            ctx.surface_format,
            nx, ny, nz,
            10000, // num_particles
            0.05, // thickness
            &flow_field.colormap_view,
            &flow_field.sampler,
        );

        let mesh = mesh_data.map(|(vertices, indices)| {
            let m = MeshRenderer::new(
                ctx.device,
                &flow_field.camera_buffer,
                ctx.surface_format,
                ctx.window.inner_size().width,
                ctx.window.inner_size().height,
                vertices,
                indices,
            );
            flow_field.update_depth_view(ctx.device, &m.depth_view);
            m
        });

        // Q-criterion Marching Cubes renderer — the new pipeline
        let q_surface = QCriterionRenderer3D::new(
            ctx.device,
            ctx.queue,
            ctx.surface_format,
            &lbm.buffers.macro_data,
            &lbm.buffers.flags,
            &flow_field.camera_buffer,
            nx,
            ny,
            nz,
            wgpu::TextureFormat::Depth32Float,
        );

        let mut camera = Camera3D::new(ctx.window.inner_size());
        camera.target = glam::Vec3::new(nx as f32 * 0.5, ny as f32 * 0.5, nz as f32 * 0.5);
        camera.distance = 350.0;
        camera.yaw = -0.3;
        camera.pitch = std::f32::consts::PI / 6.0;

        Self {
            flow_field,
            particles,
            streamlines,
            q_surface,
            mesh,
            camera,
            show_streamlines: false,
            show_q_surface: true,
            show_flow_streams: false,
            needs_q_update: true,
            iso_q: 0.0001,
            max_speed: 1.5,
            depth_texture,
            depth_view,
            mouse_pressed: false,
            right_mouse_pressed: false,
            last_cursor_pos: None,
        }
    }

    /// Get the active depth view.
    fn active_depth_view(&self) -> &wgpu::TextureView {
        if let Some(m) = &self.mesh {
            &m.depth_view
        } else {
            &self.depth_view
        }
    }
}

impl Renderer<Lbm3D> for DefaultRenderer3D {
    fn resize(&mut self, device: &wgpu::Device, _queue: &wgpu::Queue, width: u32, height: u32) {
        self.camera.resize(winit::dpi::PhysicalSize::new(width, height));

        // Resize fallback depth texture
        let (dt, dv) = create_depth_texture(device, width, height);
        self.depth_texture = dt;
        self.depth_view = dv;

        if let Some(mesh) = &mut self.mesh {
            mesh.resize(device, width, height);
            self.flow_field.update_depth_view(device, &mesh.depth_view);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.mouse_pressed = *state == ElementState::Pressed;
                } else if *button == MouseButton::Right {
                    self.right_mouse_pressed = *state == ElementState::Pressed;
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some(last_pos) = self.last_cursor_pos {
                        let dx = (position.x - last_pos.x) as f32;
                        let dy = (position.y - last_pos.y) as f32;
                        self.camera.rotate(dx, dy);
                    }
                } else if self.right_mouse_pressed {
                    if let Some(last_pos) = self.last_cursor_pos {
                        let dx = (position.x - last_pos.x) as f32;
                        let dy = (position.y - last_pos.y) as f32;
                        self.camera.pan(dx, dy);
                    }
                }
                self.last_cursor_pos = Some(*position);
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let zoom = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                self.camera.zoom(zoom);
                true
            }
            WindowEvent::KeyboardInput { event: kb, .. } if kb.state == ElementState::Pressed => {
                match kb.physical_key {
                    PhysicalKey::Code(KeyCode::ArrowUp) => {
                        self.iso_q *= 1.25;
                        self.needs_q_update = true;
                        println!("ISO_Q: {:.2e}", self.iso_q);
                    }
                    PhysicalKey::Code(KeyCode::ArrowDown) => {
                        self.iso_q /= 1.25;
                        self.needs_q_update = true;
                        println!("ISO_Q: {:.2e}", self.iso_q);
                    }
                    PhysicalKey::Code(KeyCode::BracketRight) => {
                        self.max_speed *= 1.25;
                        self.needs_q_update = true;
                        println!("MAX_SPEED: {}", self.max_speed);
                    }
                    PhysicalKey::Code(KeyCode::BracketLeft) => {
                        self.max_speed /= 1.25;
                        self.needs_q_update = true;
                        println!("MAX_SPEED: {}", self.max_speed);
                    }
                    PhysicalKey::Code(KeyCode::Digit1) => {
                        self.show_streamlines = !self.show_streamlines;
                        println!("Show Streamlines: {}", self.show_streamlines);
                    }
                    PhysicalKey::Code(KeyCode::Digit2) => {
                        self.show_q_surface = !self.show_q_surface;
                        if self.show_q_surface {
                            self.needs_q_update = true;
                        }
                        println!("Show Q Surface (Marching Cubes): {}", self.show_q_surface);
                    }
                    PhysicalKey::Code(KeyCode::Digit3) => {
                        self.show_flow_streams = !self.show_flow_streams;
                        println!("Show Flow Streams: {}", self.show_flow_streams);
                    }
                    _ => return false,
                }
                true
            }
            _ => false,
        }
    }

    fn prepare(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _simulation: &mut Lbm3D,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if self.show_streamlines && self.needs_q_update {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Streamline Compute Pass"),
                timestamp_writes: None,
            });
            self.streamlines.compute(&mut pass);
        }

        // Q-criterion Marching Cubes: compute Q + generate triangles
        if self.show_q_surface && self.needs_q_update {
            self.q_surface.prepare(queue, encoder, self.iso_q, self.max_speed);
        }

        // Compute particle advection for flow streams (every frame if visible)
        if self.show_flow_streams {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Particle Compute Pass"),
                timestamp_writes: None,
            });
            self.particles.advect(&mut pass);
        }

        if self.needs_q_update {
            self.needs_q_update = false;
        }

        let matrix = self.camera.matrix();
        let inv_matrix = matrix.inverse();

        self.flow_field.update_camera(
            queue,
            inv_matrix,
            matrix,
            self.camera.eye,
            self.max_speed,
            self.iso_q,
            0, // Mode 0 = FlowStreams
        );

        // Update Q surface camera
        if self.show_q_surface {
            self.q_surface.update_camera(queue, matrix, self.camera.eye, self.max_speed);
        }
    }

    fn render(
        &mut self,
        _simulation: &Lbm3D,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        let active_depth = self.active_depth_view();

        // Pass 1: Geometry (Mesh + Q Surface + Streamlines)
        {
            let needs_depth = self.show_q_surface || self.show_streamlines || self.mesh.is_some();
            let depth_stencil = if needs_depth {
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view: active_depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                })
            } else {
                None
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Geometry Pass (Mesh + Q Surface)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02, g: 0.02, b: 0.02, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: depth_stencil,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Draw mesh first (writes depth)
            if let Some(m) = &self.mesh {
                m.render(&mut pass);
            }

            // Draw Q surface triangles (depth test + write)
            if self.show_q_surface {
                self.q_surface.render(&mut pass);
            }

            // Draw streamlines (depth test)
            if self.show_streamlines {
                self.streamlines.render(&mut pass);
            }
        }

        // Pass 2: Volume/Particle Raycasting (no depth — overlay on top)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Volume Raycasting Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
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

            if self.show_flow_streams {
                self.particles.render(&mut pass);
            }
        }
    }

    fn on_extract(&mut self) {
        self.needs_q_update = true;
    }
}
