//! Interactive composition of mesh, Q surface, streamlines, particles, and wireframe.

use crate::camera::Camera3D;
use crate::render::{
    Render3D, RenderMode3D,
    mesh::{MeshRenderer, Vertex},
    particles::ParticleRenderer,
    qcriterion::QCriterionRenderer3D,
    streamlines::StreamlineRenderer,
    wireframe::DomainWireframeRenderer,
};
use crate::runtime::{GraphicsContext, Renderer};
use crate::sim::d3::solver::Lbm3D;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Default renderer and input state for a three-dimensional LBM simulation.
pub struct DefaultRenderer3D {
    pub flow_field: Render3D,
    pub particles: ParticleRenderer,
    pub streamlines: StreamlineRenderer,
    pub wireframe: DomainWireframeRenderer,
    pub q_surface: QCriterionRenderer3D,
    pub mesh: Option<MeshRenderer>,
    pub camera: Camera3D,

    pub show_flow_streams: bool,
    pub show_streamlines: bool,
    pub show_q_surface: bool,
    pub show_wireframe: bool,
    pub show_mesh: bool,
    streamline_visible_fraction: f32,
    q_visible_fraction: f32,
    flow_visible_fraction: f32,
    /// Section-view clip axis: 0 = X, 1 = Y, 2 = Z.
    clip_axis: u32,
    /// If true, the visible fraction is measured from the high end of the axis.
    clip_from_max: bool,

    pub needs_q_update: bool,

    pub iso_q: f32,
    pub max_speed: f32,

    /// Fallback depth texture when no mesh is present
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,

    mouse_pressed: bool,
    right_mouse_pressed: bool,
    last_cursor_pos: Option<winit::dpi::PhysicalPosition<f64>>,
    pub is_paused: bool,
}

fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("DefaultRenderer3D Depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
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
        let (depth_texture, depth_view) =
            create_depth_texture(ctx.device, window_size.width, window_size.height);

        let dummy_depth_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Depth"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_depth_view =
            dummy_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
            nx,
            ny,
            nz,
            seed_bounds_min,
            seed_bounds_max,
            seed_grid_size,
            512,                       // max_points; enough to cross the 384-cell tunnel
            0.35,                      // world-space ribbon half-width
            &flow_field.colormap_view, // Share colormap
            &flow_field.sampler,
        );

        let particles = ParticleRenderer::new(
            ctx.device,
            &flow_field.camera_buffer,
            &lbm.buffers.macro_data,
            &lbm.buffers.flags,
            ctx.surface_format,
            wgpu::TextureFormat::Depth32Float,
            nx,
            ny,
            nz,
            10000, // num_particles
            0.5,   // thickness; 0.1 projects to less than one pixel at startup
            &flow_field.colormap_view,
            &flow_field.sampler,
        );

        let wireframe = DomainWireframeRenderer::new(
            ctx.device,
            &flow_field.camera_buffer,
            ctx.surface_format,
            wgpu::TextureFormat::Depth32Float,
            nx,
            ny,
            nz,
            mesh_data,
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

        // Extract and shade a triangle isosurface from the velocity-gradient field.
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
        camera.fov = 45.0_f32.to_radians();
        let domain_radius = 0.5 * glam::Vec3::new(nx as f32, ny as f32, nz as f32).length();
        camera.distance = 0.75 * domain_radius / (0.5 * camera.fov).tan();
        camera.yaw = -0.3;
        camera.pitch = std::f32::consts::PI / 6.0;
        // Applying orbit parameters does not update `eye` automatically.
        // Leaving it at Camera3D::new()'s origin produces an invalid/common
        // view for every 3D rendering mode until the user first moves the camera.
        camera.update();

        let initial_mode = std::env::var("RUSTFLUID_RENDER_MODE").unwrap_or_else(|_| "q".into());
        println!(
            "Render keys: 1 streamlines, 2 Q, 3 flow, 4 wireframe, 5 STL mesh | clipping: I/P -/+10%, O cycle axis, L flip side"
        );

        Self {
            flow_field,
            particles,
            streamlines,
            wireframe,
            q_surface,
            mesh,
            camera,
            show_streamlines: initial_mode == "streamlines",
            show_q_surface: matches!(initial_mode.as_str(), "q" | "wireframe" | "boundaries"),
            show_flow_streams: initial_mode == "flow_streams",
            show_wireframe: initial_mode == "wireframe" || initial_mode == "boundaries",
            show_mesh: true,
            streamline_visible_fraction: 1.0,
            q_visible_fraction: 1.0,
            flow_visible_fraction: 1.0,
            clip_axis: 0,
            clip_from_max: false,
            needs_q_update: true,
            iso_q: 0.000003,
            max_speed: 0.15,
            depth_texture,
            depth_view,
            mouse_pressed: false,
            right_mouse_pressed: false,
            last_cursor_pos: None,
            is_paused: false,
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

    fn section_side_label(&self) -> &'static str {
        match (self.clip_axis, self.clip_from_max) {
            (0, false) => "left (min X)",
            (0, true) => "right (max X)",
            (1, false) => "bottom (min Y)",
            (1, true) => "top (max Y)",
            (_, false) => "front (min Z)",
            (_, true) => "back (max Z)",
        }
    }

    fn adjust_section(&mut self, delta: f32) {
        self.streamline_visible_fraction =
            (self.streamline_visible_fraction + delta).clamp(0.0, 1.0);
        self.q_visible_fraction = (self.q_visible_fraction + delta).clamp(0.0, 1.0);
        self.flow_visible_fraction = (self.flow_visible_fraction + delta).clamp(0.0, 1.0);
        println!(
            "Section visible: {:.0}% from {}",
            self.streamline_visible_fraction * 100.0,
            self.section_side_label()
        );
    }
}

impl Renderer<Lbm3D> for DefaultRenderer3D {
    fn resize(&mut self, device: &wgpu::Device, _queue: &wgpu::Queue, width: u32, height: u32) {
        self.camera
            .resize(winit::dpi::PhysicalSize::new(width, height));

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
                        if self.show_streamlines {
                            // Streamline buffers are generated lazily.  Without
                            // invalidating them here, enabling streamlines after
                            // the first frame renders the initial zeroed counts.
                            self.needs_q_update = true;
                        }
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
                    PhysicalKey::Code(KeyCode::Digit4) => {
                        self.show_wireframe = !self.show_wireframe;
                        println!("Show Domain/Geometry Wireframe: {}", self.show_wireframe);
                    }
                    PhysicalKey::Code(KeyCode::Digit5) => {
                        self.show_mesh = !self.show_mesh;
                        println!("Show STL Mesh: {}", self.show_mesh);
                    }
                    PhysicalKey::Code(KeyCode::KeyI) => {
                        self.adjust_section(-0.05);
                    }
                    PhysicalKey::Code(KeyCode::KeyP) => {
                        self.adjust_section(0.05);
                    }
                    PhysicalKey::Code(KeyCode::KeyO) => {
                        self.clip_axis = (self.clip_axis + 1) % 3;
                        let axis = match self.clip_axis {
                            0 => "X",
                            1 => "Y",
                            _ => "Z",
                        };
                        println!("Section axis: {axis} from {}", self.section_side_label());
                    }
                    PhysicalKey::Code(KeyCode::KeyL) => {
                        self.clip_from_max = !self.clip_from_max;
                        println!("Section measured from {}", self.section_side_label());
                    }
                    PhysicalKey::Code(KeyCode::Digit7) => {
                        self.q_visible_fraction = (self.q_visible_fraction - 0.1).clamp(0.0, 1.0);
                        println!(
                            "Q-criterion domain visible: {:.0}%",
                            self.q_visible_fraction * 100.0
                        );
                    }
                    PhysicalKey::Code(KeyCode::Digit8) => {
                        self.q_visible_fraction = (self.q_visible_fraction + 0.1).clamp(0.0, 1.0);
                        println!(
                            "Q-criterion domain visible: {:.0}%",
                            self.q_visible_fraction * 100.0
                        );
                    }
                    PhysicalKey::Code(KeyCode::Digit9) => {
                        self.flow_visible_fraction =
                            (self.flow_visible_fraction - 0.1).clamp(0.0, 1.0);
                        println!(
                            "Flow-stream domain visible: {:.0}%",
                            self.flow_visible_fraction * 100.0
                        );
                    }
                    PhysicalKey::Code(KeyCode::Digit0) => {
                        self.flow_visible_fraction =
                            (self.flow_visible_fraction + 0.1).clamp(0.0, 1.0);
                        println!(
                            "Flow-stream domain visible: {:.0}%",
                            self.flow_visible_fraction * 100.0
                        );
                    }
                    PhysicalKey::Code(KeyCode::Space) => {
                        self.is_paused = !self.is_paused;
                        return false;
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
        self.streamlines.set_visible_fraction(
            queue,
            self.streamline_visible_fraction,
            self.clip_axis,
            self.clip_from_max,
        );
        self.particles.set_visible_fraction(
            queue,
            self.flow_visible_fraction,
            self.clip_axis,
            self.clip_from_max,
        );

        if self.show_streamlines && self.needs_q_update {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Streamline Compute Pass"),
                timestamp_writes: None,
            });
            self.streamlines.compute(&mut pass);
        }

        // Q-criterion Marching Cubes: compute Q + generate triangles
        if self.show_q_surface && self.needs_q_update {
            self.q_surface
                .prepare(queue, encoder, self.iso_q, self.max_speed);
        }

        // Compute particle advection for flow streams (every frame if visible)
        if self.show_flow_streams && !self.is_paused {
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
            self.q_surface.update_camera(
                queue,
                matrix,
                self.camera.eye,
                self.max_speed,
                self.q_visible_fraction,
                self.clip_axis,
                self.clip_from_max,
            );
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
            let needs_depth = self.show_q_surface
                || self.show_streamlines
                || self.show_flow_streams
                || self.show_wireframe
                || (self.show_mesh && self.mesh.is_some());
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
                            r: 0.02,
                            g: 0.02,
                            b: 0.02,
                            a: 1.0,
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
            if self.show_mesh {
                if let Some(m) = &self.mesh {
                    m.render(&mut pass);
                }
            }

            // Draw Q surface triangles (depth test + write)
            if self.show_q_surface {
                self.q_surface.render(&mut pass);
            }

            if self.show_streamlines {
                self.streamlines.render(&mut pass);
            }
            if self.show_flow_streams {
                self.particles.render(&mut pass);
            }
            if self.show_wireframe {
                self.wireframe.render(&mut pass);
            }
        }
    }

    fn on_extract(&mut self) {
        self.needs_q_update = true;
    }
}
