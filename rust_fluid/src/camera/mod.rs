use glam::{Mat4, Vec3};

pub struct Camera2D {
    pub center: glam::Vec2,
    pub aspect: f32,
    pub zoom: f32,
    pub pan_sens: f32,
    pub zoom_sens: f32,
}

impl Camera2D {
    pub fn new(size: winit::dpi::PhysicalSize<u32>) -> Self {
        Self {
            center: glam::Vec2::new(0.0, 0.0),
            aspect: (size.width as f32 / size.height as f32).max(0.1),
            zoom: 1.0,
            pan_sens: 0.002, // tuned for orthographic
            zoom_sens: 0.1,  // tuned for orthographic
        }
    }

    pub fn matrix(&self) -> Mat4 {
        // Compute orthographic bounds based on zoom and window aspect ratio
        let half_width = self.aspect * self.zoom;
        let half_height = 1.0 * self.zoom;

        let left = self.center.x - half_width;
        let right = self.center.x + half_width;
        let bottom = self.center.y - half_height;
        let top = self.center.y + half_height;

        glam::camera::rh::proj::directx::orthographic(left, right, bottom, top, -1.0, 1.0)
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.height > 0 {
            self.aspect = size.width as f32 / size.height as f32;
        }
    }

    pub fn zoom(&mut self, delta: f32) {
        // delta is usually 1.0 or -1.0 per tick
        self.zoom -= delta * self.zoom_sens * self.zoom; // exponential zoom feels better
        self.zoom = self.zoom.clamp(0.05, 10.0);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        // Pan sensitivity should scale with zoom so it feels consistent
        self.center.x -= dx * self.pan_sens * self.zoom;
        self.center.y += dy * self.pan_sens * self.zoom; // flip Y for mouse drag
    }
}

pub struct Camera3D {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub aspect: f32,
    pub fov: f32,
    pub znear: f32,
    pub zfar: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub rotation_sens: f32,
    pub zoom_sens: f32,
    pub pan_sens: f32,
}

impl Camera3D {
    pub fn new(size: winit::dpi::PhysicalSize<u32>) -> Self {
        Self {
            eye: Vec3::new(0.0, 0.0, 0.0),
            target: Vec3::new(0.0, 0.0, 0.0),
            up: Vec3::Y,
            aspect: (size.width as f32 / size.height as f32),
            fov: 25.0_f32.to_radians(),
            znear: 1e-3,
            zfar: 1e4,
            yaw: 0.0,
            pitch: 0.0,
            distance: 5.0,
            rotation_sens: 0.005,
            zoom_sens: 5.0,
            pan_sens: 0.1,
        }
    }

    pub fn matrix(&self) -> Mat4 {
        let view = glam::camera::rh::view::look_at_mat4(self.eye, self.target, self.up);
        let proj = glam::camera::rh::proj::directx::perspective(
            self.fov,
            self.aspect,
            self.znear,
            self.zfar,
        );

        proj * view
    }

    pub fn update(&mut self) {
        let x = self.yaw.cos() * self.pitch.cos();
        let y = self.pitch.sin();
        let z = self.yaw.sin() * self.pitch.cos();

        let dir = Vec3::new(x, y, z).normalize();
        self.eye = self.target - (dir * self.distance);
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.aspect = size.width as f32 / size.height as f32;
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * self.rotation_sens;
        self.pitch -= dy * self.rotation_sens;
        self.pitch = self.pitch.clamp(-1.5, 1.5);
        self.update();
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance -= delta * self.zoom_sens;
        self.distance = self.distance.max(0.01);
        self.update();
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.eye).normalize();
        let right = forward.cross(self.up).normalize();
        let actual_up = right.cross(forward.normalize());
        let offset = (right * -dx * self.pan_sens) + (actual_up * dy * self.pan_sens);
        self.eye += offset;
        self.target += offset;
    }
}
