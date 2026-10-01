use std::sync::Arc;
use winit::window::Window;
// use crate::gpu::GPU;

pub struct GraphicsContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub surface_format: wgpu::TextureFormat,
    pub window: &'a Arc<Window>,
}

pub struct HeadlessContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
}
