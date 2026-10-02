//! Borrowed resources passed into simulation and renderer factories.

use std::sync::Arc;
use winit::window::Window;
// use crate::gpu::GPU;

/// Windowed GPU resources and current surface configuration.
pub struct GraphicsContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub surface_format: wgpu::TextureFormat,
    pub window: &'a Arc<Window>,
}

/// Device and queue available to a headless simulation factory.
pub struct HeadlessContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
}
