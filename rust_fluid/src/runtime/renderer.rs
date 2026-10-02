//! Renderer hooks driven by the windowed runtime.

use winit::event::WindowEvent;
// use winit::window::WindowId;

/// Prepares and renders frames for a particular simulation type.
pub trait Renderer<S> {
    fn resize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32);

    fn input(&mut self, event: &WindowEvent) -> bool {
        let _ = event;
        false
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        simulation: &mut S,
        encoder: &mut wgpu::CommandEncoder,
    );

    fn render(
        &mut self,
        simulation: &S,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    );

    /// Called by the runtime when the simulation has extracted new data to macro buffers.
    fn on_extract(&mut self) {}
}
