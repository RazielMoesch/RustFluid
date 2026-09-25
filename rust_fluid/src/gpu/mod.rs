
pub mod utils;

use std::sync::Arc;
use winit::window::Window;

pub struct GPU {
    pub instance: wgpu::Instance,
    pub surface: wgpu::Surface<'static>,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub config: wgpu::SurfaceConfiguration
}

impl GPU {

    pub async fn new( window: Arc<Window> ) -> Self {

        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).expect("Surface Creation Failed.");
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            }
        ).await.expect("Failed to Find Adapter.");

        let ( device, queue ) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("GPU"),
                required_features: wgpu::Features::VERTEX_WRITABLE_STORAGE,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            }
        ).await.expect("Failed to Get Device & Queue");

        let config = surface.get_default_config(&adapter, size.width, size.height).expect("Failed to Get Configuration");
        surface.configure(&device, &config);

        Self {
            instance,
            surface,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            config
        }

    }

    pub fn resize( &mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }

        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }

}

pub struct HeadlessGPU {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl HeadlessGPU {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .expect("Failed to find adapter for headless GPU");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Headless GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("Failed to get headless device & queue");

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        }
    }
}
