pub mod utils;

use std::sync::Arc;
use winit::window::Window;

pub struct GpuCapabilities {
    pub shader_f16: bool,
}

impl GpuCapabilities {
    pub fn from_device(device: &wgpu::Device) -> Self {
        Self {
            shader_f16: device.features().contains(wgpu::Features::SHADER_F16),
        }
    }

    pub fn validate_precision(
        &self,
        precision: crate::sim::common::precision::Precision,
    ) -> Result<(), String> {
        use crate::sim::common::precision::Precision;
        match precision {
            Precision::F32 => Ok(()),
            Precision::FP16S => {
                if self.shader_f16 {
                    Ok(())
                } else {
                    Err("FP16Storage requires SHADER_F16 support, which is unavailable on this device".to_string())
                }
            }
            Precision::Auto => Ok(()),
        }
    }

    pub fn resolve_precision(
        &self,
        precision: crate::sim::common::precision::Precision,
    ) -> Result<crate::sim::common::precision::Precision, String> {
        use crate::sim::common::precision::Precision;
        match precision {
            Precision::Auto => {
                if self.shader_f16 {
                    Ok(Precision::FP16S)
                } else {
                    Ok(Precision::F32)
                }
            }
            Precision::FP16S => {
                if self.shader_f16 {
                    Ok(Precision::FP16S)
                } else {
                    Err("FP16Storage requires SHADER_F16 support, which is unavailable on this device".to_string())
                }
            }
            Precision::F32 => Ok(Precision::F32),
        }
    }
}

pub fn check_f16_support(device: &wgpu::Device) -> bool {
    device.features().contains(wgpu::Features::SHADER_F16)
}

pub struct GPU {
    pub instance: wgpu::Instance,
    pub surface: wgpu::Surface<'static>,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub config: wgpu::SurfaceConfiguration,
}

impl GPU {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("Surface Creation Failed.");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .expect("Failed to Find Adapter.");

        let mut required = wgpu::Features::VERTEX_WRITABLE_STORAGE;
        if adapter.features().contains(wgpu::Features::SHADER_F16) {
            required |= wgpu::Features::SHADER_F16;
        }
        if adapter.features().contains(wgpu::Features::FLOAT32_FILTERABLE) {
            required |= wgpu::Features::FLOAT32_FILTERABLE;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("GPU"),
                required_features: required,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("Failed to Get Device & Queue");

        

        let config = surface
            .get_default_config(&adapter, size.width, size.height)

            .expect("Failed to Get Configuration");
        
        let cfg = wgpu::SurfaceConfiguration {
            present_mode: wgpu::PresentMode::Immediate,
            ..config
        };
        
        surface.configure(&device, &cfg);

        

        Self {
            instance,
            surface,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            config: cfg,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn capabilities(&self) -> GpuCapabilities {
        GpuCapabilities::from_device(&self.device)
    }
}

pub struct HeadlessGPU {
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl HeadlessGPU {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .expect("Failed to find adapter for headless GPU");

        let mut required = wgpu::Features::VERTEX_WRITABLE_STORAGE;
        if adapter.features().contains(wgpu::Features::SHADER_F16) {
            required |= wgpu::Features::SHADER_F16;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Headless GPU"),
                required_features: required,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("Failed to get headless device & queue");

        Self {
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
        }
    }

    pub fn capabilities(&self) -> GpuCapabilities {
        GpuCapabilities::from_device(&self.device)
    }

    pub fn adapter_name(&self) -> String {
        self.adapter.get_info().name
    }

    pub fn backend_name(&self) -> String {
        format!("{:?}", self.adapter.get_info().backend)
    }
}
