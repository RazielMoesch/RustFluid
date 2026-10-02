//! Shared headless GPU context and diagnostic result collection.

use crate::runtime::context::HeadlessContext;

/// Creates one adapter context and executes named diagnostic closures.
pub struct DiagnosticRunner {
    pub results: Vec<crate::diagnostics::result::DiagnosticResult>,
}

impl DiagnosticRunner {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    pub fn run_with_context<F>(&mut self, name: &str, f: F)
    where
        F: FnOnce(&HeadlessContext) -> crate::diagnostics::result::DiagnosticResult,
    {
        println!("Running diagnostic: {}", name);
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("Failed to get headless adapter");

        let mut required = wgpu::Features::VERTEX_WRITABLE_STORAGE;
        if adapter.features().contains(wgpu::Features::SHADER_F16) {
            required |= wgpu::Features::SHADER_F16;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Diagnostic GPU"),
            required_features: required,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("Failed to get headless device & queue");

        let ctx = HeadlessContext {
            device: &device,
            queue: &queue,
        };

        let result = f(&ctx);
        println!("Result: {:?}", result.status);
        for msg in &result.messages {
            println!("  - {}", msg);
        }
        self.results.push(result);
    }
}
