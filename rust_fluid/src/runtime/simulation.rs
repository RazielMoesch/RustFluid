pub trait Simulation {
    fn step(&mut self, encoder: &mut wgpu::CommandEncoder, steps: u32);
    fn extract(&mut self, encoder: &mut wgpu::CommandEncoder);
    fn step_count(&self) -> u64;
}

impl Simulation for crate::sim::d2::solver::Lbm2D {
    fn step(&mut self, encoder: &mut wgpu::CommandEncoder, steps: u32) {
        self.step_multiple(encoder, steps);
    }

    fn extract(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.extract(encoder);
    }

    fn step_count(&self) -> u64 {
        self.step_count as u64
    }
}

impl Simulation for crate::sim::d3::solver::Lbm3D {
    fn step(&mut self, encoder: &mut wgpu::CommandEncoder, steps: u32) {
        self.step_multiple(encoder, steps);
    }

    fn extract(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.extract(encoder);
    }

    fn step_count(&self) -> u64 {
        self.step_count as u64
    }
}
