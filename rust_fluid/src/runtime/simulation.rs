//! Common lifecycle implemented by 2D and 3D solvers.

/// Records initialization, steps, extraction, and exposes macro output.
pub trait Simulation {
    fn step(&mut self, encoder: &mut wgpu::CommandEncoder, steps: u32);
    fn extract(&mut self, encoder: &mut wgpu::CommandEncoder);
    fn step_count(&self) -> u64;
    fn cell_count(&self) -> u64 {
        0
    }
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

    fn cell_count(&self) -> u64 {
        u64::from(self.config.nx) * u64::from(self.config.ny)
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

    fn cell_count(&self) -> u64 {
        u64::from(self.config.nx) * u64::from(self.config.ny) * u64::from(self.config.nz)
    }
}
