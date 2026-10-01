pub mod bgk;
pub mod trt;
pub mod mrt;

pub trait Collision3D {
    fn name(&self) -> &'static str;
    fn wgsl(&self) -> &'static str;
}
