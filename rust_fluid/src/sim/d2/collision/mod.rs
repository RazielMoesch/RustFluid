pub mod bgk;
pub mod mrt;

pub trait Collision2D {
    fn name(&self) -> &'static str;
    fn wgsl(&self) -> &'static str;
}
