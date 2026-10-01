pub mod fluid;
pub mod bounce_back;
pub mod equilibrium_inlet;
pub mod outlet;
pub mod free_slip;
pub mod zou_he;

pub trait Boundary3D {
    fn type_id(&self) -> u32;
    fn name(&self) -> &'static str;
    fn pull_even(&self) -> Option<&'static str> { None }
    fn pull_odd(&self) -> Option<&'static str> { None }
    fn post_streaming(&self) -> Option<&'static str> { None }
}
