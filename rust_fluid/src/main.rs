//! Default executable; currently runs the headless FluidX3D comparison case.

fn main() {
    // let stl_path = "rust_fluid/assets/ferrari.stl";
    // rust_fluid::examples::stl_windtunnel::run(Some(stl_path.as_ref()));
    rust_fluid::examples::headless_3d::run();
}
