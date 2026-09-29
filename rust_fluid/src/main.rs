fn main() {
    let mut args = std::env::args();
    args.next(); // Skip executable name
    let stl_path = "assets/f35.stl"; // Optional STL path
    
    // rust_fluid::app_no_ui_2d::run(None);
    // rust_fluid::test_headless_2d::run();

    // rust_fluid::test_headless_3d::run();
    rust_fluid::example_2_3d::run(Some(stl_path.as_ref()));
}
