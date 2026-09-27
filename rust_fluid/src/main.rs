fn main() {
    let svg_path = std::env::args().nth(1);
    rust_fluid::app_no_ui::run(svg_path.as_deref());
    // rust_fluid::test_headless::run();
}
