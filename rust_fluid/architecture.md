# RustFluid Refactored Architecture

Welcome to the new architecture of RustFluid. The refactor focuses on explicit separation of concerns, composability through traits, and making the codebase friendly to contributors without sacrificing GPU performance.

## Design Philosophy
1. **Explicit Dimensions:** 2D and 3D simulation code live in completely separate modules (`sim::d2` and `sim::d3`). We prefer slight duplication over convoluted generics that try to abstract away fundamental dimensionality differences.
2. **Compile-time Composition:** We avoid dynamic trait objects in the hot loops and shader generation. Instead, we use traits like `Lattice3D`, `Collision3D`, and `Boundary3D` to cleanly injectWGSL code strings into a centralized `ShaderCompiler`.
3. **Single Responsibility:** Files should not be thousands of lines long.
   - `lattice/` contains purely the discrete velocity sets.
   - `collision/` contains purely the collision step logic (e.g., BGK, MRT).
   - `boundary/` contains purely the boundary logic (e.g., fluid streaming, bounce-back, Zou-He).
   - `shader/` handles composing these blocks into valid WGSL.
   - `solver.rs` coordinates WGPU pipeline creation and dispatch.

## Directory Structure
```
rust_fluid_refactored/
├── src/
│   ├── gpu/
│   │   ├── utils.rs       # WGPU boilerplate, bind group setup
│   │   └── mod.rs         # GPU core types
│   ├── sim/
│   │   ├── common/
│   │   │   └── precision.rs # Handles f32 / fp16 storage configurations
│   │   ├── d2/            # 2D Simulation Core
│   │   │   ├── boundary/  # Boundary condition implementations
│   │   │   ├── collision/ # Collision models
│   │   │   ├── lattice/   # Lattice models (e.g., D2Q9)
│   │   │   ├── shader/    # 2D Shader compiler and templates
│   │   │   ├── buffers.rs # 2D WGPU Buffers
│   │   │   ├── config.rs  # 2D Configuration struct
│   │   │   └── solver.rs  # Main Lbm2D Orchestrator
│   │   ├── d3/            # 3D Simulation Core
│   │   │   ├── boundary/  # Boundary condition implementations
│   │   │   ├── collision/ # Collision models
│   │   │   ├── lattice/   # Lattice models (e.g., D3Q19, D3Q27)
│   │   │   ├── shader/    # 3D Shader compiler and templates
│   │   │   ├── buffers.rs # 3D WGPU Buffers
│   │   │   ├── config.rs  # 3D Configuration struct
│   │   │   └── solver.rs  # Main Lbm3D Orchestrator
│   │   └── mod.rs
│   └── lib.rs
```

## How the Shader Compiler Works
Instead of maintaining massive WGSL strings with complex branching macros, the `ShaderCompiler` accepts a specific lattice, collision model, and list of boundaries.

1. It loads a base template (`BASE_STEP_EVEN`, `BASE_STEP_ODD`).
2. It requests unrolled fast-paths from the Lattice (based on `EX`, `EY`, `EZ` arrays).
3. It asks the Collision model for its localized string logic.
4. It iterates over the provided Boundaries and inserts their specific `pull_even`, `pull_odd`, and `post_streaming` snippets into a switch-case.
5. It handles precision modifications automatically based on the requested `PrecisionConfig`.

This ensures that adding new physics does not require editing the central template files.
