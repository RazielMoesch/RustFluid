# RustFluid architecture

RustFluid separates 2D and 3D simulation code while sharing execution,
precision, diagnostics, and rendering infrastructure. The small amount of
dimensional duplication keeps shader indexing and boundary behavior explicit.

## Data flow

1. A `SimulationConfig2D` or `SimulationConfig3D` defines the domain,
   relaxation, forcing, initialization, periodicity, and workgroup shape.
2. A lattice, collision model, and set of boundary components provide WGSL
   fragments to the dimensional shader compiler.
3. The compiler substitutes constants, emits unrolled lattice operations, and
   produces initialization, alternating AA-pattern step, outlet preparation,
   and macroscopic extraction shaders.
4. `Lbm2D` or `Lbm3D` owns the compute pipelines, bind groups, simulation
   buffers, dispatch geometry, and current AA phase (`step_count`).
5. A runtime records simulation steps and extraction in either a windowed
   event loop or a headless loop. Renderers consume extracted macroscopic
   values stored as velocity plus density.

The 3D macroscopic buffer stores `[u_x, u_y, u_z, rho]` per cell. Population
storage is in-place and alternates even/odd AA-pattern interpretation; callers
must use `extract` rather than reading populations directly.

## Module map

```text
src/
|-- sim/
|   |-- common/             Precision code generation and readback
|   |-- d2/
|   |   |-- lattice/        D2Q9 constants and direction arrays
|   |   |-- collision/      BGK and MRT collision fragments
|   |   |-- boundary/       2D flag-selected boundary fragments
|   |   |-- shader/         2D templates and composition
|   |   |-- buffers.rs      Population, flag, boundary, and macro buffers
|   |   |-- config.rs       D2 configuration
|   |   `-- solver.rs       Pipeline creation and dispatch
|   `-- d3/
|       |-- lattice/        D3Q19 constants and direction arrays
|       |-- collision/      BGK, MRT, TRT, and Smagorinsky LES
|       |-- boundary/       3D boundary fragments
|       |-- shader/         In-place shaders and compiler
|       |-- buffers.rs      Buffer allocation and byte accounting
|       |-- config.rs       D3 configuration, periodicity, and sponge
|       `-- solver.rs       Init, outlet, step, extract, and readback
|-- runtime/                Generic Simulation and Renderer orchestration
|-- render/                 Cameras and GPU visualization pipelines
|-- setup/                  Domain flag generation and geometry loading
|-- diagnostics/            Repeatable GPU correctness cases
|-- benchmark/              Timing and CSV helpers
|-- validation/             CPU-side comparison metrics
`-- examples/               Complete headless and windowed configurations
```

## Shader composition

`Lattice2D` and `Lattice3D` provide WGSL declarations plus native direction
arrays used to generate unrolled accesses. `Collision2D` and `Collision3D`
return code that updates the local population array. Boundary traits associate
a numeric cell type with pull/streaming behavior.

The compilers specialize shaders for:

- lattice direction count and opposite/reflection maps;
- collision WGSL;
- enabled boundary types;
- FP32 or packed FP16 storage;
- domain size, workgroup size, relaxation, and force constants;
- 3D periodic axes, uniform/Taylor-Green initialization, pure-fluid fast path,
  and optional outlet sponge settings.

The generated source is compiled once when the solver is constructed. Dynamic
dispatch therefore selects only the AA phase and optional outlet preparation;
the hot cell update does not use Rust-side virtual dispatch.

## Geometry flags and boundary data

Domain builders produce one `u32` flag per cell. The high byte stores the
boundary type and the remaining bits carry boundary configuration information
and internal optimization markers. Boundary configuration records are uploaded
separately and contain target velocity and density values.

`SimDomain2D` and `SimDomain3D` assign the outer faces and can be augmented with
SVG- or STL-derived solids. STL loading rotates and uniformly fits the source
into a requested voxel box, samples triangle surfaces into solid cells, and
welds sub-voxel triangles for display.

## Runtime and rendering

`Simulation` abstracts initialization, stepping, extraction, and access to the
macroscopic output. `GraphicsBuilder` combines a simulation factory with a
`Renderer`; `HeadlessBuilder` runs the same simulation contract without a
surface.

The 3D default renderer combines STL/cylinder meshes, Q-criterion
marching-cubes surfaces, compute-generated ribbon streamlines, advected
particles, and domain/obstacle wireframes. Rendering is downstream of
extraction and does not alter population state.

## Validation and performance

The `diagnostics` module executes solver-level invariants and returns structured
pass, warning, or fail results. The `validation` module computes CPU-side field
errors, mass drift, and invalid-cell reports from downloaded macro data.

The `benchmark` module provides 2D timing statistics and workgroup sweeps. The
headless 3D example owns the larger FluidX3D-layout benchmark and its detailed
probe output. Timings synchronize the GPU before measuring completed work.
