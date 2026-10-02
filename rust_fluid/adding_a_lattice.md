# Adding a lattice

A lattice defines discrete directions, equilibrium weights, opposite-direction
mapping, and optional reflection maps. RustFluid needs both WGSL declarations
and matching Rust slices because the shader compiler generates unrolled cell
accesses from the native arrays.

## 1. Implement the dimensional trait

Create a module under `src/sim/d2/lattice/` or `src/sim/d3/lattice/` and
implement `Lattice2D` or `Lattice3D`. Use `d2q9.rs` or `d3q19.rs` as the
canonical example.

```rust
use super::Lattice3D;

pub struct MyLattice;

impl MyLattice {
    pub fn new() -> Self { Self }
}

impl Lattice3D for MyLattice {
    fn name(&self) -> &'static str { "MyLattice" }
    fn q(&self) -> u32 { 27 }

    fn ex(&self) -> &'static str { "const EX = array<i32, 27>(/* ... */);" }
    fn ey(&self) -> &'static str { "const EY = array<i32, 27>(/* ... */);" }
    fn ez(&self) -> &'static str { "const EZ = array<i32, 27>(/* ... */);" }
    fn weights(&self) -> &'static str { "const WEIGHTS = array<f32, 27>(/* ... */);" }
    fn opp(&self) -> &'static str { "const OPP = array<u32, 27>(/* ... */);" }

    fn reflect_x(&self) -> Option<&'static str> { None }
    fn reflect_y(&self) -> Option<&'static str> { None }
    fn reflect_z(&self) -> Option<&'static str> { None }

    fn ex_array(&self) -> &[i32] { &[/* ... */] }
    fn ey_array(&self) -> &[i32] { &[/* ... */] }
    fn ez_array(&self) -> &[i32] { &[/* ... */] }
    fn opp_array(&self) -> &[u32] { &[/* ... */] }
}
```

The literal WGSL array lengths must equal `q()`, every native slice must have
the same length, each `opp_array()[i]` must point to the opposite direction,
and the weights should sum to one. Reflection strings are required when the
lattice is used with the corresponding free-slip boundary.

## 2. Export the module

Add the module to the dimensional `lattice/mod.rs`:

```rust
pub mod my_lattice;
```

## 3. Remove fixed-Q assumptions

The compiler is lattice-driven, but buffer allocation and some optimized shader
paths currently assume D2Q9 or D3Q19. Before using a different `q`, audit:

- the `q` passed to `SimBuffers2D::new` or `SimBuffers3D::new`;
- fixed-size WGSL local arrays and loops in shader templates;
- collision models that explicitly assume 9 or 19 directions;
- free-slip reflection tables;
- benchmark byte accounting and tests.

Adding a module alone is therefore not enough for D3Q27 today. Keep these
couplings explicit until the solver derives every allocation and template size
from the selected lattice.

## 4. Validate

Add unit tests for array lengths, opposite pairs, direction symmetry, and weight
sum. Then run:

```powershell
cargo test -p rust_fluid --lib
cargo run --release -p rust_fluid --bin diagnose_3d -- --suite smoke
```

For a new 3D lattice, also compare a headless equilibrium case in FP32 before
testing FP16 storage or complex boundaries.
