# Adding a New Lattice

Lattices define the discrete velocity directions and equilibrium weights for the simulation. Thanks to the new architecture, adding a new lattice (like D3Q27) is incredibly simple and requires modifying exactly zero existing files.

## Step 1: Create a new file
Create a new file in `src/sim/d3/lattice/d3q27.rs` (or `d2` for 2D lattices).

## Step 2: Implement the Trait
Implement the `Lattice3D` (or `Lattice2D`) trait. The trait requires returning the dimension `q` and string arrays for the WGSL shader, as well as native Rust arrays for CPU-side shader unrolling.

```rust
use super::Lattice3D;

pub struct D3Q27;

impl D3Q27 {
    pub fn new() -> Self { Self }
}

impl Lattice3D for D3Q27 {
    fn name(&self) -> &'static str { "D3Q27" }
    fn q(&self) -> u32 { 27 }

    // Provide the WGSL strings. 
    fn ex(&self) -> &'static str {
        r#"
const EX = array<i32, 27>( ... );
"#
    }

    fn ey(&self) -> &'static str { /* ... */ }
    fn ez(&self) -> &'static str { /* ... */ }
    fn weights(&self) -> &'static str { /* ... */ }
    fn opp(&self) -> &'static str { /* ... */ }

    // Optionally provide reflection indices if your lattice supports free-slip boundaries
    fn reflect_x(&self) -> Option<&'static str> { Some(...) }

    // Finally, provide the exact same array data for Rust to use when generating the fast path unrolling!
    fn ex_array(&self) -> &[i32] { &[ ... ] }
    fn ey_array(&self) -> &[i32] { &[ ... ] }
    fn ez_array(&self) -> &[i32] { &[ ... ] }
    fn opp_array(&self) -> &[u32] { &[ ... ] }
}
```

## Step 3: Export and Use
Export your new lattice in `src/sim/d3/lattice/mod.rs`:
```rust
pub mod d3q27;
```

You can now immediately use it when constructing an `Lbm3D` instance:
```rust
let lattice = D3Q27::new();
let solver = Lbm3D::new(
    &device,
    config,
    Precision::F32,
    &lattice,
    &collision,
    &boundaries
);
```

The Shader Compiler handles everything else for you, including completely unrolling the hot loops for your specific `q` count and directional vectors.
