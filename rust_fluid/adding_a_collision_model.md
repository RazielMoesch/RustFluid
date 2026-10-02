# Adding a collision model

Collision components provide the local WGSL operation applied after streaming.
They receive density, momentum, the streamed populations, lattice constants,
the configured relaxation value, and body-force constants from the surrounding
shader template.

## 1. Implement the trait

Create a module under `src/sim/d2/collision/` or `src/sim/d3/collision/` and
implement the matching trait.

```rust
use super::Collision3D;

pub struct MyCollision;

impl MyCollision {
    pub fn new() -> Self { Self }
}

impl Collision3D for MyCollision {
    fn name(&self) -> &'static str { "My collision" }

    fn wgsl(&self) -> &'static str {
        r#"
    let force = vec3<f32>(FORCE_X, FORCE_Y, FORCE_Z);
    u = (u + 0.5 * force) / rho;

    for (var i: u32 = 0u; i < Q; i += 1u) {
        // Compute equilibrium/forcing and update f_local[i].
    }
"#
    }
}
```

The injected block may use `rho`, `u`, `f_local`, the selected lattice
constants (`Q`, directions, weights, and opposite map), `OMEGA`, and the three
body-force constants. `u` contains accumulated momentum on entry; the model is
responsible for converting it to macroscopic velocity.

Follow the existing BGK implementation when adding forcing so the half-force
velocity correction and Guo source term remain consistent. Guard divisions or
square roots where the model can encounter invalid density.

## 2. Export and select it

Export the module from the dimensional `collision/mod.rs`, construct the model,
and pass it to `Lbm2D::new` or `Lbm3D::new`:

```rust
pub mod my_collision;

let collision = MyCollision::new();
let solver = Lbm3D::new(
    device,
    config,
    precision,
    &lattice,
    &collision,
    &boundaries,
);
```

The trait object is used only while assembling shader source; it is not called
once per lattice cell at runtime.

## 3. Check lattice coupling

Prefer loops bounded by `Q`. If the algorithm requires a specific moment basis
or direction order, document that requirement and validate the selected lattice
before compiling. The existing MRT and TRT implementations are examples of
models with stronger lattice coupling; Smagorinsky shows a local effective
relaxation rate built on the BGK equilibrium.

## 4. Validate

At minimum, test generated WGSL for required operations and run equilibrium,
closed-box, and Taylor-Green diagnostics in FP32. Test FP16S separately because
population quantization can expose stability problems hidden in FP32.

```powershell
cargo test -p rust_fluid --lib
cargo run --release -p rust_fluid --bin diagnose_3d -- --suite smoke
```
