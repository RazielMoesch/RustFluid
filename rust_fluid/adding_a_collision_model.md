# Adding a Collision Model

Collision models dictate how populations relax toward equilibrium. You can implement custom models like MRT, Cumulant, or Entropic LBM easily.

## Step 1: Create a new file
Create a new file in `src/sim/d3/collision/my_model.rs` (or `d2`).

## Step 2: Implement the Trait
Implement `Collision3D` (or `Collision2D`).

```rust
use super::Collision3D;

pub struct Entropic;

impl Entropic {
    pub fn new() -> Self { Self }
}

impl Collision3D for Entropic {
    fn name(&self) -> &'static str {
        "Entropic"
    }

    fn wgsl(&self) -> &'static str {
        r#"
    // You have access to:
    // `rho` (f32) - Density
    // `u` (vec3<f32>) - Macroscopic velocity
    // `f_local` (array<f32, Q>) - The local populations after streaming
    // `OMEGA` (f32) - The user's requested relaxation parameter

    if (rho > 0.0) {
        u = u / rho;
    }

    // Calculate your custom collision logic here...
    
    // Modify `f_local` in place:
    for (var i: u32 = 0u; i < Q; i += 1u) {
        // ...
        f_local[i] = ...;
    }
"#
    }
}
```

## Step 3: Export and Use
Export the model in `src/sim/d3/collision/mod.rs`:
```rust
pub mod entropic;
```

Pass it to the solver:
```rust
let collision = Entropic::new();
let solver = Lbm3D::new(..., &collision, ...);
```

The collision string is safely injected right after the streaming phase and before the final populations are written to VRAM.
