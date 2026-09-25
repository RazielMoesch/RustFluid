use crate::{
    gpu::utils::{bgl_storage_entry, create_bgl}, sim::lattices::components_2d::{BASE_EXTRACT_2D, BASE_INIT_2D, BASE_STEP_EVEN_2D, BASE_STEP_ODD_2D, BGK_COLLISION, D2Q9_EX, D2Q9_EY, D2Q9_OPP, D2Q9_Q, D2Q9_WEIGHTS, FLUID_PULL_STREAMING_EVEN, FLUID_PULL_STREAMING_ODD, INLET_EQUILIBRIUM_EVEN, INLET_EQUILIBRIUM_ODD, MRT_COLLISION, OUTLET_ZERO_GRADIENT_EVEN, OUTLET_ZERO_GRADIENT_ODD, SOLID_BOUNCE_BACK_EVEN, SOLID_BOUNCE_BACK_ODD}};
pub mod components_2d;
pub mod components_3d;


pub enum Lattice2D {
    D2Q9(D2Q9),
}



pub enum FluidBoundary {
    PullStreaming
}

pub enum SolidBoundary {
    BounceBack
}

pub enum InletBoundary {
    Equilibrium
}

pub enum OutletBoundary {
    ZeroGradient
}

pub enum CollisionLogic {
    BGK,
    MRT
}


pub struct D2Q9 {
    pub fluid_boundary: FluidBoundary,
    pub solid_boundary: SolidBoundary,
    pub inlet_boundary: InletBoundary,
    pub outlet_boundary: OutletBoundary,
    pub collision_logic: CollisionLogic,
}

impl D2Q9 {

    pub fn new() -> Self {

        Self {
            fluid_boundary: FluidBoundary::PullStreaming,
            solid_boundary: SolidBoundary::BounceBack,
            inlet_boundary: InletBoundary::Equilibrium,
            outlet_boundary: OutletBoundary::ZeroGradient,
            collision_logic: CollisionLogic::BGK,
        }
    }

    pub fn bgls(&self, device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout, wgpu::BindGroupLayout) {

        let mut entries = vec![
            bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, false),
            bgl_storage_entry(2, wgpu::ShaderStages::COMPUTE, true)
        ];

        let init_bgl = create_bgl(device, &entries);

        entries[0] = bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, true);
        entries[1] = bgl_storage_entry(2, wgpu::ShaderStages::COMPUTE, true);
        entries.push(
            bgl_storage_entry(1, wgpu::ShaderStages::COMPUTE, false)
        );
        entries.push(
            bgl_storage_entry(3, wgpu::ShaderStages::COMPUTE, true)
        );

        let step_bgl = create_bgl(device, &entries);

        entries.clear();
        entries.push(bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, true));
        entries.push(bgl_storage_entry(1, wgpu::ShaderStages::COMPUTE, false));

        let extract_bgl = create_bgl(device, &entries);

        (init_bgl, step_bgl, extract_bgl)

    }

    pub fn wgsl(&self) -> (String, String, String, String) {


        let init_wgsl = BASE_INIT_2D
            .replace("//{Q}", D2Q9_Q)
            .replace("//{EX}", D2Q9_EX)
            .replace("//{EY}", D2Q9_EY)
            .replace("//{WEIGHTS}", D2Q9_WEIGHTS);


        let step_even_wgsl = BASE_STEP_EVEN_2D
            .replace("//{Q}", D2Q9_Q)
            .replace("//{EX}", D2Q9_EX)
            .replace("//{EY}", D2Q9_EY)
            .replace("//{WEIGHTS}", D2Q9_WEIGHTS)
            .replace("//{OPP}", D2Q9_OPP)
            .replace(
                "//{FLUID_BOUNDARY_LOGIC_EVEN}",
                match self.fluid_boundary {
                    FluidBoundary::PullStreaming => FLUID_PULL_STREAMING_EVEN
                }
            )
            .replace(
                "//{SOLID_BOUNDARY_LOGIC_EVEN}",
                match self.solid_boundary {
                    SolidBoundary::BounceBack => SOLID_BOUNCE_BACK_EVEN
                }
            )
            .replace(
                "//{INLET_BOUNDARY_LOGIC_EVEN}",
                match self.inlet_boundary {
                    InletBoundary::Equilibrium => INLET_EQUILIBRIUM_EVEN
                }
            )
            .replace(
                "//{OUTLET_BOUNDARY_LOGIC_EVEN}",
                match self.outlet_boundary {
                    OutletBoundary::ZeroGradient => OUTLET_ZERO_GRADIENT_EVEN
                }
            )
            .replace(
                "//{COLLISION_LOGIC}",
                match self.collision_logic {
                    CollisionLogic::BGK => {
                        BGK_COLLISION
                    },
                    CollisionLogic::MRT => {
                        MRT_COLLISION
                    }
                }
            );
        
        let step_odd_wgsl = BASE_STEP_ODD_2D
            .replace("//{Q}", D2Q9_Q)
            .replace("//{EX}", D2Q9_EX)
            .replace("//{EY}", D2Q9_EY)
            .replace("//{WEIGHTS}", D2Q9_WEIGHTS)
            .replace("//{OPP}", D2Q9_OPP)
            .replace(
                "//{FLUID_BOUNDARY_LOGIC_ODD}",
                match self.fluid_boundary {
                    FluidBoundary::PullStreaming => FLUID_PULL_STREAMING_ODD
                }
            )
            .replace(
                "//{SOLID_BOUNDARY_LOGIC_ODD}",
                match self.solid_boundary {
                    SolidBoundary::BounceBack => SOLID_BOUNCE_BACK_ODD
                }
            )
            .replace(
                "//{INLET_BOUNDARY_LOGIC_ODD}",
                match self.inlet_boundary {
                    InletBoundary::Equilibrium => INLET_EQUILIBRIUM_ODD
                }
            )
            .replace(
                "//{OUTLET_BOUNDARY_LOGIC_ODD}",
                match self.outlet_boundary {
                    OutletBoundary::ZeroGradient => OUTLET_ZERO_GRADIENT_ODD
                }
            )
            .replace(
                "//{COLLISION_LOGIC}",
                match self.collision_logic {
                    CollisionLogic::BGK => {
                        BGK_COLLISION
                    },
                    CollisionLogic::MRT => {
                        MRT_COLLISION
                    }
                }
            );
        
        let extract_wgsl = BASE_EXTRACT_2D
            .replace("//{Q}", D2Q9_Q)
            .replace("//{EX}", D2Q9_EX)
            .replace("//{EY}", D2Q9_EY);

            ( init_wgsl, step_even_wgsl, step_odd_wgsl, extract_wgsl )

    }

    pub fn with_fluid_boundary(mut self, boundary: FluidBoundary) -> Self {
        self.fluid_boundary = boundary;
        self
    }

    pub fn with_solid_boundary(mut self, boundary: SolidBoundary) -> Self {
        self.solid_boundary = boundary;
        self
    }

    pub fn with_inlet_boundary(mut self, boundary: InletBoundary) -> Self {
        self.inlet_boundary = boundary;
        self
    }

    pub fn with_outlet_boundary(mut self, boundary: OutletBoundary) -> Self {
        self.outlet_boundary = boundary;
        self
    }

    pub fn with_collision_logic(mut self, logic: CollisionLogic) -> Self {
        self.collision_logic = logic;
        self
    }

}

