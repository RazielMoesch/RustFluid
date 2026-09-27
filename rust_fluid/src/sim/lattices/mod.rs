use crate::{
    gpu::utils::{bgl_storage_entry, create_bgl},
    sim::lattices::components_2d::{
        BASE_EXTRACT_2D, BASE_INIT_2D, BASE_STEP_EVEN_2D, BASE_STEP_ODD_2D, BGK_COLLISION, D2Q9_EX,
        D2Q9_EY, D2Q9_OPP, D2Q9_Q, D2Q9_REFLECT_Y, D2Q9_WEIGHTS, FLUID_PULL_STREAMING_EVEN,
        FLUID_PULL_STREAMING_ODD, FREE_SLIP_Y_EVEN, FREE_SLIP_Y_ODD, INLET_EQUILIBRIUM_EVEN,
        INLET_EQUILIBRIUM_ODD, MRT_COLLISION, OUTLET_ZERO_GRADIENT_EVEN, OUTLET_ZERO_GRADIENT_ODD,
        SOLID_BOUNCE_BACK_EVEN, SOLID_BOUNCE_BACK_ODD, ZOU_HE_LEFT_VELOCITY,
        FREE_SLIP_X_EVEN, FREE_SLIP_X_ODD, D2Q9_REFLECT_X,
    },
};

pub mod components_2d;
pub mod components_3d;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    F16Storage,
    Auto,
}

impl Precision {
    pub fn bytes_per_population(self) -> wgpu::BufferAddress {
        match self {
            Precision::F32 | Precision::Auto => 4,
            Precision::F16Storage => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Precision::F32 => "FP32",
            Precision::F16Storage => "FP16 Storage / FP32 Compute",
            Precision::Auto => "Auto",
        }
    }

    pub fn requires_shader_f16(self) -> bool {
        matches!(self, Precision::F16Storage)
    }

    pub fn is_f16_storage(self) -> bool {
        matches!(self, Precision::F16Storage)
    }
}

pub(crate) struct PrecisionConfig {
    pub enable_directive: &'static str,
    pub pop_type: &'static str,
}

impl From<Precision> for PrecisionConfig {
    fn from(p: Precision) -> Self {
        match p {
            Precision::F32 | Precision::Auto => PrecisionConfig {
                enable_directive: "",
                pop_type: "f32",
            },
            Precision::F16Storage => PrecisionConfig {
                enable_directive: "enable f16;\n",
                pop_type: "f16",
            },
        }
    }
}

impl PrecisionConfig {
    pub fn step_helpers(&self) -> String {
        if self.pop_type == "f32" {
            r#"
fn load_fa(index: u32) -> f32 { return fa[index]; }
fn store_fb(index: u32, value: f32) { fb[index] = value; }
"#
            .to_string()
        } else {
            format!(
                r#"
fn load_fa(index: u32) -> f32 {{ return f32(fa[index]); }}
fn store_fb(index: u32, value: f32) {{ fb[index] = {t}(value); }}
"#,
                t = self.pop_type
            )
        }
    }

    pub fn init_helpers(&self) -> String {
        if self.pop_type == "f32" {
            r#"
fn store_fa(index: u32, value: f32) { fa[index] = value; }
"#
            .to_string()
        } else {
            format!(
                r#"
fn store_fa(index: u32, value: f32) {{ fa[index] = {t}(value); }}
"#,
                t = self.pop_type
            )
        }
    }

    pub fn extract_helpers(&self) -> String {
        if self.pop_type == "f32" {
            r#"
fn load_fa(index: u32) -> f32 { return fa[index]; }
"#
            .to_string()
        } else {
            r#"
fn load_fa(index: u32) -> f32 { return f32(fa[index]); }
"#
            .to_string()
        }
    }
}

pub struct CompiledLattice2D {
    pub q: u32,
    pub bytes_per_population: wgpu::BufferAddress,
    pub precision: Precision,
    pub lattice_name: &'static str,
    pub collision_name: &'static str,

    pub init_bgl: wgpu::BindGroupLayout,
    pub step_bgl: wgpu::BindGroupLayout,
    pub extract_bgl: wgpu::BindGroupLayout,

    pub init_wgsl: String,
    pub step_even_wgsl: String,
    pub step_odd_wgsl: String,
    pub extract_wgsl: String,
}

pub enum Lattice2D {
    D2Q9(D2Q9),
}

impl Lattice2D {
    pub fn compile(&self, device: &wgpu::Device) -> CompiledLattice2D {
        match self {
            Self::D2Q9(lattice) => lattice.compile(device),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionLogic {
    BGK,
    MRT,
}

impl CollisionLogic {
    pub fn label(self) -> &'static str {
        match self {
            CollisionLogic::BGK => "BGK",
            CollisionLogic::MRT => "MRT",
        }
    }
}

pub struct D2Q9 {
    pub collision_logic: CollisionLogic,
    pub precision: Precision,
}

impl D2Q9 {
    pub fn new() -> Self {
        Self {
            collision_logic: CollisionLogic::BGK,
            precision: Precision::F32,
        }
    }

    pub fn compile(&self, device: &wgpu::Device) -> CompiledLattice2D {
        let (init_bgl, step_bgl, extract_bgl) = self.bgls(device);

        let (init_wgsl, step_even_wgsl, step_odd_wgsl, extract_wgsl) = self.wgsl();

        CompiledLattice2D {
            q: 9,
            bytes_per_population: self.precision.bytes_per_population(),
            precision: self.precision,
            lattice_name: "D2Q9",
            collision_name: self.collision_logic.label(),

            init_bgl,
            step_bgl,
            extract_bgl,

            init_wgsl,
            step_even_wgsl,
            step_odd_wgsl,
            extract_wgsl,
        }
    }

    pub fn bgls(
        &self,
        device: &wgpu::Device,
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
    ) {
        let mut entries = vec![
            bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, false),
            bgl_storage_entry(2, wgpu::ShaderStages::COMPUTE, true),
        ];

        let init_bgl = create_bgl(device, &entries);

        entries[0] = bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, true);

        entries[1] = bgl_storage_entry(2, wgpu::ShaderStages::COMPUTE, true);

        entries.push(bgl_storage_entry(1, wgpu::ShaderStages::COMPUTE, false));

        entries.push(bgl_storage_entry(3, wgpu::ShaderStages::COMPUTE, true));

        let step_bgl = create_bgl(device, &entries);

        entries.clear();

        entries.push(bgl_storage_entry(0, wgpu::ShaderStages::COMPUTE, true));

        entries.push(bgl_storage_entry(1, wgpu::ShaderStages::COMPUTE, false));

        let extract_bgl = create_bgl(device, &entries);

        (init_bgl, step_bgl, extract_bgl)
    }

    pub fn wgsl(&self) -> (String, String, String, String) {
        let pc = PrecisionConfig::from(self.precision);
        let pop_type = pc.pop_type;

        let init_wgsl = format!("{}", pc.enable_directive)
            + &BASE_INIT_2D
                .replace("//{Q}", D2Q9_Q)
                .replace("//{EX}", D2Q9_EX)
                .replace("//{EY}", D2Q9_EY)
                .replace("//{WEIGHTS}", D2Q9_WEIGHTS)
                .replace(
                    "array<f32>; // POP_STORAGE",
                    &format!("array<{}>;", pop_type),
                )
                .replace("//{PRECISION_HELPERS}", &pc.init_helpers());

        let step_even_wgsl = format!("{}", pc.enable_directive)
            + &BASE_STEP_EVEN_2D
                .replace("//{Q}", D2Q9_Q)
                .replace("//{EX}", D2Q9_EX)
                .replace("//{EY}", D2Q9_EY)
                .replace("//{WEIGHTS}", D2Q9_WEIGHTS)
                .replace("//{OPP}", D2Q9_OPP)
                .replace("//{FLUID_PULL_LOGIC_EVEN}", FLUID_PULL_STREAMING_EVEN)
                .replace("//{BOUNCE_BACK_LOGIC_EVEN}", SOLID_BOUNCE_BACK_EVEN)
                .replace("//{EQUILIBRIUM_INLET_LOGIC_EVEN}", INLET_EQUILIBRIUM_EVEN)
                .replace(
                    "//{ZERO_GRADIENT_OUTLET_LOGIC_EVEN}",
                    OUTLET_ZERO_GRADIENT_EVEN,
                )
                .replace("//{FREE_SLIP_Y_LOGIC_EVEN}", FREE_SLIP_Y_EVEN)
                .replace("//{FREE_SLIP_X_LOGIC_EVEN}", FREE_SLIP_X_EVEN)
                .replace(
                    "//{COLLISION_LOGIC}",
                    match self.collision_logic {
                        CollisionLogic::BGK => BGK_COLLISION,
                        CollisionLogic::MRT => MRT_COLLISION,
                    },
                )
                .replace("//{REFLECT_Y}", D2Q9_REFLECT_Y)
                .replace("//{REFLECT_X}", D2Q9_REFLECT_X)
                .replace("//{POST_STREAMING_CORRECTION}", ZOU_HE_LEFT_VELOCITY)
                .replace("array<f32>; // POP_FA", &format!("array<{}>;", pop_type))
                .replace("array<f32>; // POP_FB", &format!("array<{}>;", pop_type))
                .replace("//{PRECISION_HELPERS}", &pc.step_helpers());

        let step_odd_wgsl = format!("{}", pc.enable_directive)
            + &BASE_STEP_ODD_2D
                .replace("//{Q}", D2Q9_Q)
                .replace("//{EX}", D2Q9_EX)
                .replace("//{EY}", D2Q9_EY)
                .replace("//{WEIGHTS}", D2Q9_WEIGHTS)
                .replace("//{OPP}", D2Q9_OPP)
                .replace("//{FLUID_PULL_LOGIC_ODD}", FLUID_PULL_STREAMING_ODD)
                .replace("//{BOUNCE_BACK_LOGIC_ODD}", SOLID_BOUNCE_BACK_ODD)
                .replace("//{EQUILIBRIUM_INLET_LOGIC_ODD}", INLET_EQUILIBRIUM_ODD)
                .replace(
                    "//{ZERO_GRADIENT_OUTLET_LOGIC_ODD}",
                    OUTLET_ZERO_GRADIENT_ODD,
                )
                .replace("//{FREE_SLIP_Y_LOGIC_ODD}", FREE_SLIP_Y_ODD)
                .replace("//{FREE_SLIP_X_LOGIC_ODD}", FREE_SLIP_X_ODD)
                .replace(
                    "//{COLLISION_LOGIC}",
                    match self.collision_logic {
                        CollisionLogic::BGK => BGK_COLLISION,
                        CollisionLogic::MRT => MRT_COLLISION,
                    },
                )
                .replace("//{REFLECT_Y}", D2Q9_REFLECT_Y)
                .replace("//{REFLECT_X}", D2Q9_REFLECT_X)
                .replace("//{POST_STREAMING_CORRECTION}", ZOU_HE_LEFT_VELOCITY)
                .replace("array<f32>; // POP_FA", &format!("array<{}>;", pop_type))
                .replace("array<f32>; // POP_FB", &format!("array<{}>;", pop_type))
                .replace("//{PRECISION_HELPERS}", &pc.step_helpers());

        let extract_wgsl = format!("{}", pc.enable_directive)
            + &BASE_EXTRACT_2D
                .replace("//{Q}", D2Q9_Q)
                .replace("//{EX}", D2Q9_EX)
                .replace("//{EY}", D2Q9_EY)
                .replace("//{OPP}", D2Q9_OPP)
                .replace("array<f32>; // POP_FA", &format!("array<{}>;", pop_type))
                .replace("//{PRECISION_HELPERS}", &pc.extract_helpers());

        (init_wgsl, step_even_wgsl, step_odd_wgsl, extract_wgsl)
    }

    pub fn with_collision_logic(mut self, logic: CollisionLogic) -> Self {
        self.collision_logic = logic;
        self
    }

    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }
}
