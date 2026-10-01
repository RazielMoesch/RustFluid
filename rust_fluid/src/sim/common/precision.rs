#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    FP16S,
    Auto,
}

impl Precision {
    pub fn bytes_per_population(self) -> wgpu::BufferAddress {
        match self {
            Precision::F32 | Precision::Auto => 4,
            Precision::FP16S => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Precision::F32 => "FP32",
            Precision::FP16S => "FP16S",
            Precision::Auto => "Auto",
        }
    }

    pub fn requires_shader_f16(self) -> bool {
        matches!(self, Precision::FP16S)
    }

    pub fn is_f16_storage(self) -> bool {
        matches!(self, Precision::FP16S)
    }
}

pub struct PrecisionConfig {
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
            Precision::FP16S => PrecisionConfig {
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
            r#"
fn load_fa(index: u32) -> f32 {
    let dir = index / TOTAL_CELLS;
    return f32(fa[index]) / 32768.0 + WEIGHTS[dir];
}
fn store_fb(index: u32, value: f32) {
    let dir = index / TOTAL_CELLS;
    fb[index] = f16((value - WEIGHTS[dir]) * 32768.0);
}
"#
            .to_string()
        }
    }

    pub fn init_helpers(&self) -> String {
        if self.pop_type == "f32" {
            r#"
fn store_fa(index: u32, value: f32) { fa[index] = value; }
"#
            .to_string()
        } else {
            r#"
fn store_fa(index: u32, value: f32) {
    let dir = index / TOTAL_CELLS;
    fa[index] = f16((value - WEIGHTS[dir]) * 32768.0);
}
"#
            .to_string()
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
fn load_fa(index: u32) -> f32 {
    let dir = index / TOTAL_CELLS;
    return f32(fa[index]) / 32768.0 + WEIGHTS[dir];
}
"#
            .to_string()
        }
    }
}
