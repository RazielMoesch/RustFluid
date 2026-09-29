pub mod loader;
pub mod setup_renderer;

pub struct SimDomain2D {
    pub w: u32,
    pub h: u32,
    pub edge_1_type: u32, // left
    pub edge_2_type: u32, // top
    pub edge_3_type: u32, // bottom
    pub edge_4_type: u32, // right
    pub edge_1_bc: u32,
    pub edge_2_bc: u32,
    pub edge_3_bc: u32,
    pub edge_4_bc: u32,
    pub bcs: Vec<f32>,
}

impl SimDomain2D {
    pub fn new() -> Self {
        Self {
            w: 640,
            h: 360,
            edge_1_type: 1,
            edge_2_type: 1,
            edge_3_type: 1,
            edge_4_type: 1,
            edge_1_bc: 0,
            edge_2_bc: 0,
            edge_3_bc: 0,
            edge_4_bc: 0,
            bcs: vec![0.0f32; 256 * 4],
        }
    }

    pub fn flags(&self) -> Vec<u32> {
        let mut flags = vec![0 as u32; (self.w * self.h) as usize];

        for y in 0..self.h {
            for x in 0..self.w {
                let idx = (x + y * self.w) as usize;

                if y == 0 {
                    flags[idx] = (self.edge_2_type << 24) | self.edge_2_bc;
                } else if y == self.h - 1 {
                    flags[idx] = (self.edge_3_type << 24) | self.edge_3_bc;
                } else if x == 0 {
                    flags[idx] = (self.edge_1_type << 24) | self.edge_1_bc;
                } else if x == self.w - 1 {
                    flags[idx] = (self.edge_4_type << 24) | self.edge_4_bc;
                }
            }
        }

        flags
    }

    pub fn with_w(mut self, w: u32) -> Self {
        self.w = w;
        self
    }

    pub fn with_h(mut self, h: u32) -> Self {
        self.h = h;
        self
    }

    pub fn with_edge_type(mut self, edge: u32, edge_type: u32, boundary_config: u32) -> Self {
        match edge {
            1 => {
                self.edge_1_type = edge_type;
                self.edge_1_bc = boundary_config;
            }
            2 => {
                self.edge_2_type = edge_type;
                self.edge_2_bc = boundary_config;
            }
            3 => {
                self.edge_3_type = edge_type;
                self.edge_3_bc = boundary_config;
            }
            4 => {
                self.edge_4_type = edge_type;
                self.edge_4_bc = boundary_config;
            }
            _ => {
                return self;
            }
        }
        self
    }

    pub fn with_bc(mut self, bc_idx: u32, values: [f32; 4]) -> Self {
        let idx = (bc_idx * 4) as usize;
        if idx + 3 < self.bcs.len() {
            self.bcs[idx] = values[0];
            self.bcs[idx + 1] = values[1];
            self.bcs[idx + 2] = values[2];
            self.bcs[idx + 3] = values[3];
        }
        self
    }
}

pub struct SimDomain3D {
    pub w: u32,
    pub h: u32,
    pub d: u32,
    pub edge_1_type: u32, // left
    pub edge_2_type: u32, // top
    pub edge_3_type: u32, // bottom
    pub edge_4_type: u32, // right
    pub edge_5_type: u32, // front
    pub edge_6_type: u32, // back
    pub edge_1_bc: u32,
    pub edge_2_bc: u32,
    pub edge_3_bc: u32,
    pub edge_4_bc: u32,
    pub edge_5_bc: u32,
    pub edge_6_bc: u32,
    pub bcs: Vec<f32>,
}

impl SimDomain3D {
    pub fn new() -> Self {
        Self {
            w: 64,
            h: 36,
            d: 36,
            edge_1_type: 1,
            edge_2_type: 1,
            edge_3_type: 1,
            edge_4_type: 1,
            edge_5_type: 1,
            edge_6_type: 1,
            edge_1_bc: 0,
            edge_2_bc: 0,
            edge_3_bc: 0,
            edge_4_bc: 0,
            edge_5_bc: 0,
            edge_6_bc: 0,
            bcs: vec![0.0f32; 256 * 4],
        }
    }

    pub fn flags(&self) -> Vec<u32> {
        let mut flags = vec![0 as u32; (self.w * self.h * self.d) as usize];

        for z in 0..self.d {
            for y in 0..self.h {
                for x in 0..self.w {
                    let idx = (x + y * self.w + z * self.w * self.h) as usize;

                    if z == 0 {
                        flags[idx] = (self.edge_5_type << 24) | self.edge_5_bc;
                    } else if z == self.d - 1 {
                        flags[idx] = (self.edge_6_type << 24) | self.edge_6_bc;
                    } else if y == 0 {
                        flags[idx] = (self.edge_2_type << 24) | self.edge_2_bc;
                    } else if y == self.h - 1 {
                        flags[idx] = (self.edge_3_type << 24) | self.edge_3_bc;
                    } else if x == 0 {
                        flags[idx] = (self.edge_1_type << 24) | self.edge_1_bc;
                    } else if x == self.w - 1 {
                        flags[idx] = (self.edge_4_type << 24) | self.edge_4_bc;
                    }
                }
            }
        }

        flags
    }

    pub fn with_w(mut self, w: u32) -> Self {
        self.w = w;
        self
    }

    pub fn with_h(mut self, h: u32) -> Self {
        self.h = h;
        self
    }

    pub fn with_d(mut self, d: u32) -> Self {
        self.d = d;
        self
    }

    pub fn with_edge_type(mut self, edge: u32, edge_type: u32, boundary_config: u32) -> Self {
        match edge {
            1 => {
                self.edge_1_type = edge_type;
                self.edge_1_bc = boundary_config;
            }
            2 => {
                self.edge_2_type = edge_type;
                self.edge_2_bc = boundary_config;
            }
            3 => {
                self.edge_3_type = edge_type;
                self.edge_3_bc = boundary_config;
            }
            4 => {
                self.edge_4_type = edge_type;
                self.edge_4_bc = boundary_config;
            }
            5 => {
                self.edge_5_type = edge_type;
                self.edge_5_bc = boundary_config;
            }
            6 => {
                self.edge_6_type = edge_type;
                self.edge_6_bc = boundary_config;
            }
            _ => {
                return self;
            }
        }
        self
    }

    pub fn with_bc(mut self, bc_idx: u32, values: [f32; 4]) -> Self {
        let idx = (bc_idx * 4) as usize;
        if idx + 3 < self.bcs.len() {
            self.bcs[idx] = values[0];
            self.bcs[idx + 1] = values[1];
            self.bcs[idx + 2] = values[2];
            self.bcs[idx + 3] = values[3];
        }
        self
    }
}
