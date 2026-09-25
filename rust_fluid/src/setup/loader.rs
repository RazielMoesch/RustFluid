use resvg::tiny_skia::{Pixmap, Transform};
use resvg::usvg::{Options, Tree};
use std::fs;
use std::path::{Path, PathBuf};

pub struct VoxelGrid2D {
    pub xpos: i32,
    pub ypos: i32,
    pub width: u32,
    pub height: u32,
    pub data: Vec<bool>,
    pub path: PathBuf,
}

pub struct VoxelGrid3D {
    pub xpos: i32,
    pub ypos: i32,
    pub zpos: i32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub data: Vec<bool>,
    pub path: PathBuf
}

impl VoxelGrid2D {

    pub fn new( width: u32, height: u32, data: Vec<bool>, path: PathBuf ) -> Self {
        Self {
            xpos: 0,
            ypos: 0,
            width,
            height,
            data,
            path,
        }
    }

    pub fn is_solid(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false
        }
        self.data[(y * self.width + x) as usize]
    }

    pub fn scale(&mut self, factor: f32) -> Result<(), String> {
        let new_width = (self.width as f32 * factor).round() as u32;
        let new_height = (self.height as f32 * factor).round() as u32;
        let new_grid = Loader::load_svg(&self.path, new_width, new_height)?;
        self.width = new_grid.width;
        self.height = new_grid.height;
        self.data = new_grid.data;
        Ok(())
    }

    pub fn translate(&mut self, x: i32, y: i32) {
        self.xpos += x;
        self.ypos += y;
    }

}

impl VoxelGrid3D {
    pub fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        if x >= self.width || y >= self.height || z >= self.depth {
            return false
        }
        self.data[(z * self.width * self.height + y * self.width + x) as usize]
    }
    
    pub fn resize(&self, target_width: u32, target_height: u32, target_depth: u32) -> Result<Self, String> {
        Loader::load_stl(&self.path, target_width, target_height, target_depth)
    }

    pub fn translate(&mut self, x: i32, y: i32, z: i32) {
        self.xpos += x;
        self.ypos += y;
        self.zpos += z;
    }

}

pub struct Loader;

impl Loader {

     pub fn load_svg<P: AsRef<Path>>(
        path: P,
        target_width: u32,
        target_height: u32,
     ) -> Result<VoxelGrid2D, String> {

        let path = path.as_ref();
        let svg_data = fs::read(path).map_err(|e| format!("File Read Error: {}", e))?;

        let opt = Options::default();
        let tree = Tree::from_data(&svg_data, &opt)
            .map_err(|e| format!("SVG Parse Error: {}", e))?;

        let svg_size = tree.size();
        let scale_x = target_width as f32 / svg_size.width();
        let scale_y = target_height as f32 / svg_size.height();
        let scale = scale_x.min(scale_y);

        let final_width = (svg_size.width() * scale).round() as u32;
        let final_height = (svg_size.height() * scale).round() as u32;
        
        let final_width = final_width.max(1);
        let final_height = final_height.max(1);

        let mut pixmap = Pixmap::new(final_width, final_height)
            .ok_or("Failed to allocate pixmap buffer".to_string())?;

        let transform = Transform::from_scale(scale, scale);

        resvg::render(&tree, transform, &mut pixmap.as_mut());

        let pixels = pixmap.pixels();
        let mut data = Vec::with_capacity((final_width * final_height) as usize);

        for pixel in pixels {
            let is_solid = pixel.alpha() > 127;
            data.push(is_solid);
        }

        Ok(VoxelGrid2D { xpos: 0, ypos: 0, width: final_width, height: final_height, data, path: path.to_path_buf() })
    }  
    
    
    pub fn load_stl<P: AsRef<Path>>(
        path: P,
        target_width: u32,
        target_height: u32,
        target_depth: u32,
    ) -> Result<VoxelGrid3D, String> {
        let path = path.as_ref();
        
        let mut file = std::fs::OpenOptions::new().read(true).open(path)
            .map_err(|e| format!("File open error: {}", e))?;
        let mesh = stl_io::read_stl(&mut file)
            .map_err(|e| format!("STL Parse error: {}", e))?;

        let mut min_x = f32::MAX; let mut max_x = f32::MIN;
        let mut min_y = f32::MAX; let mut max_y = f32::MIN;
        let mut min_z = f32::MAX; let mut max_z = f32::MIN;

        for v in &mesh.vertices {
            min_x = min_x.min(v[0]); max_x = max_x.max(v[0]);
            min_y = min_y.min(v[1]); max_y = max_y.max(v[1]);
            min_z = min_z.min(v[2]); max_z = max_z.max(v[2]);
        }

        let size_x = max_x - min_x;
        let size_y = max_y - min_y;
        let size_z = max_z - min_z;

        let scale_x = target_width as f32 / size_x;
        let scale_y = target_height as f32 / size_y;
        let scale_z = target_depth as f32 / size_z;
        let scale = scale_x.min(scale_y).min(scale_z);

        let dx = (target_width as f32 - size_x * scale) / 2.0;
        let dy = (target_height as f32 - size_y * scale) / 2.0;
        let dz = (target_depth as f32 - size_z * scale) / 2.0;

        let data_size = (target_width * target_height * target_depth) as usize;
        let mut data = vec![false; data_size];

        let distance = |v1: [f32; 3], v2: [f32; 3]| -> f32 {
            ((v1[0] - v2[0]).powi(2) + (v1[1] - v2[1]).powi(2) + (v1[2] - v2[2]).powi(2)).sqrt()
        };

        for face in &mesh.faces {
            let v0_orig = mesh.vertices[face.vertices[0]];
            let v1_orig = mesh.vertices[face.vertices[1]];
            let v2_orig = mesh.vertices[face.vertices[2]];

            let transform = |v: stl_io::Vertex| -> [f32; 3] {
                [
                    (v[0] - min_x) * scale + dx,
                    (v[1] - min_y) * scale + dy,
                    (v[2] - min_z) * scale + dz,
                ]
            };

            let v0 = transform(v0_orig);
            let v1 = transform(v1_orig);
            let v2 = transform(v2_orig);

            let mut max_dist = 0.0_f32;
            max_dist = max_dist.max(distance(v0, v1));
            max_dist = max_dist.max(distance(v1, v2));
            max_dist = max_dist.max(distance(v2, v0));
            
            let steps = (max_dist * 2.0).ceil() as usize;
            let steps = steps.max(1);

            for i in 0..=steps {
                for j in 0..=(steps - i) {
                    let u = i as f32 / steps as f32;
                    let v = j as f32 / steps as f32;
                    let w = 1.0 - u - v;

                    let px = v0[0] * u + v1[0] * v + v2[0] * w;
                    let py = v0[1] * u + v1[1] * v + v2[1] * w;
                    let pz = v0[2] * u + v1[2] * v + v2[2] * w;

                    let vx = px.round() as i32;
                    let vy = py.round() as i32;
                    let vz = pz.round() as i32;

                    if vx >= 0 && vx < target_width as i32 &&
                       vy >= 0 && vy < target_height as i32 &&
                       vz >= 0 && vz < target_depth as i32 {
                        let idx = (vz as u32 * target_width * target_height) + 
                                  (vy as u32 * target_width) + 
                                  vx as u32;
                        data[idx as usize] = true;
                    }
                }
            }
        }
        
        Ok(VoxelGrid3D {
            xpos: 0,
            ypos: 0,
            zpos: 0,
            width: target_width,
            height: target_height,
            depth: target_depth,
            data,
            path: path.to_path_buf(),
        })
    }

}
