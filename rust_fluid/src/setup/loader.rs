//! Converts SVG and STL assets into solver flags and display geometry.

use resvg::tiny_skia::{Pixmap, Transform};
use resvg::usvg::{Options, Tree};
use std::fs;
use std::path::{Path, PathBuf};

/// Positioned two-dimensional solid mask rasterized from an SVG.
pub struct VoxelGrid2D {
    pub xpos: i32,
    pub ypos: i32,
    pub width: u32,
    pub height: u32,
    pub rotation_deg: f32,
    pub data: Vec<bool>,
    pub path: PathBuf,
}

/// Positioned three-dimensional solid mask plus a welded display mesh.
pub struct VoxelGrid3D {
    pub xpos: i32,
    pub ypos: i32,
    pub zpos: i32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub data: Vec<bool>,
    pub path: PathBuf,
    pub mesh_vertices: Vec<[f32; 3]>,
    pub mesh_normals: Vec<[f32; 3]>,
    pub mesh_indices: Vec<u32>,
}

impl VoxelGrid2D {
    pub fn new(width: u32, height: u32, data: Vec<bool>, path: PathBuf) -> Self {
        Self {
            xpos: 0,
            ypos: 0,
            width,
            height,
            rotation_deg: 0.0,
            data,
            path,
        }
    }

    pub fn is_solid(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        self.data[(y * self.width + x) as usize]
    }

    pub fn scale(&mut self, factor: f32) -> Result<(), String> {
        let new_width = (self.width as f32 * factor).round() as u32;
        let new_height = (self.height as f32 * factor).round() as u32;
        let new_grid = Loader::load_svg(&self.path, new_width, new_height, self.rotation_deg)?;
        self.width = new_grid.width;
        self.height = new_grid.height;
        self.data = new_grid.data;
        Ok(())
    }

    pub fn translate(&mut self, x: i32, y: i32) {
        self.xpos += x;
        self.ypos += y;
    }

    pub fn rotate(&mut self, rotation_deg: f32) -> Result<(), String> {
        // Recover the pre-rotation target box so repeated rotations do not
        // progressively inflate the raster dimensions.
        let rad = self.rotation_deg.to_radians();
        let cos_r = rad.cos().abs();
        let sin_r = rad.sin().abs();

        let denom = cos_r * cos_r - sin_r * sin_r;
        let orig_width = if denom.abs() > 1e-4 {
            ((self.width as f32 * cos_r - self.height as f32 * sin_r) / denom).max(1.0)
        } else {
            self.width as f32
        };
        let orig_height = if denom.abs() > 1e-4 {
            ((self.height as f32 * cos_r - self.width as f32 * sin_r) / denom).max(1.0)
        } else {
            self.height as f32
        };

        self.rotation_deg += rotation_deg;
        let new_grid = Loader::load_svg(
            &self.path,
            orig_width.round() as u32,
            orig_height.round() as u32,
            self.rotation_deg,
        )?;
        self.width = new_grid.width;
        self.height = new_grid.height;
        self.data = new_grid.data;
        Ok(())
    }
}

impl VoxelGrid3D {
    pub fn is_solid(&self, x: u32, y: u32, z: u32) -> bool {
        if x >= self.width || y >= self.height || z >= self.depth {
            return false;
        }
        self.data[(z * self.width * self.height + y * self.width + x) as usize]
    }

    pub fn resize(
        &self,
        target_width: u32,
        target_height: u32,
        target_depth: u32,
    ) -> Result<Self, String> {
        Loader::load_stl(
            &self.path,
            target_width,
            target_height,
            target_depth,
            0.0,
            0.0,
            0.0,
        )
    }

    pub fn translate(&mut self, x: i32, y: i32, z: i32) {
        self.xpos += x;
        self.ypos += y;
        self.zpos += z;
    }
}

/// Stateless entry point for SVG rasterization and STL voxelization.
pub struct Loader;

impl Loader {
    /// Rasterizes an SVG into a rotated, aspect-preserving boolean mask.
    pub fn load_svg<P: AsRef<Path>>(
        path: P,
        target_width: u32,
        target_height: u32,
        rotation_deg: f32,
    ) -> Result<VoxelGrid2D, String> {
        let path = path.as_ref();
        let svg_data = fs::read(path).map_err(|e| format!("File Read Error: {}", e))?;

        let opt = Options::default();
        let tree =
            Tree::from_data(&svg_data, &opt).map_err(|e| format!("SVG Parse Error: {}", e))?;

        let svg_size = tree.size();
        let scale_x = target_width as f32 / svg_size.width();
        let scale_y = target_height as f32 / svg_size.height();
        let scale = scale_x.min(scale_y);

        let base_width = svg_size.width() * scale;
        let base_height = svg_size.height() * scale;

        let rad = rotation_deg.to_radians();
        let cos_r = rad.cos().abs();
        let sin_r = rad.sin().abs();

        let final_width = (base_width * cos_r + base_height * sin_r).round() as u32;
        let final_height = (base_width * sin_r + base_height * cos_r).round() as u32;

        let final_width = final_width.max(1);
        let final_height = final_height.max(1);

        let mut pixmap = Pixmap::new(final_width, final_height)
            .ok_or("Failed to allocate pixmap buffer".to_string())?;

        let cx = svg_size.width() / 2.0;
        let cy = svg_size.height() / 2.0;

        let rot_cx = final_width as f32 / 2.0;
        let rot_cy = final_height as f32 / 2.0;

        let sx = scale * rad.cos();
        let ky = scale * rad.sin();
        let kx = -scale * rad.sin();
        let sy = scale * rad.cos();

        let tx = -cx * sx - cy * kx + rot_cx;
        let ty = -cx * ky - cy * sy + rot_cy;

        let transform = Transform::from_row(sx, ky, kx, sy, tx, ty);

        resvg::render(&tree, transform, &mut pixmap.as_mut());

        let pixels = pixmap.pixels();
        let mut data = Vec::with_capacity((final_width * final_height) as usize);

        for pixel in pixels {
            let is_solid = pixel.alpha() > 127;
            data.push(is_solid);
        }

        Ok(VoxelGrid2D {
            xpos: 0,
            ypos: 0,
            width: final_width,
            height: final_height,
            rotation_deg,
            data,
            path: path.to_path_buf(),
        })
    }

    /// Rotates and fits an STL, samples its surface, and builds a display mesh.
    pub fn load_stl<P: AsRef<Path>>(
        path: P,
        target_width: u32,
        target_height: u32,
        target_depth: u32,
        rot_x_deg: f32,
        rot_y_deg: f32,
        rot_z_deg: f32,
    ) -> Result<VoxelGrid3D, String> {
        let path = path.as_ref();

        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| format!("File open error: {}", e))?;
        let mesh = stl_io::read_stl(&mut file).map_err(|e| format!("STL Parse error: {}", e))?;

        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;

        for v in &mesh.vertices {
            min_x = min_x.min(v[0]);
            max_x = max_x.max(v[0]);
            min_y = min_y.min(v[1]);
            max_y = max_y.max(v[1]);
            min_z = min_z.min(v[2]);
            max_z = max_z.max(v[2]);
        }

        let cx = (min_x + max_x) / 2.0;
        let cy = (min_y + max_y) / 2.0;
        let cz = (min_z + max_z) / 2.0;

        let rx = rot_x_deg.to_radians();
        let ry = rot_y_deg.to_radians();
        let rz = rot_z_deg.to_radians();

        let cos_x = rx.cos();
        let sin_x = rx.sin();
        let cos_y = ry.cos();
        let sin_y = ry.sin();
        let cos_z = rz.cos();
        let sin_z = rz.sin();

        min_x = f32::MAX;
        max_x = f32::MIN;
        min_y = f32::MAX;
        max_y = f32::MIN;
        min_z = f32::MAX;
        max_z = f32::MIN;

        let mut rotated_vertices = Vec::with_capacity(mesh.vertices.len());
        for v in &mesh.vertices {
            let mut x = v[0] - cx;
            let mut y = v[1] - cy;
            let mut z = v[2] - cz;

            let y1 = y * cos_x - z * sin_x;
            let z1 = y * sin_x + z * cos_x;
            y = y1;
            z = z1;

            let x1 = x * cos_y + z * sin_y;
            let z2 = -x * sin_y + z * cos_y;
            x = x1;
            z = z2;

            let x2 = x * cos_z - y * sin_z;
            let y2 = x * sin_z + y * cos_z;
            x = x2;
            y = y2;

            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            min_z = min_z.min(z);
            max_z = max_z.max(z);

            rotated_vertices.push([x, y, z]);
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

        let mut mesh_vertices = Vec::new();
        let mut mesh_normals = Vec::new();
        let mut mesh_indices = Vec::new();
        let mut index_map = std::collections::HashMap::new();
        let mut triangle_set = std::collections::HashSet::new();

        // The source STL can contain millions of triangles even though the
        // simulation only resolves geometry at lattice scale.  Weld the
        // visualization mesh to half-cell coordinates while retaining the
        // original triangles below for voxelization.  This avoids spending
        // most render time drawing sub-voxel detail that cannot affect the
        // simulation or be seen at the target resolution.
        const RENDER_QUANTIZATION: f32 = 0.5;

        for face in &mesh.faces {
            let v0_orig = rotated_vertices[face.vertices[0]];
            let v1_orig = rotated_vertices[face.vertices[1]];
            let v2_orig = rotated_vertices[face.vertices[2]];

            let transform = |v: [f32; 3]| -> [f32; 3] {
                [
                    (v[0] - min_x) * scale + dx,
                    (v[1] - min_y) * scale + dy,
                    (v[2] - min_z) * scale + dz,
                ]
            };

            let v0 = transform(v0_orig);
            let v1 = transform(v1_orig);
            let v2 = transform(v2_orig);

            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let nx = e1[1] * e2[2] - e1[2] * e2[1];
            let ny = e1[2] * e2[0] - e1[0] * e2[2];
            let nz = e1[0] * e2[1] - e1[1] * e2[0];

            let mut idxs = [0u32; 3];
            for (i, v) in [v0, v1, v2].iter().enumerate() {
                let qx = (v[0] / RENDER_QUANTIZATION).round() as i32;
                let qy = (v[1] / RENDER_QUANTIZATION).round() as i32;
                let qz = (v[2] / RENDER_QUANTIZATION).round() as i32;
                let key = (qx, qy, qz);
                let idx = *index_map.entry(key).or_insert_with(|| {
                    let new_idx = mesh_vertices.len() as u32;
                    mesh_vertices.push([
                        qx as f32 * RENDER_QUANTIZATION,
                        qy as f32 * RENDER_QUANTIZATION,
                        qz as f32 * RENDER_QUANTIZATION,
                    ]);
                    mesh_normals.push([0.0, 0.0, 0.0]);
                    new_idx
                });
                idxs[i] = idx;
            }

            // Quantization collapses many tiny source triangles.  Skip those
            // and remove duplicate triangles before uploading the GPU mesh.
            if idxs[0] != idxs[1] && idxs[1] != idxs[2] && idxs[0] != idxs[2] {
                let mut triangle_key = idxs;
                triangle_key.sort_unstable();
                if triangle_set.insert(triangle_key) {
                    mesh_indices.extend_from_slice(&idxs);
                    for idx in idxs {
                        mesh_normals[idx as usize][0] += nx;
                        mesh_normals[idx as usize][1] += ny;
                        mesh_normals[idx as usize][2] += nz;
                    }
                }
            }

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

                    if vx >= 0
                        && vx < target_width as i32
                        && vy >= 0
                        && vy < target_height as i32
                        && vz >= 0
                        && vz < target_depth as i32
                    {
                        let idx = (vz as u32 * target_width * target_height)
                            + (vy as u32 * target_width)
                            + vx as u32;
                        data[idx as usize] = true;
                    }
                }
            }
        }

        for n in &mut mesh_normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 0.0 {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            } else {
                n[1] = 1.0;
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
            mesh_vertices,
            mesh_normals,
            mesh_indices,
        })
    }
}
