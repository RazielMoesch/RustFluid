//! Depth-tested wireframes for the domain box and optional obstacle mesh.

use std::collections::HashSet;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::mesh::Vertex;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LineVertex {
    position: [f32; 3],
}

/// Bright diagnostic outlines for the domain box and imported geometry.
/// Renders the outer lattice bounds and deduplicated obstacle edges.
pub struct DomainWireframeRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
}

impl DomainWireframeRenderer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        nx: u32,
        ny: u32,
        nz: u32,
        geometry: Option<(&[Vertex], &[u32])>,
    ) -> Self {
        let vertices = build_line_vertices(nx, ny, nz, geometry);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Domain/geometry wireframe vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Domain/geometry wireframe bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Domain/geometry wireframe bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Domain/geometry wireframe shader"),
            source: wgpu::ShaderSource::Wgsl(WIREFRAME_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Domain/geometry wireframe pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Domain/geometry wireframe pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                // A diagnostic outline should expose rear-facing edges too.
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group,
            vertex_buffer,
            vertex_count: vertices.len() as u32,
        }
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..self.vertex_count, 0..1);
    }
}

fn build_line_vertices(
    nx: u32,
    ny: u32,
    nz: u32,
    geometry: Option<(&[Vertex], &[u32])>,
) -> Vec<LineVertex> {
    let corners = [
        [0.0, 0.0, 0.0],
        [nx as f32, 0.0, 0.0],
        [0.0, ny as f32, 0.0],
        [nx as f32, ny as f32, 0.0],
        [0.0, 0.0, nz as f32],
        [nx as f32, 0.0, nz as f32],
        [0.0, ny as f32, nz as f32],
        [nx as f32, ny as f32, nz as f32],
    ];
    let box_edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    let mut lines = Vec::with_capacity(24);
    for (a, b) in box_edges {
        lines.push(LineVertex {
            position: corners[a],
        });
        lines.push(LineVertex {
            position: corners[b],
        });
    }

    if let Some((mesh_vertices, indices)) = geometry {
        let mut unique_edges = HashSet::new();
        for triangle in indices.chunks_exact(3) {
            for (a, b) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                unique_edges.insert(if a < b { (a, b) } else { (b, a) });
            }
        }
        let mut unique_edges: Vec<_> = unique_edges.into_iter().collect();
        unique_edges.sort_unstable();
        lines.reserve(unique_edges.len() * 2);
        for (a, b) in unique_edges {
            let (Some(a), Some(b)) = (mesh_vertices.get(a as usize), mesh_vertices.get(b as usize))
            else {
                continue;
            };
            lines.push(LineVertex {
                position: a.position,
            });
            lines.push(LineVertex {
                position: b.position,
            });
        }
    }
    lines
}

const WIREFRAME_SHADER: &str = r#"
struct Uniforms {
    inv_view_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    eye: vec3<f32>,
    max_speed: f32,
    iso_q: f32,
    mode: u32,
    pad: f32,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return uniforms.view_proj * vec4<f32>(position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.55, 1.0, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_has_twelve_edges_and_mesh_edges_are_deduplicated() {
        let mesh = [
            Vertex {
                position: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
            Vertex {
                position: [0.0, 1.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        ];
        let lines = build_line_vertices(2, 3, 4, Some((&mesh, &[0, 1, 2, 2, 1, 0])));
        assert_eq!(lines.len(), 30);
    }
}
