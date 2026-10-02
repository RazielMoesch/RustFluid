//! GPU storage allocated by a two-dimensional simulation.

use crate::gpu::utils::bg_entry;

/// Population, flags, boundary settings, macro output, and bind groups.
pub struct SimBuffers2D {
    pub fa: wgpu::Buffer,
    pub fb: wgpu::Buffer,
    pub flags: wgpu::Buffer,
    pub boundary_configs: wgpu::Buffer,
    pub macro_data: wgpu::Buffer,

    pub init_bg: wgpu::BindGroup,
    pub step_bg_a: wgpu::BindGroup,
    pub step_bg_b: wgpu::BindGroup,

    pub extract_bg_a: wgpu::BindGroup,
    pub extract_bg_b: wgpu::BindGroup,
}

impl SimBuffers2D {
    pub fn new(
        device: &wgpu::Device,
        nx: u32,
        ny: u32,
        q: u32,
        bytes_per_population: wgpu::BufferAddress,
        num_boundary_configs: u32,
        init_bgl: &wgpu::BindGroupLayout,
        step_bgl: &wgpu::BindGroupLayout,
        extract_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let total_cells = (nx * ny) as wgpu::BufferAddress;

        let f_size = total_cells * (q as wgpu::BufferAddress) * bytes_per_population;
        let flags_size = total_cells * 4;

        let config_count = num_boundary_configs.max(1) as wgpu::BufferAddress;
        let config_size = config_count * 16;

        let macro_data_size = total_cells * 16;

        let fa = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Population FA Buffer"),
            size: f_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let fb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Population FB Buffer"),
            size: f_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let flags = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Geometry Flags Buffer"),
            size: flags_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let boundary_configs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Boundary Configs Buffer"),
            size: config_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let macro_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Macro Data Buffer"),
            size: macro_data_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let init_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Init Bind Group"),
            layout: init_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fa.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: flags.as_entire_binding(),
                },
            ],
        });

        let step_a_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: fa.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: fb.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: flags.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: boundary_configs.as_entire_binding(),
            },
        ];

        let step_b_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: fb.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: fa.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: flags.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: boundary_configs.as_entire_binding(),
            },
        ];

        let step_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Step Bind Group A"),
            layout: step_bgl,
            entries: &step_a_entries,
        });

        let step_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Step Bind Group B"),
            layout: step_bgl,
            entries: &step_b_entries,
        });

        let extract_a_entries = vec![
            bg_entry(0, fa.as_entire_binding()),
            bg_entry(1, macro_data.as_entire_binding()),
        ];

        let extract_b_entries = vec![
            bg_entry(0, fb.as_entire_binding()),
            bg_entry(1, macro_data.as_entire_binding()),
        ];

        let extract_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Extract Bind Group A"),
            layout: extract_bgl,
            entries: &extract_a_entries,
        });

        let extract_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Extract Bind Group B"),
            layout: extract_bgl,
            entries: &extract_b_entries,
        });

        Self {
            fa,
            fb,
            flags,
            boundary_configs,
            macro_data,
            init_bg,
            step_bg_a,
            step_bg_b,
            extract_bg_a,
            extract_bg_b,
        }
    }
}
