//! GPU storage allocation and memory accounting for D3Q19.

use crate::gpu::utils::bg_entry;

/// Byte sizes of every GPU buffer the 3D solver allocates.
///
/// Computed without a device so the footprint can be reported before anything
/// is reserved, which is what you want when sizing a grid against a card.
#[derive(Debug, Clone, Copy)]
pub struct SimBufferSizes {
    /// Size of the in-place population buffer.
    pub population: wgpu::BufferAddress,
    pub flags: wgpu::BufferAddress,
    pub boundary_configs: wgpu::BufferAddress,
    pub macro_data: wgpu::BufferAddress,
    /// Sum of every buffer above, i.e. the solver's share of VRAM.
    pub total: wgpu::BufferAddress,
}

impl SimBufferSizes {
    pub fn total_mib(&self) -> f64 {
        self.total as f64 / (1024.0 * 1024.0)
    }
}

/// Population, flags, boundary settings, macro output, and bind groups.
pub struct SimBuffers3D {
    pub fa: wgpu::Buffer,
    pub flags: wgpu::Buffer,
    pub boundary_configs: wgpu::Buffer,
    pub macro_data: wgpu::Buffer,

    pub init_bg: wgpu::BindGroup,
    pub step_bg: wgpu::BindGroup,
    pub extract_bg: wgpu::BindGroup,
}

impl SimBuffers3D {
    /// Exact allocation sizes for a grid, without touching the device.
    pub fn byte_sizes(
        nx: u32,
        ny: u32,
        nz: u32,
        q: u32,
        bytes_per_population: wgpu::BufferAddress,
        num_boundary_configs: u32,
    ) -> SimBufferSizes {
        let total_cells = (nx * ny * nz) as wgpu::BufferAddress;

        let population = total_cells * (q as wgpu::BufferAddress) * bytes_per_population;
        let flags = total_cells * 4;

        let config_count = num_boundary_configs.max(1) as wgpu::BufferAddress;
        let boundary_configs = config_count * 16;

        let macro_data = total_cells * 16;

        SimBufferSizes {
            population,
            flags,
            boundary_configs,
            macro_data,
            // in-place populations + flags + boundary_configs + macro_data
            total: population + flags + boundary_configs + macro_data,
        }
    }

    pub fn new(
        device: &wgpu::Device,
        nx: u32,
        ny: u32,
        nz: u32,
        q: u32,
        bytes_per_population: wgpu::BufferAddress,
        num_boundary_configs: u32,
        init_bgl: &wgpu::BindGroupLayout,
        step_bgl: &wgpu::BindGroupLayout,
        extract_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let sizes = Self::byte_sizes(nx, ny, nz, q, bytes_per_population, num_boundary_configs);

        let f_size = sizes.population;
        let flags_size = sizes.flags;
        let config_size = sizes.boundary_configs;
        let macro_data_size = sizes.macro_data;

        let fa = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Population FA Buffer 3D"),
            size: f_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let flags = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Geometry Flags Buffer 3D"),
            size: flags_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let boundary_configs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Boundary Configs Buffer 3D"),
            size: config_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let macro_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Macro Data Buffer 3D"),
            size: macro_data_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let init_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Init Bind Group 3D"),
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

        let step_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
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

        let step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("In-place Step Bind Group 3D"),
            layout: step_bgl,
            entries: &step_entries,
        });

        let extract_entries = vec![
            bg_entry(0, fa.as_entire_binding()),
            bg_entry(1, macro_data.as_entire_binding()),
        ];

        let extract_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("In-place Extract Bind Group 3D"),
            layout: extract_bgl,
            entries: &extract_entries,
        });

        Self {
            fa,
            flags,
            boundary_configs,
            macro_data,
            init_bg,
            step_bg,
            extract_bg,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SimBuffers3D;

    #[test]
    fn byte_sizes_match_the_buffers_that_get_allocated() {
        let sizes = SimBuffers3D::byte_sizes(4, 4, 4, 19, 2, 1);

        // 64 cells, 19 populations at 2 bytes.
        assert_eq!(sizes.population, 64 * 19 * 2);
        assert_eq!(sizes.flags, 64 * 4);
        assert_eq!(sizes.boundary_configs, 16);
        assert_eq!(sizes.macro_data, 64 * 16);
        assert_eq!(
            sizes.total,
            sizes.population + sizes.flags + sizes.boundary_configs + sizes.macro_data
        );
    }

    #[test]
    fn fp16_is_half_the_footprint_of_fp32() {
        let fp32 = SimBuffers3D::byte_sizes(64, 32, 32, 19, 4, 1);
        let fp16 = SimBuffers3D::byte_sizes(64, 32, 32, 19, 2, 1);
        // Only the population buffer scales with precision.
        assert_eq!(fp16.population * 2, fp32.population);
        assert!(fp16.total < fp32.total);
    }

    #[test]
    fn zero_boundary_configs_still_allocates_one_slot() {
        let sizes = SimBuffers3D::byte_sizes(8, 8, 8, 19, 2, 0);
        assert_eq!(sizes.boundary_configs, 16);
    }
}
