//! Helpers for synchronously downloading macroscopic solver output.

use std::sync::mpsc;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Device, MapMode, Queue,
};

/// Copies a macro buffer to staging memory and returns its `f32` values.
pub fn readback_macro_data(
    device: &Device,
    queue: &Queue,
    macro_buffer: &Buffer,
    total_cells: usize,
) -> Vec<f32> {
    let size = (total_cells * 16) as u64; // vec4<f32> is 16 bytes per cell
    let readback_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Macro Data Readback Buffer"),
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Macro Data Readback Encoder"),
    });

    encoder.copy_buffer_to_buffer(macro_buffer, 0, &readback_buffer, 0, size);
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = readback_buffer.slice(..);
    let (tx, rx) = mpsc::channel();

    buffer_slice.map_async(MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });

    // Wait for the GPU to finish
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv()
        .unwrap()
        .expect("Failed to map the macro data readback buffer");

    let data_ref = buffer_slice
        .get_mapped_range()
        .expect("Failed to get mapped range");
    let floats: Vec<f32> = bytemuck::cast_slice(&data_ref).to_vec();

    drop(data_ref);
    readback_buffer.unmap();

    floats
}
