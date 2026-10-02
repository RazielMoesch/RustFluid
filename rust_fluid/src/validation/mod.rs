//! CPU-side error, conservation, and invalid-cell analysis.

use std::fmt;

/// Absolute and relative error summary for one scalar field.
pub struct FieldErrors {
    pub mean_abs: f64,
    pub rms: f64,
    pub max_abs: f64,
    pub relative_l2: f64,
}

impl fmt::Display for FieldErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MAE={:.2e}  RMS={:.2e}  MaxAbs={:.2e}  RelL2={:.2e}",
            self.mean_abs, self.rms, self.max_abs, self.relative_l2
        )
    }
}

/// Error summaries for density and each velocity component.
pub struct FullFieldComparison {
    pub rho: FieldErrors,
    pub ux: FieldErrors,
    pub uy: FieldErrors,
}

impl FullFieldComparison {
    pub fn print(&self) {
        println!("  rho:  {}", self.rho);
        println!("  ux:   {}", self.ux);
        println!("  uy:   {}", self.uy);
    }
}

fn compute_field_errors(reference: &[f32], test: &[f32]) -> FieldErrors {
    let n = reference.len();
    assert_eq!(n, test.len());

    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut ref_norm_sq = 0.0_f64;

    for i in 0..n {
        let r = reference[i] as f64;
        let t = test[i] as f64;
        let diff = (t - r).abs();
        sum_abs += diff;
        sum_sq += diff * diff;
        if diff > max_abs {
            max_abs = diff;
        }
        ref_norm_sq += r * r;
    }

    let mean_abs = sum_abs / n as f64;
    let rms = (sum_sq / n as f64).sqrt();
    let ref_norm = ref_norm_sq.sqrt();
    let diff_norm = sum_sq.sqrt();
    let relative_l2 = if ref_norm > 1e-30 {
        diff_norm / ref_norm
    } else {
        0.0
    };

    FieldErrors {
        mean_abs,
        rms,
        max_abs,
        relative_l2,
    }
}

pub fn compare_fields(reference: &[f32], test: &[f32], total_cells: usize) -> FullFieldComparison {
    assert!(reference.len() >= total_cells * 4);
    assert!(test.len() >= total_cells * 4);

    let mut ref_rho = vec![0.0f32; total_cells];
    let mut ref_ux = vec![0.0f32; total_cells];
    let mut ref_uy = vec![0.0f32; total_cells];
    let mut test_rho = vec![0.0f32; total_cells];
    let mut test_ux = vec![0.0f32; total_cells];
    let mut test_uy = vec![0.0f32; total_cells];

    for i in 0..total_cells {
        let b = i * 4;
        ref_ux[i] = reference[b];
        ref_uy[i] = reference[b + 1];
        ref_rho[i] = reference[b + 3];
        test_ux[i] = test[b];
        test_uy[i] = test[b + 1];
        test_rho[i] = test[b + 3];
    }

    FullFieldComparison {
        rho: compute_field_errors(&ref_rho, &test_rho),
        ux: compute_field_errors(&ref_ux, &test_ux),
        uy: compute_field_errors(&ref_uy, &test_uy),
    }
}

/// Initial/final mass values and normalized drift.
pub struct MassDiagnostics {
    pub initial_mass: f64,
    pub final_mass: f64,
    pub absolute_drift: f64,
    pub relative_drift: f64,
}

impl MassDiagnostics {
    pub fn print(&self) {
        println!("  Initial mass:    {:.6}", self.initial_mass);
        println!("  Final mass:      {:.6}", self.final_mass);
        println!("  Absolute drift:  {:.2e}", self.absolute_drift);
        println!("  Relative drift:  {:.2e}", self.relative_drift);
    }
}

pub fn compute_total_mass(macro_data: &[f32], total_cells: usize) -> f64 {
    let mut mass = 0.0_f64;
    for i in 0..total_cells {
        mass += macro_data[i * 4 + 3] as f64;
    }
    mass
}

pub fn compute_mass_diagnostics(initial_mass: f64, final_mass: f64) -> MassDiagnostics {
    let absolute_drift = (final_mass - initial_mass).abs();
    let relative_drift = if initial_mass.abs() > 1e-30 {
        absolute_drift / initial_mass.abs()
    } else {
        0.0
    };
    MassDiagnostics {
        initial_mass,
        final_mass,
        absolute_drift,
        relative_drift,
    }
}

/// Location and values of a cell that failed a stability check.
pub struct InvalidCell {
    pub cell_idx: usize,
    pub x: u32,
    pub y: u32,
    pub rho: f32,
    pub ux: f32,
    pub uy: f32,
    pub reason: &'static str,
}

impl fmt::Display for InvalidCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "  [{}: cell={} ({},{}) rho={:.6} ux={:.6} uy={:.6}]",
            self.reason, self.cell_idx, self.x, self.y, self.rho, self.ux, self.uy
        )
    }
}

pub fn scan_stability(macro_data: &[f32], nx: u32, ny: u32, max_velocity: f32) -> Vec<InvalidCell> {
    let total_cells = (nx * ny) as usize;
    let mut invalid = Vec::new();

    for idx in 0..total_cells {
        let b = idx * 4;
        let ux = macro_data[b];
        let uy = macro_data[b + 1];
        let rho = macro_data[b + 3];
        let x = (idx % nx as usize) as u32;
        let y = (idx / nx as usize) as u32;

        let reason = if rho.is_nan() || ux.is_nan() || uy.is_nan() {
            Some("NaN")
        } else if rho.is_infinite() || ux.is_infinite() || uy.is_infinite() {
            Some("Inf")
        } else if rho < 0.0 {
            Some("negative rho")
        } else if (ux * ux + uy * uy).sqrt() > max_velocity {
            Some("excessive velocity")
        } else {
            None
        };

        if let Some(reason) = reason {
            invalid.push(InvalidCell {
                cell_idx: idx,
                x,
                y,
                rho,
                ux,
                uy,
                reason,
            });
            if invalid.len() >= 20 {
                break;
            }
        }
    }

    invalid
}
