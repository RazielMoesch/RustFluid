//! Cylinder-wake checks for zero-gradient outlet and sponge damping behavior.

use crate::diagnostics::metrics::MacroMetrics;
use crate::diagnostics::result::{DiagnosticResult, DiagnosticStatus};
use crate::runtime::context::HeadlessContext;
use crate::sim::common::precision::Precision;
use crate::sim::d3::boundary::Boundary3D;
use crate::sim::d3::boundary::bounce_back::BounceBack;
use crate::sim::d3::boundary::equilibrium_inlet::EquilibriumInlet;
use crate::sim::d3::boundary::fluid::Fluid;
use crate::sim::d3::boundary::free_slip::{FreeSlipX, FreeSlipY, FreeSlipZ};
use crate::sim::d3::boundary::outlet::ZeroGradientOutlet;
use crate::sim::d3::collision::bgk::Bgk;
use crate::sim::d3::config::SimulationConfig3D;
use crate::sim::d3::lattice::d3q19::D3Q19;
use crate::sim::d3::solver::Lbm3D;

/// Free-stream velocity of the tunnel, in lattice units.
const U_REF: f32 = 0.1;

const NX: u32 = 64;
const NY: u32 = 32;
const NZ: u32 = 32;
const OMEGA: f32 = 1.0 / 0.55;

/// Snapshot of the outlet plane, sampled once per `SAMPLE_EVERY` steps.
struct OutletProbe {
    mean_ux: f32,
    transverse_rms: f32,
    q_rms: f32,
}

struct ProbeRun {
    probes: Vec<OutletProbe>,
    q_peak: f32,
    nan_count: usize,
}

fn build_flags(nx: u32, ny: u32, nz: u32, inlet: u32, outlet: u32) -> Vec<u32> {
    let mut flags = vec![0u32; (nx * ny * nz) as usize];

    // Face precedence mirrors SimDomain3D::flags so the corners agree with the
    // rest of the project: z, then y, then x.
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = (x + y * nx + z * nx * ny) as usize;
                flags[idx] = if z == 0 {
                    FreeSlipZ.type_id() << 24
                } else if z == nz - 1 {
                    FreeSlipZ.type_id() << 24
                } else if y == 0 {
                    FreeSlipY.type_id() << 24
                } else if y == ny - 1 {
                    FreeSlipY.type_id() << 24
                } else if x == 0 {
                    inlet << 24
                } else if x == nx - 1 {
                    outlet << 24
                } else {
                    0
                };
            }
        }
    }

    // A sphere as a blunt obstacle, well clear of the side walls so the wake
    // stays two-dimensional-ish and the blockage stays low.
    let cx = nx as f32 * 0.35;
    let cy = ny as f32 * 0.5;
    let cz = nz as f32 * 0.5;
    let r = ny as f32 / 6.0;
    let r2 = r * r;

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dz = z as f32 - cz;
                if dx * dx + dy * dy + dz * dz <= r2 {
                    flags[(x + y * nx + z * nx * ny) as usize] = BounceBack.type_id() << 24;
                }
            }
        }
    }

    flags
}

/// Q-criterion on one x-slice, mirroring render::qcriterion so the diagnostic
/// measures the same quantity the marching-cubes surface is built from.
/// Bounce-back cells are treated as invalid, exactly as the shader does.
fn q_on_slice(data: &[[f32; 4]], flags: &[u32], nx: u32, ny: u32, nz: u32, x: usize) -> f32 {
    let at = |xi: i32, yi: i32, zi: i32| -> Option<[f32; 3]> {
        if xi < 0 || yi < 0 || zi < 0 || xi >= nx as i32 || yi >= ny as i32 || zi >= nz as i32 {
            return None;
        }
        let idx = (xi as u32 + yi as u32 * nx + zi as u32 * nx * ny) as usize;
        if (flags[idx] >> 24) == BounceBack.type_id() {
            return None;
        }
        Some([data[idx][0], data[idx][1], data[idx][2]])
    };

    let mut sum = 0.0f32;
    let mut count = 0u32;

    let diff =
        |c: Option<[f32; 3]>, hi: Option<[f32; 3]>, lo: Option<[f32; 3]>| -> Option<[f32; 3]> {
            match (c, hi, lo) {
                // Central difference.
                (Some(_c), Some(hi), Some(lo)) => Some([
                    0.5 * (hi[0] - lo[0]),
                    0.5 * (hi[1] - lo[1]),
                    0.5 * (hi[2] - lo[2]),
                ]),
                // One-sided fallback, matching the renderer's boundary-aware
                // derivatives so the outermost planes are not silently dropped.
                (Some(c), Some(hi), None) => Some([hi[0] - c[0], hi[1] - c[1], hi[2] - c[2]]),
                (Some(c), None, Some(lo)) => Some([c[0] - lo[0], c[1] - lo[1], c[2] - lo[2]]),
                _ => None,
            }
        };

    for zi in 1..nz as i32 - 1 {
        for yi in 1..ny as i32 - 1 {
            let xi = x as i32;
            let c = at(xi, yi, zi);
            if c.is_none() {
                continue;
            }

            let dx = diff(c, at(xi + 1, yi, zi), at(xi - 1, yi, zi));
            let dy = diff(c, at(xi, yi + 1, zi), at(xi, yi - 1, zi));
            let dz = diff(c, at(xi, yi, zi + 1), at(xi, yi, zi - 1));
            let (Some(dx), Some(dy), Some(dz)) = (dx, dy, dz) else {
                continue;
            };

            let (ux_x, uy_x, uz_x) = (dx[0], dx[1], dx[2]);
            let (ux_y, uy_y, uz_y) = (dy[0], dy[1], dy[2]);
            let (ux_z, uy_z, uz_z) = (dz[0], dz[1], dz[2]);

            let s_xx = ux_x;
            let s_yy = uy_y;
            let s_zz = uz_z;
            let s_xy = 0.5 * (ux_y + uy_x);
            let s_xz = 0.5 * (ux_z + uz_x);
            let s_yz = 0.5 * (uy_z + uz_y);
            let norm_s2 = s_xx * s_xx
                + s_yy * s_yy
                + s_zz * s_zz
                + 2.0 * (s_xy * s_xy + s_xz * s_xz + s_yz * s_yz);

            let o_xy = 0.5 * (ux_y - uy_x);
            let o_xz = 0.5 * (ux_z - uz_x);
            let o_yz = 0.5 * (uy_z - uz_y);
            let norm_o2 = 2.0 * (o_xy * o_xy + o_xz * o_xz + o_yz * o_yz);

            let q = 0.5 * (norm_o2 - norm_s2);
            sum += q * q;
            count += 1;
        }
    }

    if count == 0 {
        f32::NAN
    } else {
        (sum / count as f32).sqrt()
    }
}

fn q_peak(data: &[[f32; 4]], flags: &[u32], nx: u32, ny: u32, nz: u32) -> f32 {
    let mut peak: f32 = 0.0;
    for x in 1..nx as usize - 1 {
        let q = q_on_slice(data, flags, nx, ny, nz, x);
        if q.is_finite() {
            peak = peak.max(q);
        }
    }
    peak
}

fn run_once(
    ctx: &HeadlessContext,
    precision: Precision,
    steps: u32,
    sponge_len: u32,
    sponge_strength: f32,
) -> ProbeRun {
    let lattice = D3Q19::new();
    let collision = Bgk::new();
    let boundaries: Vec<&dyn Boundary3D> = vec![
        &Fluid,
        &BounceBack,
        &ZeroGradientOutlet,
        &EquilibriumInlet,
        &FreeSlipX,
        &FreeSlipY,
        &FreeSlipZ,
    ];

    let config = SimulationConfig3D {
        nx: NX,
        ny: NY,
        nz: NZ,
        init_type: crate::sim::d3::config::InitType::Uniform,
        rho_init: 1.0,
        u_x_init: U_REF,
        u_y_init: 0.0,
        u_z_init: 0.0,
        wgs_x: 8,
        wgs_y: 8,
        wgs_z: 1,
        omega: OMEGA,
        force_x: 0.0,
        force_y: 0.0,
        force_z: 0.0,
        periodic_x: false,
        periodic_y: false,
        periodic_z: false,
        pure_fluid: false,
        num_boundary_configs: 1,
        sponge_len,
        sponge_strength,
        sponge_cfg: 0,
    };

    let mut lbm = Lbm3D::new(
        ctx.device,
        config,
        precision,
        &lattice,
        &collision,
        &boundaries,
    );

    let flags = build_flags(
        NX,
        NY,
        NZ,
        EquilibriumInlet.type_id(),
        ZeroGradientOutlet.type_id(),
    );
    // Config 0 doubles as the inlet state and the sponge target state.
    let bcs = vec![U_REF, 0.0, 0.0, 1.0];
    lbm.write_buffers(ctx.queue, &flags, &bcs);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Wake Init Encoder"),
        });
    lbm.init(&mut encoder);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    // An odd stride keeps the two half-step parity phases in the sample set: a
    // wrong odd-phase slot makes the outlet plane alternate between +U_REF and
    // -U_REF every step, and an even stride would alias that away.
    const SAMPLE_EVERY: u32 = 17;
    let mut probes = Vec::new();
    let mut done = 0u32;

    while done < steps {
        let chunk = SAMPLE_EVERY.min(steps - done);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Wake Step Encoder"),
            });
        for _ in 0..chunk {
            lbm.step(&mut encoder);
        }
        lbm.extract(&mut encoder);
        ctx.queue.submit(std::iter::once(encoder.finish()));
        done += chunk;

        let data = lbm.download_macro_data(ctx.device, ctx.queue);

        let mut ux_sum = 0.0f32;
        let mut ut2_sum = 0.0f32;
        let mut cells = 0u32;
        for z in 0..NZ {
            for y in 0..NY {
                let cell = data[((NX - 1) + y * NX + z * NX * NY) as usize];
                ux_sum += cell[0];
                ut2_sum += cell[1] * cell[1] + cell[2] * cell[2];
                cells += 1;
            }
        }

        probes.push(OutletProbe {
            mean_ux: ux_sum / cells as f32,
            transverse_rms: (ut2_sum / cells as f32).sqrt(),
            q_rms: q_on_slice(&data, &flags, NX, NY, NZ, (NX - 1) as usize),
        });
    }

    let data = lbm.download_macro_data(ctx.device, ctx.queue);
    let metrics = MacroMetrics::compute(&data);

    ProbeRun {
        probes,
        q_peak: q_peak(&data, &flags, NX, NY, NZ),
        nan_count: metrics.nan_count + metrics.inf_count,
    }
}

fn std_dev(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32;
    var.sqrt()
}

/// Wind-tunnel wake case: verifies the outlet plane behaves as an open outflow.
///
/// Two failure modes are checked.
///
/// 1. A zero-gradient outlet that copies its own cell (instead of extrapolating
///    from the interior) freezes the outflow populations, and reading the wrong
///    slot on the odd phase additionally mirrors the streamwise momentum every
///    step. The outlet plane then oscillates between +U_REF and -U_REF, so its
///    mean u_x is both far from the free stream and wildly unsteady in time.
/// 2. Even with a correct outlet, wake vorticity reaching the plane registers
///    as a Q-criterion pile-up at the back of the tunnel. The sponge layer must
///    reduce that relative to running the same case without it.
/// Compares wake behavior with and without the high-X sponge layer.
pub fn run_wake_outlet(
    ctx: &HeadlessContext,
    precision: Precision,
    steps: u32,
) -> DiagnosticResult {
    let name = format!("wake_outlet_{precision:?}_{steps}steps");

    // The tunnel here is far shorter than the wind-tunnel example, so the
    // sponge needs to be aggressive for its effect on the outlet plane to be
    // unambiguous rather than marginal.
    let damped = run_once(ctx, precision, steps, NX / 8, 0.2);
    let undamped = run_once(ctx, precision, steps, 0, 0.0);

    let mut messages = Vec::new();
    let mut status = DiagnosticStatus::Pass;
    let mut check = |ok: bool, msg: String| {
        if !ok {
            status = DiagnosticStatus::Fail;
            messages.push(msg);
        }
    };

    check(
        damped.nan_count == 0 && undamped.nan_count == 0,
        format!(
            "Non-finite cells: sponge={} no_sponge={}",
            damped.nan_count, undamped.nan_count
        ),
    );

    // 1. The outlet plane must be a steady, open outflow at the free-stream
    //    velocity, not a plug that mirrors the streamwise momentum.
    let outlet_ux: Vec<f32> = damped.probes.iter().map(|p| p.mean_ux).collect();
    let ux_std = std_dev(&outlet_ux);
    let ux_mean = outlet_ux.iter().sum::<f32>() / outlet_ux.len() as f32;
    let ux_worst = outlet_ux
        .iter()
        .map(|v| (v - U_REF).abs())
        .fold(0.0f32, f32::max);

    check(
        ux_std <= 0.1 * U_REF,
        format!(
            "Outlet plane u_x is unsteady: std={ux_std:.5} ({:.1}% of U_REF) — outlet is not convecting flow out",
            100.0 * ux_std / U_REF
        ),
    );
    check(
        ux_worst <= 0.2 * U_REF,
        format!(
            "Outlet plane u_x deviates from the free stream by up to {ux_worst:.5} (mean {ux_mean:.5}, U_REF {U_REF:.5})"
        ),
    );

    // 2. The sponge must absorb the wake before it reaches the plane: both the
    //    Q-criterion and the transverse velocity there must drop.
    let damped_q = damped.probes.last().map(|p| p.q_rms).unwrap_or(f32::NAN);
    let undamped_q = undamped.probes.last().map(|p| p.q_rms).unwrap_or(f32::NAN);
    check(
        damped_q.is_finite() && undamped_q.is_finite() && damped_q < undamped_q,
        format!(
            "Sponge did not reduce outlet Q-criterion: sponge={damped_q:.3e} no_sponge={undamped_q:.3e}"
        ),
    );

    let damped_ut = damped
        .probes
        .last()
        .map(|p| p.transverse_rms)
        .unwrap_or(f32::NAN);
    let undamped_ut = undamped
        .probes
        .last()
        .map(|p| p.transverse_rms)
        .unwrap_or(f32::NAN);
    check(
        damped_ut.is_finite() && undamped_ut.is_finite() && damped_ut < undamped_ut,
        format!(
            "Sponge did not reduce outlet transverse velocity: sponge={damped_ut:.3e} no_sponge={undamped_ut:.3e}"
        ),
    );

    // 3. Sanity: the outlet must still be far less energetic than the wake it
    //    is absorbing, otherwise the wake was never near the outlet at all and
    //    the comparison above proves nothing.
    let outlet_energy = damped_q.max(0.0) / damped.q_peak.max(f32::MIN_POSITIVE);
    check(
        outlet_energy < 0.25,
        format!(
            "Outlet Q-criterion is {:.1}% of the domain peak — vorticity is piling up at the outlet plane",
            100.0 * outlet_energy
        ),
    );

    println!(
        "  [wake] outlet u_x std={ux_std:.2e} worst dev={ux_worst:.2e} \
         q_sponge={damped_q:.3e} q_nosponge={undamped_q:.3e} q_peak={:.3e} \
         outlet/peak={:.4} transverse_rms sponge={:.3e} nosponge={:.3e}",
        damped.q_peak, outlet_energy, damped_ut, undamped_ut
    );

    if messages.is_empty() {
        DiagnosticResult::pass(&name)
    } else {
        DiagnosticResult {
            name,
            status,
            messages,
        }
    }
}
