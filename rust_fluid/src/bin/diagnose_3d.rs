//! Command-line entry point for repeatable three-dimensional diagnostics.

use rust_fluid::diagnostics::cases::closed_box::run_closed_box;
use rust_fluid::diagnostics::cases::parity::run_parity;
use rust_fluid::diagnostics::cases::taylor_green::run_taylor_green;
use rust_fluid::diagnostics::cases::uniform::run_uniform;
use rust_fluid::diagnostics::cases::wake::run_wake_outlet;
use rust_fluid::diagnostics::runner::DiagnosticRunner;
use rust_fluid::sim::common::precision::Precision;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let suite = if args.len() > 2 && args[1] == "--suite" {
        args[2].clone()
    } else {
        "smoke".to_string()
    };

    println!("Running diagnostic suite: {}", suite);
    let mut runner = DiagnosticRunner::new();

    if suite == "smoke" {
        runner.run_with_context("Uniform Equilibrium FP32 32^3 100 steps", |ctx| {
            run_uniform(ctx, Precision::F32, 100, 32)
        });

        runner.run_with_context("Uniform Equilibrium FP16 32^3 100 steps", |ctx| {
            run_uniform(ctx, Precision::FP16S, 100, 32)
        });

        runner.run_with_context("AA Parity FP32 16^3", |ctx| {
            run_parity(ctx, Precision::F32, 16)
        });

        runner.run_with_context("AA Parity FP16 16^3", |ctx| {
            run_parity(ctx, Precision::FP16S, 16)
        });

        runner.run_with_context("Closed Box FP32 16^3", |ctx| {
            run_closed_box(ctx, Precision::F32, 1000, 16)
        });

        runner.run_with_context("Closed Box FP16 16^3", |ctx| {
            run_closed_box(ctx, Precision::FP16S, 1000, 16)
        });

        runner.run_with_context("Taylor-Green FP32 32^3", |ctx| {
            run_taylor_green(ctx, Precision::F32, 500, 32)
        });

        runner.run_with_context("Taylor-Green FP16 32^3", |ctx| {
            run_taylor_green(ctx, Precision::FP16S, 500, 32)
        });

        runner.run_with_context("Wake outlet + sponge FP32 64x32x32", |ctx| {
            run_wake_outlet(ctx, Precision::F32, 600)
        });

        // The wind-tunnel example runs FP16S, so the sponge must be verified
        // there too.
        runner.run_with_context("Wake outlet + sponge FP16S 64x32x32", |ctx| {
            run_wake_outlet(ctx, Precision::FP16S, 600)
        });
    } else {
        println!("Suite {} not implemented yet.", suite);
    }

    let fails = runner
        .results
        .iter()
        .filter(|r| r.status == rust_fluid::diagnostics::result::DiagnosticStatus::Fail)
        .count();
    let warns = runner
        .results
        .iter()
        .filter(|r| r.status == rust_fluid::diagnostics::result::DiagnosticStatus::Warning)
        .count();
    let passes = runner
        .results
        .iter()
        .filter(|r| r.status == rust_fluid::diagnostics::result::DiagnosticStatus::Pass)
        .count();

    println!("------------------------------------------------------------");
    println!("DIAGNOSTICS SUMMARY");
    println!("Passed:  {}", passes);
    println!("Warning: {}", warns);
    println!("Failed:  {}", fails);
    println!("------------------------------------------------------------");

    if fails > 0 {
        std::process::exit(1);
    }
}
