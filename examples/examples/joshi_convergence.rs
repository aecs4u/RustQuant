// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Joshi "Concepts and Practice of Mathematical Finance" - Part 7
// SimpleMC7: Statistics Gathering and Convergence Analysis
//
// Joshi introduces a ConvergenceTable that tracks how MC estimates
// converge as the number of paths increases. This example implements:
//   1. Running mean and standard error tracking
//   2. Convergence table (doubling path count)
//   3. Confidence intervals
//   4. Antithetic variates for variance reduction
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

use std::f64::consts::PI;
use time::macros::date;
use RustQuant::instruments::options::*;
use RustQuant::instruments::*;
use RustQuant::stochastics::*;

/// Statistics gatherer inspired by Joshi's ConvergenceTable.
struct ConvergenceTable {
    running_sum: f64,
    running_sum_sq: f64,
    count: usize,
    snapshots: Vec<(usize, f64, f64, f64)>, // (n, mean, stderr, 95% half-width)
}

impl ConvergenceTable {
    fn new() -> Self {
        Self {
            running_sum: 0.0,
            running_sum_sq: 0.0,
            count: 0,
            snapshots: Vec::new(),
        }
    }

    fn add(&mut self, value: f64) {
        self.running_sum += value;
        self.running_sum_sq += value * value;
        self.count += 1;
    }

    fn mean(&self) -> f64 {
        self.running_sum / self.count as f64
    }

    fn variance(&self) -> f64 {
        let n = self.count as f64;
        self.running_sum_sq / n - (self.running_sum / n).powi(2)
    }

    fn standard_error(&self) -> f64 {
        (self.variance() / self.count as f64).sqrt()
    }

    fn snapshot(&mut self) {
        let mean = self.mean();
        let se = self.standard_error();
        let hw = 1.96 * se; // 95% confidence
        self.snapshots.push((self.count, mean, se, hw));
    }
}

fn main() {
    let spot = 100.0;
    let strike = 100.0;
    let rate = 0.05;
    let volatility = 0.20;
    let expiry_years = 1.0;

    println!("==========================================================");
    println!("Joshi Part 7: Statistics Gathering & Convergence (SimpleMC7)");
    println!("==========================================================\n");

    // Analytic reference price
    let expiry = date!(2027 - 03 - 22);
    let bsm = BlackScholesMertonBuilder::default()
        .underlying_price(spot)
        .strike_price(strike)
        .volatility(volatility)
        .risk_free_rate(rate)
        .cost_of_carry(rate)
        .expiration_date(expiry)
        .option_type(TypeFlag::Call)
        .build()
        .unwrap();
    let exact_price = bsm.price();
    println!("Exact BS Call Price: {:.6}\n", exact_price);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 1. Convergence Table - Standard MC
    //
    // Generate paths in batches, doubling each time.
    // Track how the estimate converges to the true price.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("--- 1. Convergence Table (Standard MC) ---");

    let mut table = ConvergenceTable::new();
    let mut rng = rand::thread_rng();
    let total_paths = 1_048_576; // 2^20
    let mut next_snapshot = 256;

    let df = (-rate * expiry_years).exp();

    for _ in 0..total_paths {
        let u1: f64 = rand::Rng::gen(&mut rng);
        let u2: f64 = rand::Rng::gen(&mut rng);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

        let s_t = spot * ((rate - 0.5 * volatility * volatility) * expiry_years
            + volatility * expiry_years.sqrt() * z)
            .exp();

        let discounted_payoff = df * (s_t - strike).max(0.0);
        table.add(discounted_payoff);

        if table.count == next_snapshot {
            table.snapshot();
            next_snapshot *= 2;
        }
    }
    table.snapshot(); // Final

    println!("  {:>10} {:>12} {:>12} {:>12} {:>12}",
        "Paths", "Price", "Std Err", "95% CI ±", "Error");
    println!("  {}", "-".repeat(58));
    for (n, mean, se, hw) in &table.snapshots {
        println!("  {:>10} {:>12.4} {:>12.6} {:>12.6} {:>12.6}",
            n, mean, se, hw, mean - exact_price);
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 2. Antithetic Variates
    //
    // Joshi discusses variance reduction. Antithetic variates use
    // both Z and -Z for each random draw, reducing variance.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- 2. Antithetic Variates Convergence ---");

    let mut table_av = ConvergenceTable::new();
    next_snapshot = 256;

    for _ in 0..total_paths / 2 {
        let u1: f64 = rand::Rng::gen(&mut rng);
        let u2: f64 = rand::Rng::gen(&mut rng);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

        // Original path
        let s_t_pos = spot * ((rate - 0.5 * volatility * volatility) * expiry_years
            + volatility * expiry_years.sqrt() * z)
            .exp();
        // Antithetic path (use -Z)
        let s_t_neg = spot * ((rate - 0.5 * volatility * volatility) * expiry_years
            + volatility * expiry_years.sqrt() * (-z))
            .exp();

        // Average the two payoffs
        let payoff = 0.5 * ((s_t_pos - strike).max(0.0) + (s_t_neg - strike).max(0.0));
        let discounted = df * payoff;

        table_av.add(discounted);
        // Count effective paths as 2 per iteration
        table_av.add(discounted); // double-count for fair comparison

        if table_av.count >= next_snapshot {
            table_av.snapshot();
            next_snapshot *= 2;
        }
    }
    table_av.snapshot();

    println!("  {:>10} {:>12} {:>12} {:>12} {:>12}",
        "Paths", "Price", "Std Err", "95% CI ±", "Error");
    println!("  {}", "-".repeat(58));
    for (n, mean, se, hw) in &table_av.snapshots {
        println!("  {:>10} {:>12.4} {:>12.6} {:>12.6} {:>12.6}",
            n, mean, se, hw, mean - exact_price);
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 3. Comparison using RustQuant's MC engine at different path counts
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- 3. RustQuant MC Engine at Various Path Counts ---");
    println!("  {:>10} {:>12} {:>12}", "Paths", "MC Price", "Error");
    println!("  {}", "-".repeat(34));

    let gbm = GeometricBrownianMotion::new(rate, volatility);
    let vanilla = EuropeanVanillaOption::new(strike, expiry, TypeFlag::Call);

    for &n in &[1_000, 5_000, 10_000, 50_000, 100_000, 500_000] {
        let cfg = StochasticProcessConfig::new(
            spot, 0.0, expiry_years, 1,
            StochasticScheme::EulerMaruyama,
            n, true, None,
        );
        let price = vanilla.price_monte_carlo(&gbm, &cfg, rate);
        println!("  {:>10} {:>12.4} {:>12.6}", n, price, price - exact_price);
    }

    println!("\n--- Summary ---");
    println!("  Standard MC convergence rate: O(1/sqrt(N))");
    println!("  Antithetic variates roughly halve the variance.");
    println!("  The 95% CI narrows as sqrt(N) grows.");
}
