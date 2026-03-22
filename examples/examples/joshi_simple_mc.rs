// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Joshi "Concepts and Practice of Mathematical Finance" - Part 7
// SimpleMC1 through SimpleMC3: A Simple Monte Carlo Pricer
//
// This example progressively builds a Monte Carlo pricer:
//   1. From-scratch vanilla MC (like Joshi's SimpleMC1)
//   2. Using RustQuant's GBM for path generation
//   3. Using RustQuant's full MC pricing engine
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

use std::f64::consts::PI;
use time::macros::date;
use RustQuant::instruments::*;
use RustQuant::stochastics::*;

fn main() {
    let spot = 100.0;
    let strike = 100.0;
    let rate = 0.05;
    let volatility = 0.20;
    let expiry_years = 1.0;
    let n_paths: usize = 100_000;

    println!("==========================================================");
    println!("Joshi Part 7: Simple Monte Carlo Pricer");
    println!("==========================================================");
    println!("Parameters: S={}, K={}, r={}, vol={}, T={}", spot, strike, rate, volatility, expiry_years);
    println!("Paths: {}\n", n_paths);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // STEP 1: SimpleMC1 - Raw Monte Carlo from scratch
    // Joshi's first example: direct GBM terminal value sampling.
    //
    // S_T = S_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    // where Z ~ N(0,1)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("--- Step 1: Raw MC from scratch (Joshi SimpleMC1) ---");

    let mut rng = rand::thread_rng();
    let mut payoff_sum = 0.0;

    for _ in 0..n_paths {
        // Box-Muller transform for normal random
        let u1: f64 = rand::Rng::gen(&mut rng);
        let u2: f64 = rand::Rng::gen(&mut rng);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

        // GBM terminal value
        let s_t = spot * ((rate - 0.5 * volatility * volatility) * expiry_years
            + volatility * expiry_years.sqrt() * z)
            .exp();

        // Call payoff
        payoff_sum += (s_t - strike).max(0.0);
    }

    let mc_price_raw = (-rate * expiry_years).exp() * payoff_sum / n_paths as f64;
    println!("  MC Call Price: {:.4}", mc_price_raw);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // STEP 2: SimpleMC2 - Using RustQuant's GBM simulation
    // Same idea but leveraging the library for path generation.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- Step 2: MC with RustQuant GBM paths (Joshi SimpleMC2) ---");

    let gbm = GeometricBrownianMotion::new(rate, volatility);
    let config = StochasticProcessConfig::new(
        spot, 0.0, expiry_years, 1, // 1 step = terminal value only
        StochasticScheme::EulerMaruyama,
        n_paths, true, None,
    );

    let output = gbm.generate(&config);

    let payoff_sum: f64 = output
        .paths
        .iter()
        .map(|path| (*path.last().unwrap() - strike).max(0.0))
        .sum();

    let mc_price_gbm = (-rate * expiry_years).exp() * payoff_sum / n_paths as f64;
    println!("  MC Call Price: {:.4}", mc_price_gbm);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // STEP 3: SimpleMC3 - Using RustQuant's full MC pricing engine
    // The library's MonteCarloPricer trait handles everything.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- Step 3: RustQuant MC pricing engine (Joshi SimpleMC3) ---");

    let expiry = date!(2027 - 03 - 22);
    let config_full = StochasticProcessConfig::new(
        spot, 0.0, expiry_years, 252,
        StochasticScheme::EulerMaruyama,
        n_paths, true, None,
    );

    let vanilla_call = EuropeanVanillaOption::new(strike, expiry, TypeFlag::Call);
    let mc_price_engine = vanilla_call.price_monte_carlo(&gbm, &config_full, rate);
    println!("  MC Call Price: {:.4}", mc_price_engine);

    let vanilla_put = EuropeanVanillaOption::new(strike, expiry, TypeFlag::Put);
    let mc_put = vanilla_put.price_monte_carlo(&gbm, &config_full, rate);
    println!("  MC Put Price:  {:.4}", mc_put);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Analytic comparison (Black-Scholes)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- Analytic Black-Scholes-Merton ---");

    let bsm_call = options::BlackScholesMertonBuilder::default()
        .underlying_price(spot)
        .strike_price(strike)
        .volatility(volatility)
        .risk_free_rate(rate)
        .cost_of_carry(rate)
        .expiration_date(expiry)
        .option_type(TypeFlag::Call)
        .build()
        .unwrap();

    let bsm_put = options::BlackScholesMertonBuilder::default()
        .underlying_price(spot)
        .strike_price(strike)
        .volatility(volatility)
        .risk_free_rate(rate)
        .cost_of_carry(rate)
        .expiration_date(expiry)
        .option_type(TypeFlag::Put)
        .build()
        .unwrap();

    println!("  BS Call Price: {:.4}", bsm_call.price());
    println!("  BS Put Price:  {:.4}", bsm_put.price());

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Summary
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- Summary ---");
    println!("  {:>25} {:>12} {:>12}", "Method", "Call", "Put");
    println!("  {:>25} {:>12.4} {:>12}", "Raw MC (SimpleMC1)", mc_price_raw, "N/A");
    println!("  {:>25} {:>12.4} {:>12}", "GBM MC (SimpleMC2)", mc_price_gbm, "N/A");
    println!("  {:>25} {:>12.4} {:>12.4}", "Engine MC (SimpleMC3)", mc_price_engine, mc_put);
    println!("  {:>25} {:>12.4} {:>12.4}", "Black-Scholes (exact)", bsm_call.price(), bsm_put.price());
}
