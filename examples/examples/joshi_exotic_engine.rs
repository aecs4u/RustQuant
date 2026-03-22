// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Joshi "Concepts and Practice of Mathematical Finance" - Part 7
// Exotic Engine: Path-Dependent Monte Carlo Pricing
//
// Joshi's ExoticEngine extends the simple MC to handle path-dependent
// derivatives. RustQuant implements this via the Payoff trait with
// `Underlying = Vec<f64>`. This example demonstrates:
//   1. Asian options (arithmetic and geometric averaging)
//   2. Barrier options (knock-in and knock-out)
//   3. Lookback options (fixed and floating strike)
//   4. Comparing path-dependent vs path-independent prices
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

use time::macros::date;
use RustQuant::instruments::options::*;
use RustQuant::instruments::*;
use RustQuant::stochastics::*;

fn main() {
    let spot = 100.0;
    let strike = 100.0;
    let rate = 0.05;
    let volatility = 0.20;
    let expiry = date!(2027 - 03 - 22);
    let n_paths = 100_000;
    let n_steps = 252; // Daily steps for path-dependent options

    let gbm = GeometricBrownianMotion::new(rate, volatility);
    let config = StochasticProcessConfig::new(
        spot, 0.0, 1.0, n_steps,
        StochasticScheme::EulerMaruyama,
        n_paths, true, None,
    );

    let contract_call = OptionContractBuilder::default()
        .type_flag(TypeFlag::Call)
        .exercise_flag(ExerciseFlag::European { expiry })
        .strike_flag(Some(StrikeFlag::Fixed))
        .build()
        .unwrap();

    let contract_put = OptionContractBuilder::default()
        .type_flag(TypeFlag::Put)
        .exercise_flag(ExerciseFlag::European { expiry })
        .strike_flag(Some(StrikeFlag::Fixed))
        .build()
        .unwrap();

    println!("==========================================================");
    println!("Joshi Part 7: Exotic Engine - Path-Dependent Pricing");
    println!("==========================================================");
    println!("Parameters: S={}, K={}, r={}, vol={}, T=1Y", spot, strike, rate, volatility);
    println!("Steps: {}, Paths: {}\n", n_steps, n_paths);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Reference: Vanilla European
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let vanilla_call = EuropeanVanillaOption::new(strike, expiry, TypeFlag::Call);
    let vanilla_put = EuropeanVanillaOption::new(strike, expiry, TypeFlag::Put);
    let v_call = vanilla_call.price_monte_carlo(&gbm, &config, rate);
    let v_put = vanilla_put.price_monte_carlo(&gbm, &config, rate);

    println!("--- Reference: Vanilla European ---");
    println!("  Call: {:.4}", v_call);
    println!("  Put:  {:.4}", v_put);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 1. Asian Options
    //
    // The payoff depends on the average price over the option's life.
    // Asian options are cheaper than vanilla because averaging reduces
    // effective volatility (a key insight from Joshi).
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- 1. Asian Options ---");

    let asian_arith_call = AsianOption::new(
        contract_call.clone(),
        AveragingMethod::ArithmeticDiscrete,
        Some(strike),
    );
    let asian_arith_put = AsianOption::new(
        contract_put.clone(),
        AveragingMethod::ArithmeticDiscrete,
        Some(strike),
    );
    let asian_geo_call = AsianOption::new(
        contract_call.clone(),
        AveragingMethod::GeometricDiscrete,
        Some(strike),
    );

    println!("  Arithmetic Asian Call: {:.4} (vanilla: {:.4})",
        asian_arith_call.price_monte_carlo(&gbm, &config, rate), v_call);
    println!("  Arithmetic Asian Put:  {:.4} (vanilla: {:.4})",
        asian_arith_put.price_monte_carlo(&gbm, &config, rate), v_put);
    println!("  Geometric Asian Call:  {:.4}",
        asian_geo_call.price_monte_carlo(&gbm, &config, rate));

    // Floating strike Asian
    let contract_float = OptionContractBuilder::default()
        .type_flag(TypeFlag::Call)
        .exercise_flag(ExerciseFlag::European { expiry })
        .strike_flag(Some(StrikeFlag::Floating))
        .build()
        .unwrap();

    let asian_float = AsianOption::new(
        contract_float,
        AveragingMethod::ArithmeticDiscrete,
        None,
    );
    println!("  Floating Strike Asian Call: {:.4}",
        asian_float.price_monte_carlo(&gbm, &config, rate));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 2. Barrier Options
    //
    // The payoff depends on whether the price crosses a barrier level.
    // Knock-out options become worthless if the barrier is hit.
    // Knock-in options only activate when the barrier is hit.
    //
    // Key relation: Knock-In + Knock-Out = Vanilla
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- 2. Barrier Options ---");

    let barrier_up = 120.0;
    let barrier_down = 80.0;

    let uo_call = BarrierOption::new(contract_call.clone(), BarrierType::UpAndOut, barrier_up, strike);
    let ui_call = BarrierOption::new(contract_call.clone(), BarrierType::UpAndIn, barrier_up, strike);
    let do_call = BarrierOption::new(contract_call.clone(), BarrierType::DownAndOut, barrier_down, strike);
    let di_call = BarrierOption::new(contract_call.clone(), BarrierType::DownAndIn, barrier_down, strike);

    let uo_price = uo_call.price_monte_carlo(&gbm, &config, rate);
    let ui_price = ui_call.price_monte_carlo(&gbm, &config, rate);
    let do_price = do_call.price_monte_carlo(&gbm, &config, rate);
    let di_price = di_call.price_monte_carlo(&gbm, &config, rate);

    println!("  Barrier Up={}:", barrier_up);
    println!("    Up-and-Out Call:  {:.4}", uo_price);
    println!("    Up-and-In Call:   {:.4}", ui_price);
    println!("    Sum (≈ Vanilla):  {:.4} (vanilla: {:.4})", uo_price + ui_price, v_call);

    println!("  Barrier Down={}:", barrier_down);
    println!("    Down-and-Out Call: {:.4}", do_price);
    println!("    Down-and-In Call:  {:.4}", di_price);
    println!("    Sum (≈ Vanilla):   {:.4} (vanilla: {:.4})", do_price + di_price, v_call);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 3. Lookback Options
    //
    // The payoff depends on the maximum or minimum price during the
    // option's life. These are the most expensive path-dependent options.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- 3. Lookback Options ---");

    let lookback_fixed_call = LookbackOption::new(
        contract_call.clone(),
        Some(strike),
    );
    let lookback_fixed_put = LookbackOption::new(
        contract_put.clone(),
        Some(strike),
    );

    let contract_float_call = OptionContractBuilder::default()
        .type_flag(TypeFlag::Call)
        .exercise_flag(ExerciseFlag::European { expiry })
        .strike_flag(Some(StrikeFlag::Floating))
        .build()
        .unwrap();
    let contract_float_put = OptionContractBuilder::default()
        .type_flag(TypeFlag::Put)
        .exercise_flag(ExerciseFlag::European { expiry })
        .strike_flag(Some(StrikeFlag::Floating))
        .build()
        .unwrap();

    let lookback_float_call = LookbackOption::new(contract_float_call, None);
    let lookback_float_put = LookbackOption::new(contract_float_put, None);

    println!("  Fixed Strike:");
    println!("    Call (max(S_max - K, 0)): {:.4} (vanilla: {:.4})",
        lookback_fixed_call.price_monte_carlo(&gbm, &config, rate), v_call);
    println!("    Put  (max(K - S_min, 0)): {:.4} (vanilla: {:.4})",
        lookback_fixed_put.price_monte_carlo(&gbm, &config, rate), v_put);
    println!("  Floating Strike:");
    println!("    Call (S_T - S_min): {:.4}",
        lookback_float_call.price_monte_carlo(&gbm, &config, rate));
    println!("    Put  (S_max - S_T): {:.4}",
        lookback_float_put.price_monte_carlo(&gbm, &config, rate));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 4. Price Comparison Summary
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- 4. Price Ordering (Calls, K=100) ---");
    println!("  Asian (cheapest - averaging reduces vol)");
    println!("  < Vanilla (standard)");
    println!("  < Lookback (most expensive - optimal hindsight)");
    println!();
    println!("  Barrier In + Barrier Out ≈ Vanilla (parity)");
}
