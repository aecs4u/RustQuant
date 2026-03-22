// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Joshi "Concepts and Practice of Mathematical Finance" - Part 7
// SimpleMC3 through SimpleMC5: PayOff Classes and Custom Payoffs
//
// Joshi introduces polymorphic PayOff classes. RustQuant's `Payoff` trait
// serves the same purpose. This example demonstrates:
//   1. Vanilla call/put payoffs
//   2. Digital (binary) payoffs
//   3. Double digital payoffs (range binary)
//   4. Power payoffs
//   5. Custom payoff via the Payoff trait
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

    let gbm = GeometricBrownianMotion::new(rate, volatility);
    let config = StochasticProcessConfig::new(
        spot, 0.0, 1.0, 252,
        StochasticScheme::EulerMaruyama,
        100_000, true, None,
    );

    println!("==========================================================");
    println!("Joshi Part 7: PayOff Classes (SimpleMC3-5)");
    println!("==========================================================");
    println!("Parameters: S={}, K={}, r={}, vol={}, T=1Y\n", spot, strike, rate, volatility);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 1. Vanilla Call and Put (Joshi PayOffCall / PayOffPut)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("--- 1. Vanilla Options ---");

    let call = EuropeanVanillaOption::new(strike, expiry, TypeFlag::Call);
    let put = EuropeanVanillaOption::new(strike, expiry, TypeFlag::Put);

    let call_price = call.price_monte_carlo(&gbm, &config, rate);
    let put_price = put.price_monte_carlo(&gbm, &config, rate);

    println!("  Call: {:.4}", call_price);
    println!("  Put:  {:.4}", put_price);
    println!("  Put-Call Parity check: C-P = {:.4}, S-K*exp(-rT) = {:.4}",
        call_price - put_price,
        spot - strike * (-rate * 1.0_f64).exp()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 2. Digital / Binary Options (Joshi PayOffDigitalCall / PayOffDigitalPut)
    //
    // Cash-or-nothing: pays K if ITM, 0 otherwise
    // Asset-or-nothing: pays S_T if ITM, 0 otherwise
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- 2. Digital (Binary) Options ---");

    let contract = OptionContractBuilder::default()
        .type_flag(TypeFlag::Call)
        .exercise_flag(ExerciseFlag::European { expiry })
        .strike_flag(Some(StrikeFlag::Fixed))
        .build()
        .unwrap();

    let cash_call = BinaryOption::new(contract.clone(), BinaryType::CashOrNothing, strike);
    let asset_call = BinaryOption::new(contract.clone(), BinaryType::AssetOrNothing, strike);

    println!("  Cash-or-Nothing Call: {:.4}", cash_call.price_monte_carlo(&gbm, &config, rate));
    println!("  Asset-or-Nothing Call: {:.4}", asset_call.price_monte_carlo(&gbm, &config, rate));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 3. Double Digital (Supershare) - pays S_T/K_low if K_low < S_T < K_high
    //
    // This is Joshi's PayOffDoubleDigital. RustQuant implements this as
    // the SupershareOption.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- 3. Double Digital / Supershare ---");

    let supershare = SupershareOption::new(90.0, 110.0);
    let ss_price = supershare.price_monte_carlo(&gbm, &config, rate);
    println!("  Supershare (K_low=90, K_high=110): {:.4}", ss_price);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 4. Power Options
    //
    // Power contract: (S_T/K)^n
    // Power option:   max(S_T^n - K, 0)
    // Capped power:   min(max(S_T^n - K, 0), cap)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- 4. Power Options ---");

    let power_contract = PowerContract::new(contract.clone(), strike, 2.0);
    let power_option = PowerOption::new(contract.clone(), strike, 2.0);

    println!("  Power Contract (n=2): {:.4}", power_contract.price_monte_carlo(&gbm, &config, rate));
    println!("  Power Option (n=2):   {:.4}", power_option.price_monte_carlo(&gbm, &config, rate));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 5. Log Contracts
    //
    // Log moneyness: ln(S_T / K)
    // Log option:    max(ln(S_T / K), 0)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- 5. Log Contracts ---");

    let log_moneyness = LogMoneynessContract::new(strike);
    let log_option = LogOption::new(strike);

    println!("  Log Moneyness Contract: {:.4}", log_moneyness.price_monte_carlo(&gbm, &config, rate));
    println!("  Log Option:             {:.4}", log_option.price_monte_carlo(&gbm, &config, rate));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 6. Strike sensitivity (like Joshi's varying strike)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    println!("\n--- 6. Strike Sensitivity ---");
    println!("  {:>8} {:>12} {:>12} {:>12}", "Strike", "Vanilla", "Digital", "Supershare");

    for k in (80..=120).step_by(10) {
        let k = k as f64;
        let v = EuropeanVanillaOption::new(k, expiry, TypeFlag::Call);

        let contract_k = OptionContractBuilder::default()
            .type_flag(TypeFlag::Call)
            .exercise_flag(ExerciseFlag::European { expiry })
            .strike_flag(Some(StrikeFlag::Fixed))
            .build()
            .unwrap();

        let d = BinaryOption::new(contract_k, BinaryType::CashOrNothing, k);
        let ss = SupershareOption::new(k - 5.0, k + 5.0);

        println!("  {:>8.0} {:>12.4} {:>12.4} {:>12.4}",
            k,
            v.price_monte_carlo(&gbm, &config, rate),
            d.price_monte_carlo(&gbm, &config, rate),
            ss.price_monte_carlo(&gbm, &config, rate),
        );
    }
}
