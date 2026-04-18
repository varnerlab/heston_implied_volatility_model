using Test
using HestonIV

# Note: prepare_calibration_data depends on a fitted JumpHMM model and price
# series. Tests here construct CalibrationData directly so we exercise the
# optimizer (calibrate) without an HMM fixture.

@testset "Calibration" begin
    @testset "CalibrationData — constructor" begin
        cd = CalibrationData(
            [1, 2, 3],            # dates
            [95.0, 100.0, 105.0], # strikes
            [30, 30, 30],         # dtes
            [0.22, 0.20, 0.21],   # market_ivs
            [100.0, 100.0, 100.0],# spot_prices
            [1, 1, 1],            # hmm_states
            [0.0, 0.0, 0.0],      # moods
        )
        @test length(cd.market_ivs) == 3
        @test cd.strikes[2] == 100.0
    end

    @testset "calibrate — recovers θ on synthetic ATM data" begin
        # Build perfectly flat ATM observations: σ = 0.20 ⇒ θ should ≈ 0.04
        n = 50
        cd = CalibrationData(
            collect(1:n),
            fill(100.0, n),
            fill(30, n),
            fill(0.20, n),
            fill(100.0, n),
            fill(1, n),
            fill(0.0, n),
        )
        _, θ_func = calibrate(cd, 1; maxiter=2000)
        # ψ at ATM with any β reduces to a function of DTE only;
        # the fit can absorb that into β₁ or θ. Either way, model IV ≈ market IV.
        # Check the fitted IV matches:
        DTE = 30.0
        s_t = 1
        moneyness = 1.0
        mood = 0.0
        θ_t = compute_theta(θ_func, s_t, DTE, moneyness, mood)
        σ_model = sqrt(θ_t)
        @test isapprox(σ_model, 0.20; atol=0.01)
    end

    @testset "calibrate — fits per-state θ from synthetic two-regime data" begin
        # Two regimes: state 1 → IV 0.15, state 2 → IV 0.30
        n_per = 40
        n = 2 * n_per
        states = vcat(fill(1, n_per), fill(2, n_per))
        ivs = vcat(fill(0.15, n_per), fill(0.30, n_per))
        cd = CalibrationData(
            collect(1:n),
            fill(100.0, n),
            fill(30, n),
            ivs,
            fill(100.0, n),
            states,
            fill(0.0, n),  # mood off so γ doesn't enter
        )
        _, θ_func = calibrate(cd, 2; maxiter=3000)
        # Reconstruct fitted IV per state at ATM
        DTE, moneyness, mood = 30.0, 1.0, 0.0
        σ1 = sqrt(compute_theta(θ_func, 1, DTE, moneyness, mood))
        σ2 = sqrt(compute_theta(θ_func, 2, DTE, moneyness, mood))
        @test isapprox(σ1, 0.15; atol=0.02)
        @test isapprox(σ2, 0.30; atol=0.02)
    end

    @testset "calibrate — returns valid Heston params (positive κ, σ_v)" begin
        n = 30
        cd = CalibrationData(
            collect(1:n),
            fill(100.0, n),
            fill(30, n),
            fill(0.20, n),
            fill(100.0, n),
            fill(1, n),
            fill(0.0, n),
        )
        heston, θ_func = calibrate(cd, 1; maxiter=500)
        @test heston.κ > 0.0
        @test heston.σ_v > 0.0
        @test all(θ_func.θ_states .> 0.0)
        @test length(θ_func.β) == 4
    end
end
