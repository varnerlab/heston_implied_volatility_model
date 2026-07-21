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

    @testset "calibrate — κ, σ_v pass through unchanged (not identified)" begin
        # Under the equilibrium initialization v₀ = θ(t=0), the static IV
        # objective never evaluates κ or σ_v; calibrate must return them
        # exactly as supplied rather than pretending to estimate them.
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
        heston, θ_func = calibrate(cd, 1; κ_init=7.5, σv_init=0.42, maxiter=500)
        @test heston.κ == 7.5
        @test heston.σ_v == 0.42
        @test all(θ_func.θ_states .> 0.0)
        @test length(θ_func.β) == 5
    end

    @testset "calibrate — γ constrained non-negative" begin
        # mood=1 rows carry LOWER IV than mood=0 rows in the same state, so an
        # unconstrained fit would push γ below zero; the model requires γ ≥ 0.
        n_per = 30
        n = 2 * n_per
        cd = CalibrationData(
            collect(1:n),
            fill(100.0, n),
            fill(30, n),
            vcat(fill(0.30, n_per), fill(0.15, n_per)),
            fill(100.0, n),
            fill(1, n),
            vcat(fill(0.0, n_per), fill(1.0, n_per)),
        )
        _, θ_func = calibrate(cd, 1; maxiter=3000)
        @test θ_func.γ >= 0.0
    end

    @testset "initialize_theta_states — mean IV² per state" begin
        cd = CalibrationData(
            [1, 2, 3],
            fill(100.0, 3),
            fill(30, 3),
            [0.10, 0.30, 0.25],
            fill(100.0, 3),
            [1, 1, 2],
            zeros(3),
        )
        θ0 = HestonIV.initialize_theta_states(cd, 3)
        @test θ0[1] ≈ (0.10^2 + 0.30^2) / 2   # mean, not last-observation overwrite
        @test θ0[2] ≈ 0.25^2
        @test θ0[3] ≈ 0.04                     # default for states with no observations
    end

    @testset "state_index_for_observation — return ending at t" begin
        # state_sequence[i] classifies the return over prices[i] → prices[i+1],
        # so an observation at price index t must use state_sequence[t-1]; using
        # state_sequence[t] would leak the FOLLOWING day's return state.
        @test HestonIV.state_index_for_observation(3, 4) == 2
        @test HestonIV.state_index_for_observation(5, 4) == 4
        @test HestonIV.state_index_for_observation(1, 4) == 1  # no completed return yet
        @test HestonIV.state_index_for_observation(9, 4) == 4  # clamp at final return
    end
end
