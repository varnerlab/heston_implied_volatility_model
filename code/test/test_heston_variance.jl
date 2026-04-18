using Test
using HestonIV
using Random
using Statistics

@testset "HestonVariance" begin
    @testset "variance_to_iv — sqrt with floor at 0" begin
        @test variance_to_iv(0.04) ≈ 0.2
        @test variance_to_iv(0.0) ≈ 0.0
        @test variance_to_iv(-0.01) ≈ 0.0  # floored
    end

    @testset "variance_path_to_iv — elementwise" begin
        v = [0.04, 0.09, 0.0, -0.001]
        iv = variance_path_to_iv(v)
        @test iv ≈ [0.2, 0.3, 0.0, 0.0]
    end

    @testset "simulate_variance — initial v₀ equals θ(t=0)" begin
        params = HestonParameters(5.0, 0.3)
        θ_func = ThetaHybrid([0.04, 0.09], [0.0, 0.0, 0.0, 0.0], 0.0)
        n_steps = 30
        hmm_states = fill(2, n_steps)  # state 2 throughout
        S_path = fill(100.0, n_steps)
        contract = OptionContract(100.0, 30, :call, :american)
        mood = zeros(n_steps)

        v = simulate_variance(params, θ_func, hmm_states, S_path, contract, mood;
                              rng=MersenneTwister(0))
        # ATM, no mood, ψ=1 → v[1] = θ_states[2] = 0.09
        @test v[1] ≈ 0.09
        @test length(v) == n_steps
    end

    @testset "simulate_variance — non-negativity (reflecting boundary)" begin
        params = HestonParameters(5.0, 1.0)  # high vol-of-vol
        θ_func = ThetaHybrid([0.04], [0.0, 0.0, 0.0, 0.0], 0.0)
        n_steps = 100
        hmm_states = fill(1, n_steps)
        S_path = fill(100.0, n_steps)
        contract = OptionContract(100.0, n_steps, :call, :american)
        mood = zeros(n_steps)

        v = simulate_variance(params, θ_func, hmm_states, S_path, contract, mood;
                              rng=MersenneTwister(42))
        @test all(v .>= 0.0)
    end

    @testset "simulate_variance — zero σ_v collapses to deterministic" begin
        params = HestonParameters(5.0, 0.0)  # no stochastic shocks
        θ_func = ThetaHybrid([0.04], [0.0, 0.0, 0.0, 0.0], 0.0)
        n_steps = 50
        hmm_states = fill(1, n_steps)
        S_path = fill(100.0, n_steps)
        contract = OptionContract(100.0, 50, :call, :american)
        mood = zeros(n_steps)

        v1 = simulate_variance(params, θ_func, hmm_states, S_path, contract, mood;
                               rng=MersenneTwister(1))
        v2 = simulate_variance(params, θ_func, hmm_states, S_path, contract, mood;
                               rng=MersenneTwister(999))
        @test v1 ≈ v2  # RNG shouldn't matter
        # Starting at θ with target=θ, the deterministic path stays at θ
        @test all(isapprox.(v1, 0.04))
    end

    @testset "simulate_variance — reproducible with same seed" begin
        params = HestonParameters(5.0, 0.3)
        θ_func = ThetaHybrid([0.04], [0.0, 0.0, 0.0, 0.0], 0.0)
        n_steps = 20
        hmm_states = fill(1, n_steps)
        S_path = fill(100.0, n_steps)
        contract = OptionContract(100.0, 20, :call, :american)
        mood = zeros(n_steps)

        v1 = simulate_variance(params, θ_func, hmm_states, S_path, contract, mood;
                               rng=MersenneTwister(123))
        v2 = simulate_variance(params, θ_func, hmm_states, S_path, contract, mood;
                               rng=MersenneTwister(123))
        @test v1 ≈ v2
    end

    @testset "simulate_variance_ensemble — shape and reproducibility" begin
        params = HestonParameters(5.0, 0.3)
        θ_func = ThetaHybrid([0.04], [0.0, 0.0, 0.0, 0.0], 0.0)
        n_paths, n_steps = 4, 20
        hmm_states_mat = fill(1, n_paths, n_steps)
        S_paths_mat = fill(100.0, n_paths, n_steps)
        contract = OptionContract(100.0, 20, :call, :american)
        mood = zeros(n_steps)

        V = simulate_variance_ensemble(params, θ_func, hmm_states_mat, S_paths_mat,
                                       contract, mood; n_var_paths=3,
                                       rng=MersenneTwister(7))
        @test size(V) == (n_paths * 3, n_steps)
        @test all(V .>= 0.0)
    end
end
