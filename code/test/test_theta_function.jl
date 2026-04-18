using Test
using HestonIV

@testset "ThetaFunction" begin
    @testset "ψ — zero β reduces to 1" begin
        @test ψ([0.0, 0.0, 0.0, 0.0], 30.0, 1.0) ≈ 1.0
        @test ψ([0.0, 0.0, 0.0, 0.0], 1.0, 1.0) ≈ 1.0
    end

    @testset "ψ — ATM (moneyness=1) collapses skew/curvature/interaction" begin
        # log(1) = 0, so β₂, β₃, β₄ terms all vanish
        β = [0.1, -0.5, 0.3, 0.4]
        atm = ψ(β, 30.0, 1.0)
        # Only β₁·ln(DTE) term remains
        @test atm ≈ exp(0.1 * log(30.0))
    end

    @testset "ψ — DTE floor at 1 day" begin
        # log(0) would diverge; impl floors DTE at 1 so ln(DTE)=0
        β = [0.5, 0.0, 0.0, 0.0]
        @test ψ(β, 0.5, 1.0) ≈ 1.0  # floored to ln(1)=0
        @test ψ(β, 0.0, 1.0) ≈ 1.0
    end

    @testset "ψ — backward compat with 3-element β" begin
        β3 = [0.1, -0.5, 0.3]
        β4 = [0.1, -0.5, 0.3, 0.0]
        @test ψ(β3, 30.0, 1.05) ≈ ψ(β4, 30.0, 1.05)
    end

    @testset "ψ — positive smile curvature elevates wings symmetrically" begin
        β = [0.0, 0.0, 0.0, 0.5]  # only β₄ active
        wing_otm = ψ(β, 30.0, 1.10)
        wing_itm = ψ(β, 30.0, 1.0 / 1.10)  # log-symmetric
        @test wing_otm ≈ wing_itm
        @test wing_otm > 1.0  # both wings elevated above ATM
    end

    @testset "ψ — negative skew (β₂<0) lifts OTM puts" begin
        β = [0.0, -1.0, 0.0, 0.0]
        otm_put = ψ(β, 30.0, 0.90)   # K/S < 1
        otm_call = ψ(β, 30.0, 1.10)  # K/S > 1
        @test otm_put > otm_call  # put skew
    end

    @testset "compute_theta — assembles base × mood × ψ" begin
        θ_states = [0.04, 0.09]
        γ = 0.5
        β = [0.0, 0.0, 0.0, 0.0]  # ψ = 1
        θ_func = ThetaHybrid(θ_states, β, γ)
        # State 1, no mood → base only
        @test compute_theta(θ_func, 1, 30.0, 1.0, 0.0) ≈ 0.04
        # State 2, full mood → base × (1+γ)
        @test compute_theta(θ_func, 2, 30.0, 1.0, 1.0) ≈ 0.09 * 1.5
    end

    @testset "compute_theta — ψ scales the result" begin
        θ_states = [0.04]
        β = [0.0, 0.0, 0.0, 0.5]
        θ_func = ThetaHybrid(θ_states, β, 0.0)
        # OTM call: ψ > 1 lifts θ above base
        θ_otm = compute_theta(θ_func, 1, 30.0, 1.10, 0.0)
        θ_atm = compute_theta(θ_func, 1, 30.0, 1.00, 0.0)
        @test θ_atm ≈ 0.04
        @test θ_otm > θ_atm
    end

    @testset "compute_mood — fraction in tails" begin
        # 5 states, 1 tail at each end ⇒ tail states = {1, 5}
        @test compute_mood([1, 2, 3, 4, 5], 5, 1) ≈ 2 / 5
        @test compute_mood([3, 3, 3], 5, 1) ≈ 0.0
        @test compute_mood([1, 5, 1, 5], 5, 1) ≈ 1.0
        @test compute_mood(Int[], 5, 1) ≈ 0.0  # empty
    end

    @testset "compute_mood — wider tail band" begin
        # 5 states, 2 tail at each end ⇒ tail states = {1, 2, 4, 5}
        @test compute_mood([1, 2, 3, 4, 5], 5, 2) ≈ 4 / 5
    end

    @testset "compute_mood_path — per-timestep" begin
        # 3 tickers × 4 timesteps
        states = [1 2 3 1;
                  3 3 3 5;
                  5 4 3 5]
        # 5 states, n_tail=1 → tails = {1, 5}
        path = compute_mood_path(states, 5, 1)
        @test length(path) == 4
        @test path[1] ≈ 2 / 3  # ticker1=1 (tail), ticker3=5 (tail)
        @test path[2] ≈ 0.0    # all middle
        @test path[3] ≈ 0.0
        @test path[4] ≈ 1.0    # all tails
    end
end
