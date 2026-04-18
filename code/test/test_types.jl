using Test
using HestonIV

@testset "Types" begin
    @testset "HestonParameters" begin
        p = HestonParameters(5.0, 0.3)
        @test p.κ == 5.0
        @test p.σ_v == 0.3
    end

    @testset "ThetaHybrid" begin
        θ = ThetaHybrid([0.04, 0.09], [0.1, -0.5, 0.0, 0.2], 0.3)
        @test length(θ.θ_states) == 2
        @test length(θ.β) == 4
        @test θ.γ == 0.3
    end

    @testset "OptionContract" begin
        c = OptionContract(100.0, 30, :call, :american)
        @test c.K == 100.0
        @test c.DTE == 30
        @test c.option_type == :call
        @test c.style == :american
    end

    @testset "IVPoint" begin
        pt = IVPoint(105.0, 45, 0.22, 100.0)
        @test pt.K == 105.0
        @test pt.DTE == 45
        @test pt.σ_imp == 0.22
        @test pt.S == 100.0
    end
end
