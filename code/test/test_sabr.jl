using Test

# SABR.jl is a bare-include script dependency (not part of the HestonIV module),
# mirroring how calibrate_sabr_per_day.jl loads it.
include(joinpath(@__DIR__, "..", "src", "SABR.jl"))

@testset "SABR" begin
    @testset "hagan_sabr_iv — ATM branch matches the near-ATM limit" begin
        F, T = 100.0, 0.25
        α, β, ρ, ν = 0.2, 0.5, -0.3, 0.4
        iv_atm = hagan_sabr_iv(F, F, T, α, β, ρ, ν)
        iv_near = hagan_sabr_iv(F, F * (1 + 1e-6), T, α, β, ρ, ν)
        @test isapprox(iv_atm, iv_near; rtol=1e-4)
    end

    @testset "hagan_sabr_iv — ATM matches the Hagan 2002 closed form" begin
        F, T = 100.0, 0.25
        α, β, ρ, ν = 0.2, 0.5, -0.3, 0.4
        FmB = F^(1 - β)
        corrections = ((1 - β)^2 / 24) * α^2 / FmB^2 +
                      (ρ * β * ν * α) / (4 * FmB) +
                      ((2 - 3ρ^2) / 24) * ν^2
        expected = (α / FmB) * (1.0 + corrections * T)
        @test isapprox(hagan_sabr_iv(F, F, T, α, β, ρ, ν), expected; rtol=1e-12)
    end
end
