using Test
using Flux

ENV["GKSwstype"] = "100"  # headless GR for the Plots dependency

include(joinpath(@__DIR__, "..", "src", "ScenarioTemplate.jl"))

@testset "Scenario cache validation" begin
    @testset "_cache_matches — full match" begin
        ref = Dict{String,Any}("S_0" => 100.0, "seed" => 20260429, "n_paths" => 1000)
        cache = merge(ref, Dict{String,Any}("v_put_paths" => zeros(2, 2)))
        @test ScenarioTemplate._cache_matches(cache, ref)
    end

    @testset "_cache_matches — missing key invalidates" begin
        ref = Dict{String,Any}("S_0" => 100.0, "seed" => 20260429, "n_paths" => 1000)
        cache = Dict{String,Any}("S_0" => 100.0, "seed" => 20260429)
        @test !ScenarioTemplate._cache_matches(cache, ref)
    end

    @testset "_cache_matches — drifted value invalidates" begin
        ref = Dict{String,Any}("S_0" => 100.0, "seed" => 20260429, "n_paths" => 1000)
        @test !ScenarioTemplate._cache_matches(merge(ref, Dict{String,Any}("seed" => 1)), ref)
        @test !ScenarioTemplate._cache_matches(merge(ref, Dict{String,Any}("n_paths" => 500)), ref)
    end

    @testset "_cache_matches — floating-point tolerance" begin
        ref = Dict{String,Any}("S_0" => 100.0)
        @test ScenarioTemplate._cache_matches(Dict{String,Any}("S_0" => 100.0 * (1 + 1e-14)), ref)
    end

    @testset "_cache_matches — string values compare exactly" begin
        ref = Dict{String,Any}("psi_src" => "per-ticker")
        @test ScenarioTemplate._cache_matches(Dict{String,Any}("psi_src" => "per-ticker"), ref)
        @test !ScenarioTemplate._cache_matches(Dict{String,Any}("psi_src" => "sector"), ref)
    end

    @testset "_psi_checksum — detects weight changes" begin
        nn = Chain(Dense(2 => 4, tanh), Dense(4 => 1))
        c1 = ScenarioTemplate._psi_checksum(nn)
        nn[1].weight[1, 1] += 0.5
        c2 = ScenarioTemplate._psi_checksum(nn)
        @test c1 != c2
    end
end
