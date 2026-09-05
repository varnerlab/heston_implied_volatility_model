using Test
using HestonIV

@testset "HestonIV" begin
    include("test_types.jl")
    include("test_theta_function.jl")
    include("test_heston_variance.jl")
    include("test_crr_tree.jl")
    include("test_calibration.jl")
end

@testset "Corpus tooling" begin
    include("test_sync_ladder_extended.jl")
    include("test_temporal_folds_roots.jl")
    include("test_promote_figures.jl")
end

@testset "Baselines and scenario tooling" begin
    include("test_sabr.jl")
    include("test_scenario_cache.jl")
    include("test_dynamic_ablation.jl")
end
