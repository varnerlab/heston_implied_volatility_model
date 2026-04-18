using Test
using HestonIV

@testset "HestonIV" begin
    include("test_types.jl")
    include("test_theta_function.jl")
    include("test_heston_variance.jl")
    include("test_crr_tree.jl")
    include("test_calibration.jl")
end
