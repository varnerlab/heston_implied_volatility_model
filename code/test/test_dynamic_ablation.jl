using Test, Statistics
include(joinpath(@__DIR__, "..", "src", "DynamicAblation.jl"))
using .DynamicAblation

@testset "Dynamic IV ablation invariants" begin
    target = fill(0.09,4,3,2)
    zs = [-1.0 0.0 1.0; 0.5 -0.5 0.0; 0.0 1.0 -1.0]
    zi = reverse(zs;dims=2)
    for mode in MODES
        v = variance_paths(target,zs,zi;mode)
        @test v[1,:,:] == target[1,:,:]
        @test v[:,:,1] == v[:,:,2] # identical contracts share innovations
        @test all(isfinite,v) && minimum(v)>0
    end
    @test variance_paths(target,zs,zi;mode=:relaxation) == target
    @test variance_paths(target,zs,zi;mode=:coupled,sigma_v=0) == target
    @test variance_paths(target,zs,zi;mode=:coupled,rho=0) ==
          variance_paths(target,zs,zi;mode=:uncoupled)
    target[2:end,:,2] .= 0.16
    both = variance_paths(target,zs,zi;mode=:coupled)
    one = variance_paths(target[:,:,1:1],zs,zi;mode=:coupled)
    @test both[:,:,1] == one[:,:,1] # adding a contract does not alter existing paths
    @test variance_paths(target,zs,zi;mode=:surface) == target
    @test all(variance_paths(target,zs,zi;mode=:frozen) .== 0.09)
    @test both[2,1,2] == both[2,1,1] # start-of-transition target, not next-date target
    @test_throws ArgumentError variance_paths(target,zs,zi;mode=:unknown)
    @test_throws DimensionMismatch variance_paths(target,zs[1:2,:],zi;mode=:coupled)

    # Expiry is independent of IV, and the low-IV fallback respects exercise.
    @test option_mark(80.0,100.0,0.01,0,:put) == 20.0
    @test option_mark(80.0,100.0,1.0,0,:put) == 20.0
    @test option_mark(80.0,100.0,0.005^2,10,:put) >= 20.0
    @test option_mark(100.0,100.0,0.09,31,:put;depth=201) ≈
          option_mark(100.0,100.0,0.09,31,:put;depth=401) atol=0.01
    s=paired_summary([2.0,3.0,4.0],[1.0,2.0,3.0],5.0)
    @test s.mean_pnl_change == -1.0
    @test s.paired_se == 0
    @test s.mean_abs_mark_change == 1.0
    checks=strike_checks([20.0,10.0,0.0],[80.0,90.0,100.0],100.0,:call)
    @test all(x->x.violations==0,checks)
    checks=strike_checks([20.0,16.0,0.0],[80.0,90.0,100.0],100.0,:call)
    @test only(filter(x->x.check=="convexity",checks)).violations == 1
end
