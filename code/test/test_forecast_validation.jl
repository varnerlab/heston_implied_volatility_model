using Test, Dates, Statistics
include(joinpath(@__DIR__,"..","src","ForecastValidation.jl"))
using .ForecastValidation
@testset "Chronological forecast scoring" begin
    # Session horizons cross weekends and exchange holidays, not capture indices.
    @test advance_session(Date("2026-08-14"),5)==Date("2026-08-21")
    @test advance_session(Date("2026-07-02"),1)==Date("2026-07-06")
    @test advance_session(Date("2026-09-04"),1)==Date("2026-09-08")
    @test first(session_path(Date("2026-08-04"),5))==Date("2026-08-04")
    for (values,y) in [([0.,2.],1.),([-2.,0.,1.,9.],3.),([4.,4.,4.],1.)]
        brute=mean(abs.(values.-y))-.5mean(abs(a-b) for a in values,b in values)
        @test distribution_scores(values,y).crps ≈ brute
    end
    @test distribution_scores([4.,4.,4.],1.).crps==3
    surface=fill(.09,3,2,2);surface[2,:,1].=.18;surface[3,:,2].=.045
    target=anchored_targets(surface,[.16,.25])
    @test target[1,1,:]==[.16,.25]
    @test target[2,1,1]≈.32
    @test target[3,1,2]≈.125
    @test surface[1,1,1]==.09
end
