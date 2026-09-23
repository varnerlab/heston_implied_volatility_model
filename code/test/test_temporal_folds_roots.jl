using Test

include(joinpath(@__DIR__, "..", "src", "TemporalFolds.jl"))
using .TemporalFolds

@testset "multi-root day resolution" begin

    @testset "EXTENDED_DAYS shape" begin
        @test length(TemporalFolds.EXTENDED_DAYS) == 58
        # Strictly sorted by date, no duplicates.
        dates = [TemporalFolds.dir_to_date(d) for d in TemporalFolds.EXTENDED_DAYS]
        @test issorted(dates)
        @test length(unique(dates)) == 58
        @test first(TemporalFolds.EXTENDED_DAYS) == "options-04-14-2026"
        @test last(TemporalFolds.EXTENDED_DAYS)  == "options-07-17-2026"
        # The first 15 are exactly the frozen arm, in order.
        @test TemporalFolds.EXTENDED_DAYS[15] == "options-05-11-2026"
        @test TemporalFolds.EXTENDED_DAYS[16] == "options-05-12-2026"
    end

    @testset "resolve_day" begin
        mktempdir() do tmp
            a = joinpath(tmp, "ladder"); b = joinpath(tmp, "ladder_extended")
            mkpath(joinpath(a, "options-05-11-2026"))
            mkpath(joinpath(b, "options-05-12-2026"))

            @test TemporalFolds.resolve_day([a, b], "options-05-11-2026") ==
                  joinpath(a, "options-05-11-2026")
            @test TemporalFolds.resolve_day([a, b], "options-05-12-2026") ==
                  joinpath(b, "options-05-12-2026")

            # Missing in every root is a hard error, not a silent skip.
            @test_throws ErrorException TemporalFolds.resolve_day(
                [a, b], "options-06-01-2026")

            # Present in more than one root is ambiguous and also a hard error.
            mkpath(joinpath(b, "options-05-11-2026"))
            @test_throws ErrorException TemporalFolds.resolve_day(
                [a, b], "options-05-11-2026")
        end
    end
end
