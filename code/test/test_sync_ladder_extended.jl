using Test
using Dates

include(joinpath(@__DIR__, "..", "scripts", "sync_ladder_extended.jl"))
using .SyncLadderExtended

@testset "sync_ladder_extended" begin

    @testset "dirname conversion" begin
        @test sdk_to_project_dirname("options-05-12-26") == "options-05-12-2026"
        @test sdk_to_project_dirname("options-07-17-26") == "options-07-17-2026"
        # Non-date directories in the SDK data/ folder must be ignored.
        @test sdk_to_project_dirname("options") === nothing
        @test sdk_to_project_dirname("options-partial") === nothing
        # Already-converted four-digit form is not an SDK name.
        @test sdk_to_project_dirname("options-05-12-2026") === nothing
    end

    @testset "date parsing" begin
        @test sdk_dir_date("options-05-12-26") == Date(2026, 5, 12)
        @test sdk_dir_date("options-partial") === nothing
    end

    @testset "frozen cutoff" begin
        # Strictly after 2026-05-11 syncs.
        @test should_sync("options-05-12-26")
        @test should_sync("options-07-17-26")
        # The cutoff date itself is the last frozen day.
        @test !should_sync("options-05-11-26")
        @test !should_sync("options-04-14-26")
        # 04-20 is the partial 23-ticker capture, already held out.
        @test !should_sync("options-04-20-26")
        @test !should_sync("options-partial")
    end

    @testset "copies only post-cutoff dirs" begin
        mktempdir() do tmp
            sdk = joinpath(tmp, "sdk"); dest = joinpath(tmp, "dest")
            for d in ["options-05-11-26", "options-05-12-26", "options-partial"]
                mkpath(joinpath(sdk, d))
                write(joinpath(sdk, d, "AAPL_dte_ladder_x.csv"), "a\n1\n")
            end
            copied = sync_extended(sdk_dir=sdk, dest_dir=dest)
            @test copied == ["options-05-12-2026"]
            @test isfile(joinpath(dest, "options-05-12-2026", "AAPL_dte_ladder_x.csv"))
            @test !isdir(joinpath(dest, "options-05-11-2026"))
        end
    end

    @testset "idempotent" begin
        mktempdir() do tmp
            sdk = joinpath(tmp, "sdk"); dest = joinpath(tmp, "dest")
            mkpath(joinpath(sdk, "options-05-12-26"))
            write(joinpath(sdk, "options-05-12-26", "AAPL_dte_ladder_x.csv"), "a\n1\n")
            sync_extended(sdk_dir=sdk, dest_dir=dest)
            second = sync_extended(sdk_dir=sdk, dest_dir=dest)
            @test isempty(second)
        end
    end

    @testset "refuses to write into the frozen root" begin
        mktempdir() do tmp
            sdk = joinpath(tmp, "sdk"); mkpath(sdk)
            @test_throws ErrorException sync_extended(
                sdk_dir=sdk, dest_dir=joinpath(tmp, "data", "ladder"))
            # Path-normalization bypasses: these all resolve to the same
            # frozen root as "data/ladder" even though they don't string-match
            # the raw guard, so they must be rejected too.
            @test_throws ErrorException sync_extended(
                sdk_dir=sdk, dest_dir=joinpath(tmp, "data", "ladder") * "/")
            @test_throws ErrorException sync_extended(
                sdk_dir=sdk, dest_dir=joinpath(tmp, "data", "ladder", "."))
            @test_throws ErrorException sync_extended(
                sdk_dir=sdk, dest_dir=joinpath(tmp, "data", "ladder", "..", "ladder"))
            # The legitimate destination shares the "ladder" prefix but is
            # not the frozen root itself, so it must be allowed.
            @test isempty(sync_extended(
                sdk_dir=sdk, dest_dir=joinpath(tmp, "data", "ladder_extended")))
        end
    end

    @testset "returns chronological order across a year boundary" begin
        mktempdir() do tmp
            sdk = joinpath(tmp, "sdk"); dest = joinpath(tmp, "dest")
            # Lexical sort would put "options-01-05-27" before
            # "options-12-15-26" (month digit "0" < "1"); chronological
            # order must put it after.
            for d in ["options-01-05-27", "options-12-15-26"]
                mkpath(joinpath(sdk, d))
                write(joinpath(sdk, d, "AAPL_dte_ladder_x.csv"), "a\n1\n")
            end
            copied = sync_extended(sdk_dir=sdk, dest_dir=dest)
            @test copied == ["options-12-15-2026", "options-01-05-2027"]
        end
    end
end
