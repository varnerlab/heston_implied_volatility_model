using Test
using Flux
using Dates

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

    @testset "calendar and trading clocks remain distinct" begin
        spec = ScenarioTemplate.ScenarioSpec(
            ticker="GS", anchor_date=Date("2026-04-28"),
            expiry_date=Date("2026-05-29"),
            K_put=890.0, K_call=970.0,
            market_premium_put=16.51, market_premium_call=16.085,
            market_iv_put=0.3125, market_iv_call=0.2893,
            market_delta_put=-0.2951, market_delta_call=0.3278,
            expiry_label="2026-05-29", ticker_prior_ccgr_pct=10.0,
            market_holidays=[Date("2026-05-25")],
        )
        dates = ScenarioTemplate._trading_dates(spec)
        dtes = ScenarioTemplate._calendar_dtes(spec, dates)
        @test length(dates) - 1 == 22
        @test first(dtes) == 31
        @test last(dtes) == 0
        @test Date("2026-05-25") ∉ dates
    end

    # Helpers for the paper_figure_dirs rules below.
    _variant(tmp, v) = (mkpath(joinpath(tmp, v, "sections")); joinpath(tmp, v))
    _cite(tmp, v, sub) = write(joinpath(tmp, v, "sections", "body.tex"),
                               "\\includegraphics{sections/figures/$sub/x.pdf}")
    _figdir(tmp, v, sub) = mkpath(joinpath(tmp, v, "sections", "figures", sub))
    _target(tmp, v, sub) = joinpath(tmp, v, "sections", "figures", sub)

    @testset "paper_figure_dirs returns every variant that cites the subdir" begin
        mktempdir() do tmp
            for v in ("paper-arxiv", "paper-jcf")
                _variant(tmp, v); _cite(tmp, v, "gs")
            end
            dirs = ScenarioTemplate.paper_figure_dirs("gs"; repo_root=tmp)
            @test Set(dirs) == Set([_target(tmp, "paper-arxiv", "gs"),
                                    _target(tmp, "paper-jcf", "gs")])
        end
    end

    @testset "paper_figure_dirs also keeps an existing copy fresh" begin
        # One variant cites the figures, the other merely holds a copy from an
        # earlier run. Both must be written, or the second one goes stale.
        mktempdir() do tmp
            for v in ("paper-arxiv", "paper-jcf"); _variant(tmp, v); end
            _cite(tmp, "paper-arxiv", "lly")
            _figdir(tmp, "paper-jcf", "lly")
            dirs = ScenarioTemplate.paper_figure_dirs("lly"; repo_root=tmp)
            @test Set(dirs) == Set([_target(tmp, "paper-arxiv", "lly"),
                                    _target(tmp, "paper-jcf", "lly")])
        end
    end

    @testset "paper_figure_dirs does not duplicate uncited scratch panels" begin
        # The expanded cross-ticker and INTC panels are cited by neither paper.
        # They belong only where they already live.
        mktempdir() do tmp
            for v in ("paper-arxiv", "paper-jcf"); _variant(tmp, v); end
            _figdir(tmp, "paper-jcf", "spy")
            @test ScenarioTemplate.paper_figure_dirs("spy"; repo_root=tmp) ==
                  [_target(tmp, "paper-jcf", "spy")]
        end
    end

    @testset "paper_figure_dirs falls back for a brand-new figure set" begin
        mktempdir() do tmp
            for v in ("paper-arxiv", "paper-jcf"); _variant(tmp, v); end
            @test ScenarioTemplate.paper_figure_dirs("newticker"; repo_root=tmp) ==
                  [_target(tmp, "paper-arxiv", "newticker")]
        end
    end

    @testset "paper_figure_dirs skips absent variants" begin
        mktempdir() do tmp
            _variant(tmp, "paper-jcf"); _cite(tmp, "paper-jcf", "lly")
            @test ScenarioTemplate.paper_figure_dirs("lly"; repo_root=tmp) ==
                  [_target(tmp, "paper-jcf", "lly")]
        end
    end

    @testset "paper_figure_dirs fails loudly when no variant exists" begin
        mktempdir() do tmp
            @test_throws ErrorException ScenarioTemplate.paper_figure_dirs("gs"; repo_root=tmp)
        end
    end

    @testset "figure mirroring copies the prefixed set only" begin
        mktempdir() do tmp
            src = joinpath(tmp, "src"); mkpath(src)
            for f in ("gs_short_paths.pdf", "gs_short_paths.png", "gs_iv_trajectories.pdf")
                write(joinpath(src, f), "fresh-$f")
            end
            write(joinpath(src, "lly_short_paths.pdf"), "other ticker")
            write(joinpath(src, "gs_notes.txt"), "not a figure")

            dst = joinpath(tmp, "dst")
            # A stale copy in the destination must be overwritten, since that is
            # exactly the paper-jcf failure this mirroring exists to prevent.
            mkpath(dst)
            write(joinpath(dst, "gs_short_paths.pdf"), "stale")

            names = ScenarioTemplate._mirror_figures(src, [dst], "gs")
            @test Set(names) == Set(["gs_short_paths.pdf", "gs_short_paths.png",
                                     "gs_iv_trajectories.pdf"])
            @test read(joinpath(dst, "gs_short_paths.pdf"), String) == "fresh-gs_short_paths.pdf"
            @test isfile(joinpath(dst, "gs_iv_trajectories.pdf"))
            @test !isfile(joinpath(dst, "lly_short_paths.pdf"))
            @test !isfile(joinpath(dst, "gs_notes.txt"))
        end
    end

    @testset "figure mirroring is a no-op with a single destination" begin
        mktempdir() do tmp
            src = joinpath(tmp, "src"); mkpath(src)
            write(joinpath(src, "gs_short_paths.pdf"), "x")
            @test ScenarioTemplate._mirror_figures(src, String[], "gs") == String[]
        end
    end

    @testset "Wilson interval" begin
        lo, hi = ScenarioTemplate._wilson_interval(50, 100)
        @test lo ≈ 0.4038 atol=1e-4
        @test hi ≈ 0.5962 atol=1e-4
    end
end
