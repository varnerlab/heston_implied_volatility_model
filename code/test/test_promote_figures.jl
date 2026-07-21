using Test

include(joinpath(@__DIR__, "..", "scripts", "promote_figures.jl"))

# Build a miniature repo: code/figures/ plus two paper variants whose .tex
# files reference overlapping-but-different figure sets.
function _fixture(tmp; arxiv_refs, jcf_refs, flat_sources)
    figs = joinpath(tmp, "code", "figures")
    mkpath(figs)
    for f in flat_sources
        write(joinpath(figs, f), "%PDF-$f\n")
    end
    for (variant, refs) in ("paper-arxiv" => arxiv_refs, "paper-jcf" => jcf_refs)
        sections = joinpath(tmp, variant, "sections")
        mkpath(joinpath(sections, "figures"))
        body = join(["\\includegraphics[width=\\linewidth]{sections/figures/$r}" for r in refs], "\n")
        write(joinpath(sections, "body.tex"), body * "\n")
    end
    return figs
end

_run(tmp, figs; kwargs...) = promote_figures(; fig_dir=figs, repo_root=tmp, verbose=false, kwargs...)

@testset "promote_figures" begin

    @testset "reference scanning handles nested paths" begin
        # The pre-fix regex excluded '/', so every sections/figures/<ticker>/<f>.pdf
        # reference was invisible. That is how gs/ and lly/ went unmanaged.
        mktempdir() do tmp
            sections = joinpath(tmp, "sections")
            mkpath(sections)
            write(joinpath(sections, "a.tex"), """
            \\includegraphics{sections/figures/flat.pdf}
            \\includegraphics[width=0.9\\linewidth]{sections/figures/gs/gs_short_paths.pdf}
            \\includegraphics{sections/figures/intc_earnings/panel.pdf}
            """)
            refs = _referenced_pdfs(sections)
            @test refs == ["flat.pdf", "gs/gs_short_paths.pdf", "intc_earnings/panel.pdf"]
        end
    end

    @testset "flat figures promote to both variants" begin
        mktempdir() do tmp
            figs = _fixture(tmp;
                arxiv_refs=["shared.pdf"],
                jcf_refs=["shared.pdf"],
                flat_sources=["shared.pdf"])
            r = _run(tmp, figs)
            for v in ("paper-arxiv", "paper-jcf")
                @test isfile(joinpath(tmp, v, "sections", "figures", "shared.pdf"))
                @test "shared.pdf" in r.promoted[v]
            end
        end
    end

    @testset "each variant gets only what it references" begin
        # paper-jcf carries the LoRA section; paper-arxiv must not inherit it.
        mktempdir() do tmp
            figs = _fixture(tmp;
                arxiv_refs=["shared.pdf"],
                jcf_refs=["shared.pdf", "lora_trigger_roc.pdf"],
                flat_sources=["shared.pdf", "lora_trigger_roc.pdf"])
            _run(tmp, figs)
            @test isfile(joinpath(tmp, "paper-jcf", "sections", "figures", "lora_trigger_roc.pdf"))
            @test !isfile(joinpath(tmp, "paper-arxiv", "sections", "figures", "lora_trigger_roc.pdf"))
        end
    end

    @testset "content is refreshed, not just created" begin
        mktempdir() do tmp
            figs = _fixture(tmp;
                arxiv_refs=["shared.pdf"], jcf_refs=["shared.pdf"],
                flat_sources=["shared.pdf"])
            dest = joinpath(tmp, "paper-arxiv", "sections", "figures", "shared.pdf")
            write(dest, "stale\n")
            _run(tmp, figs)
            @test read(dest, String) == "%PDF-shared.pdf\n"
        end
    end

    @testset "missing flat source is reported, not silently skipped" begin
        mktempdir() do tmp
            figs = _fixture(tmp;
                arxiv_refs=["ghost.pdf"], jcf_refs=["ghost.pdf"],
                flat_sources=String[])
            r = _run(tmp, figs)
            @test "ghost.pdf" in r.missing["paper-arxiv"]
            @test "ghost.pdf" in r.missing["paper-jcf"]
        end
    end

    @testset "scenario-owned nested figures are not sourced from code/figures" begin
        # gs/lly/avgo/intc_earnings are written straight into the paper tree by
        # scenario scripts. A same-named flat file in code/figures must not be
        # mistaken for their source.
        mktempdir() do tmp
            figs = _fixture(tmp;
                arxiv_refs=["gs/gs_short_paths.pdf"],
                jcf_refs=["gs/gs_short_paths.pdf"],
                flat_sources=["gs_short_paths.pdf"])
            for v in ("paper-arxiv", "paper-jcf")
                d = joinpath(tmp, v, "sections", "figures", "gs")
                mkpath(d)
                write(joinpath(d, "gs_short_paths.pdf"), "%PDF-scenario\n")
            end
            r = _run(tmp, figs)
            # The scenario output survives; the like-named flat file is not its source.
            for v in ("paper-arxiv", "paper-jcf")
                @test read(joinpath(tmp, v, "sections", "figures", "gs", "gs_short_paths.pdf"),
                           String) == "%PDF-scenario\n"
                @test isempty(r.promoted[v])
            end
            # It is also not mistaken for a promotable flat reference.
            @test !isfile(joinpath(tmp, "paper-arxiv", "sections", "figures", "gs_short_paths.pdf"))
        end
    end

    @testset "nested figure present in one variant is copied to the other" begin
        # gs_short_premium_simulation.jl writes only into paper-jcf, so
        # paper-arxiv's copy goes stale. Syncing both dirs means closing that gap.
        mktempdir() do tmp
            figs = _fixture(tmp;
                arxiv_refs=["gs/gs_short_paths.pdf"],
                jcf_refs=["gs/gs_short_paths.pdf"],
                flat_sources=String[])
            src = joinpath(tmp, "paper-jcf", "sections", "figures", "gs")
            mkpath(src)
            write(joinpath(src, "gs_short_paths.pdf"), "%PDF-fresh\n")
            r = _run(tmp, figs)
            dest = joinpath(tmp, "paper-arxiv", "sections", "figures", "gs", "gs_short_paths.pdf")
            @test isfile(dest)
            @test read(dest, String) == "%PDF-fresh\n"
            @test "gs/gs_short_paths.pdf" in r.cross_synced
        end
    end

    @testset "nested figure missing everywhere is reported" begin
        mktempdir() do tmp
            figs = _fixture(tmp;
                arxiv_refs=["gs/nowhere.pdf"], jcf_refs=["gs/nowhere.pdf"],
                flat_sources=String[])
            r = _run(tmp, figs)
            @test "gs/nowhere.pdf" in r.missing["paper-arxiv"]
        end
    end

    @testset "unreferenced scratch is the union across variants" begin
        # A figure referenced by only one paper is still "referenced" and must
        # not be listed as stale scratch.
        mktempdir() do tmp
            figs = _fixture(tmp;
                arxiv_refs=["shared.pdf"],
                jcf_refs=["shared.pdf", "jcf_only.pdf"],
                flat_sources=["shared.pdf", "jcf_only.pdf", "scratch.pdf"])
            r = _run(tmp, figs)
            @test r.stale == ["scratch.pdf"]
        end
    end

    @testset "errors loudly when no paper variant exists" begin
        # The silent 0/0 no-op after the paper/ -> paper-arxiv+paper-jcf split
        # went unnoticed for two months. Absent destinations must throw.
        mktempdir() do tmp
            figs = joinpath(tmp, "code", "figures")
            mkpath(figs)
            @test_throws ErrorException promote_figures(; fig_dir=figs, repo_root=tmp, verbose=false)
        end
    end

    @testset "idempotent" begin
        mktempdir() do tmp
            figs = _fixture(tmp;
                arxiv_refs=["shared.pdf"], jcf_refs=["shared.pdf"],
                flat_sources=["shared.pdf"])
            first = _run(tmp, figs)
            second = _run(tmp, figs)
            @test first.promoted == second.promoted
            @test isempty(second.missing["paper-arxiv"])
        end
    end
end
