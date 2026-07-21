"""
    promote_figures(; fig_dir, repo_root, variants, verbose)

Copy the figures each paper variant actually references out of `code/figures/`
and into that variant's `sections/figures/`, so every paper builds against the
latest script outputs.

The repo carries two paper variants — `paper-arxiv/` and `paper-jcf/` — with
overlapping but *different* figure sets (only the JCF variant has the LoRA
section). Each gets exactly what its own `.tex` files cite; nothing more.

Two figure conventions coexist, and they are handled differently:

- **Flat figures** (`sections/figures/<name>.pdf`) are produced into
  `code/figures/` by a calibration or analysis script. `code/figures/` is the
  source of truth and these are copied outward.
- **Nested figures** (`sections/figures/<ticker>/<name>.pdf`) are written
  straight into the paper tree by the scenario scripts
  (`gs_`/`lly_`/`avgo_short_premium_simulation.jl`, `earnings_window_scenario.jl`),
  which never touch `code/figures/`. They are not sourced from `code/figures/`.
  Because those scripts target a single variant, a nested figure can go stale
  in the other one; when a variant is missing a nested figure that another
  variant has, the newest copy is propagated across.

Warns about referenced figures that exist nowhere, and about unreferenced PDFs
accumulating in `code/figures/`. Throws if no paper variant is found at all —
a silent no-op here went unnoticed for two months after the `paper/` →
`paper-arxiv/` + `paper-jcf/` split.

Call at the end of any script that writes to `code/figures/`.
Idempotent — safe to invoke even when nothing changed.
"""

const FIG_DIR = abspath(joinpath(@__DIR__, "..", "figures"))
const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const PAPER_VARIANTS = ["paper-arxiv", "paper-jcf"]

# Reference paths may contain a subdirectory component, e.g.
# sections/figures/gs/gs_short_paths.pdf — the character class must admit '/'.
const _REF_PAT = r"sections/figures/([A-Za-z0-9_][A-Za-z0-9_./\-]*\.pdf)"

"""Figure paths cited by any `.tex` under `sections_dir`, relative to `figures/`."""
function _referenced_pdfs(sections_dir::AbstractString)
    isdir(sections_dir) || return String[]
    refs = Set{String}()
    for (root, _, files) in walkdir(sections_dir)
        for f in files
            endswith(f, ".tex") || continue
            for m in eachmatch(_REF_PAT, read(joinpath(root, f), String))
                push!(refs, m.captures[1])
            end
        end
    end
    return sort(collect(refs))
end

_figures_dir(repo_root, variant) = joinpath(repo_root, variant, "sections", "figures")
_sections_dir(repo_root, variant) = joinpath(repo_root, variant, "sections")

"""Copy `src` over `dst` only when the bytes differ, so re-runs are no-ops."""
function _copy_if_changed(src::AbstractString, dst::AbstractString)
    isfile(dst) && read(dst) == read(src) && return false
    mkpath(dirname(dst))
    cp(src, dst; force=true)
    return true
end

function promote_figures(; fig_dir::AbstractString=FIG_DIR,
                           repo_root::AbstractString=REPO_ROOT,
                           variants::AbstractVector{<:AbstractString}=PAPER_VARIANTS,
                           verbose::Bool=true)
    present = [v for v in variants if isdir(_sections_dir(repo_root, v))]
    if isempty(present)
        error("promote_figures: no paper variant found under $repo_root " *
              "(looked for $(join(variants, ", "))). Figures were NOT promoted. " *
              "If the paper directories moved again, update PAPER_VARIANTS.")
    end

    refs = Dict(v => _referenced_pdfs(_sections_dir(repo_root, v)) for v in present)
    promoted = Dict(v => String[] for v in present)
    missing_refs = Dict(v => String[] for v in present)

    # Flat figures: code/figures/ is the source of truth.
    for v in present, ref in refs[v]
        occursin('/', ref) && continue
        src = joinpath(fig_dir, ref)
        if isfile(src)
            _copy_if_changed(src, joinpath(_figures_dir(repo_root, v), ref))
            push!(promoted[v], ref)
        else
            push!(missing_refs[v], ref)
        end
    end

    # Nested figures: owned by the scenario scripts, which write into one
    # variant only. Propagate the freshest copy to any variant that cites it.
    nested = sort(unique(ref for v in present for ref in refs[v] if occursin('/', ref)))
    cross_synced = String[]
    for ref in nested
        copies = [joinpath(_figures_dir(repo_root, v), ref) for v in present]
        existing = filter(isfile, copies)
        if isempty(existing)
            for v in present
                ref in refs[v] && push!(missing_refs[v], ref)
            end
            continue
        end
        newest = argmax(mtime, existing)
        changed = false
        for v in present
            ref in refs[v] || continue
            dst = joinpath(_figures_dir(repo_root, v), ref)
            dst == newest && continue
            changed |= _copy_if_changed(newest, dst)
        end
        changed && push!(cross_synced, ref)
    end

    # Unreferenced PDFs accumulating in code/figures/. A figure cited by only
    # one variant is still referenced, so the union is the right baseline.
    all_flat = Set(ref for v in present for ref in refs[v] if !occursin('/', ref))
    stale = isdir(fig_dir) ?
        sort([f for f in readdir(fig_dir) if endswith(f, ".pdf") && !(f in all_flat)]) :
        String[]

    if verbose
        for v in present
            n = length(promoted[v]) + length(missing_refs[v])
            println("[promote_figures] $(length(promoted[v]))/$n referenced PDFs synced -> $v/sections/figures/")
        end
        if !isempty(cross_synced)
            println("[promote_figures] propagated $(length(cross_synced)) scenario figure(s) across variants:")
            foreach(f -> println("    - $f"), cross_synced)
        end
        for v in present
            isempty(missing_refs[v]) && continue
            println("[promote_figures] WARNING: $v references figures that exist nowhere:")
            foreach(f -> println("    - $f"), missing_refs[v])
        end
        if !isempty(stale)
            shown = first(stale, 10)
            println("[promote_figures] NOTE: $(length(stale)) unreferenced PDF(s) in code/figures/ (scratch):")
            foreach(f -> println("    - $f"), shown)
            length(stale) > length(shown) && println("    ... and $(length(stale) - length(shown)) more")
        end
    end

    return (promoted=promoted, missing=missing_refs, cross_synced=cross_synced, stale=stale)
end

# Allow direct invocation: `julia --project=. code/scripts/promote_figures.jl`
if abspath(PROGRAM_FILE) == @__FILE__
    promote_figures()
end
