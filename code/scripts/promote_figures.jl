"""
    promote_figures()

Copy figures the paper actually references from `code/figures/` to
`paper/sections/figures/`, so the paper always builds against the latest
script outputs.

Behavior:
- Scans every `paper/sections/*.tex` for `\\includegraphics{sections/figures/<file>}`
- Copies each referenced PDF (and matching PNG, if present) from
  `code/figures/<file>` to `paper/sections/figures/<file>`, overwriting.
- Warns if a referenced figure is missing from `code/figures/`.
- Does NOT touch unreferenced figures in either dir.

Call this at the end of any script that writes to `code/figures/`.
Idempotent — safe to invoke even when nothing changed.
"""

const FIG_DIR = abspath(joinpath(@__DIR__, "..", "figures"))
const PAPER_FIG_DIR = abspath(joinpath(@__DIR__, "..", "..", "paper", "sections", "figures"))
const PAPER_SECTIONS_DIR = abspath(joinpath(@__DIR__, "..", "..", "paper", "sections"))

function _referenced_pdfs()
    refs = Set{String}()
    pat = r"sections/figures/([A-Za-z0-9_.\-]+\.pdf)"
    isdir(PAPER_SECTIONS_DIR) || return String[]
    for (root, _, fs) in walkdir(PAPER_SECTIONS_DIR)
        for f in fs
            endswith(f, ".tex") || continue
            txt = read(joinpath(root, f), String)
            for m in eachmatch(pat, txt)
                push!(refs, m.captures[1])
            end
        end
    end
    return sort(collect(refs))
end

function promote_figures()
    mkpath(PAPER_FIG_DIR)
    refs = _referenced_pdfs()
    copied = String[]
    missing_pdfs = String[]
    for pdf in refs
        src = joinpath(FIG_DIR, pdf)
        if isfile(src)
            cp(src, joinpath(PAPER_FIG_DIR, pdf); force=true)
            push!(copied, pdf)
        else
            push!(missing_pdfs, pdf)
        end
    end
    println("\n[promote_figures] $(length(copied))/$(length(refs)) referenced PDFs synced -> paper/sections/figures/")
    if !isempty(missing_pdfs)
        println("[promote_figures] WARNING: referenced figures missing from code/figures/:")
        for f in missing_pdfs
            println("    - $f")
        end
    end
    return (copied=copied, missing=missing_pdfs)
end

# Allow direct invocation: `julia --project=. scripts/promote_figures.jl`
if abspath(PROGRAM_FILE) == @__FILE__
    promote_figures()
end
