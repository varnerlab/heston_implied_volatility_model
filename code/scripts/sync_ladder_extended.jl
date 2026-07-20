"""
    SyncLadderExtended

Copy post-cutoff option-ladder captures from the alpaca-markets-sdk sibling
repository into `code/data/ladder_extended/`, renaming the SDK's two-digit
year to the four-digit form this project uses.

`code/data/ladder/` holds the 15-date arm and is never written to. The cutoff
guarantees that: nothing dated on or before 2026-05-11 is copied, which also
excludes the SDK's partial 04-20 capture (23 tickers), already held out in
`code/data/ladder_excluded/`.

Run:
    julia --project=. scripts/sync_ladder_extended.jl
    julia --project=. scripts/sync_ladder_extended.jl /path/to/sdk/data
"""
module SyncLadderExtended

using Dates

export sdk_to_project_dirname, sdk_dir_date, should_sync, sync_extended

const SDK_DIR_RE = r"^options-(\d{2})-(\d{2})-(\d{2})$"
const FROZEN_CUTOFF = Date(2026, 5, 11)
const DEFAULT_SDK_DIR =
    "/Users/jdv27/Desktop/julia_work/alpaca-markets-sdk/data"

"Convert an SDK capture dirname to this project's four-digit-year form, or nothing."
function sdk_to_project_dirname(d::AbstractString)
    m = match(SDK_DIR_RE, String(d))
    m === nothing && return nothing
    mm, dd, yy = m.captures
    return "options-$(mm)-$(dd)-20$(yy)"
end

"Parse an SDK capture dirname to a Date, or nothing if it is not a capture dir."
function sdk_dir_date(d::AbstractString)
    m = match(SDK_DIR_RE, String(d))
    m === nothing && return nothing
    mm, dd, yy = m.captures
    return Date(2000 + parse(Int, yy), parse(Int, mm), parse(Int, dd))
end

"True when a directory is a capture strictly after the frozen cutoff."
function should_sync(d::AbstractString)
    dt = sdk_dir_date(d)
    return dt !== nothing && dt > FROZEN_CUTOFF
end

"""
    sync_extended(; sdk_dir, dest_dir, dry_run=false) -> Vector{String}

Copy every post-cutoff capture directory from `sdk_dir` into `dest_dir`,
renamed to four-digit-year form. Skips directories already present, so it is
idempotent. Returns the project-form names actually copied, sorted.
"""
function sync_extended(; sdk_dir::AbstractString = DEFAULT_SDK_DIR,
                         dest_dir::AbstractString,
                         dry_run::Bool = false)
    if occursin(Regex("(^|/)ladder\$"), rstrip(String(dest_dir), '/'))
        error("refusing to write into the frozen 15-date root: $(dest_dir)")
    end
    isdir(sdk_dir) || error("SDK data directory not found: $(sdk_dir)")

    copied = String[]
    for d in sort(readdir(sdk_dir))
        should_sync(d) || continue
        target_name = sdk_to_project_dirname(d)
        target = joinpath(dest_dir, target_name)
        isdir(target) && continue
        if !dry_run
            mkpath(dest_dir)
            cp(joinpath(sdk_dir, d), target)
        end
        push!(copied, target_name)
    end
    return copied
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    using .SyncLadderExtended
    sdk = length(ARGS) >= 1 ? ARGS[1] : SyncLadderExtended.DEFAULT_SDK_DIR
    dest = joinpath(@__DIR__, "..", "data", "ladder_extended")
    copied = sync_extended(sdk_dir=sdk, dest_dir=dest)
    println("synced $(length(copied)) capture directories into $(dest)")
    for c in copied
        println("  $(c)")
    end
end
