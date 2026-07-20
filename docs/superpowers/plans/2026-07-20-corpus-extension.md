# Corpus Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the ladder corpus from 15 to 58 capture dates and run an expanding-window walk-forward across 53 folds, testing whether the sector model's edge holds over fourteen weeks instead of four.

**Architecture:** New captures land in `code/data/ladder_extended/`, a sibling of `code/data/ladder/` rather than a child. All 20 existing scripts resolve the literal path `data/ladder` with no wildcard, so a sibling directory is invisible to them and the 15-date arm stays independently runnable. `TemporalFolds.load_split` gains a multi-root method by Julia dispatch; the existing single-root method delegates to it, so the four current callers are unchanged.

**Tech Stack:** Julia 1.x, DataFrames, CSV, Flux, Plots, Dates, Test. Python 3 + yfinance for the earnings fetcher only.

## Global Constraints

- Never write to `code/data/ladder/`. It holds the 15-date arm and must keep reproducing 234,549 observations.
- Never call `promote_figures()`. Nothing may be written to `paper-arxiv/**` or `paper-jcf/**` under this plan.
- Do not modify `examples/walk_forward_temporal.jl`, `examples/calibrate_ladders*.jl`, or any existing example script.
- Capture directories in this project use a four-digit year (`options-05-12-2026`). The SDK emits two digits (`options-05-12-26`). `TemporalFolds.dir_to_date` at `code/src/TemporalFolds.jl:62` matches `r"options-(\d{2})-(\d{2})-(\d{4})"` and will fail on the SDK form.
- Frozen cutoff date is `2026-05-11`. Nothing on or before it is ever synced.
- SDK source path default: `/Users/jdv27/Desktop/julia_work/alpaca-markets-sdk/data`.
- All Julia commands run from `code/` as `julia --project=. <path>`.

---

### Task 1: Sync script for extended captures

**Files:**
- Create: `code/scripts/sync_ladder_extended.jl`
- Create: `code/test/test_sync_ladder_extended.jl`
- Modify: `code/test/runtests.jl`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `sdk_to_project_dirname(d::AbstractString) -> Union{String,Nothing}`, `sdk_dir_date(d::AbstractString) -> Union{Date,Nothing}`, `should_sync(d::AbstractString) -> Bool`, `sync_extended(; sdk_dir::AbstractString, dest_dir::AbstractString, dry_run::Bool=false) -> Vector{String}` returning the project-form directory names copied.

- [ ] **Step 1: Write the failing test**

Create `code/test/test_sync_ladder_extended.jl`:

```julia
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
        end
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd code && julia --project=. test/test_sync_ladder_extended.jl`
Expected: FAIL — `SystemError: opening file ".../scripts/sync_ladder_extended.jl"`, because the script does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `code/scripts/sync_ladder_extended.jl`:

```julia
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd code && julia --project=. test/test_sync_ladder_extended.jl`
Expected: PASS — `Test Summary: | Pass  Total` with 0 failures across all six testsets.

- [ ] **Step 5: Wire into the suite**

Modify `code/test/runtests.jl`. The existing file is:

```julia
using Test
using HestonIV

@testset "HestonIV" begin
    include("test_types.jl")
    include("test_theta_function.jl")
    include("test_heston_variance.jl")
    include("test_crr_tree.jl")
    include("test_calibration.jl")
end
```

Add the new file as its own testset, outside the `HestonIV` one, because the sync script is standalone and does not depend on the `HestonIV` package:

```julia
using Test
using HestonIV

@testset "HestonIV" begin
    include("test_types.jl")
    include("test_theta_function.jl")
    include("test_heston_variance.jl")
    include("test_crr_tree.jl")
    include("test_calibration.jl")
end

@testset "Corpus tooling" begin
    include("test_sync_ladder_extended.jl")
end
```

- [ ] **Step 6: Run the full suite**

Run: `cd code && julia --project=. test/runtests.jl`
Expected: PASS — existing HestonIV testsets unchanged, new "Corpus tooling" testset passing.

- [ ] **Step 7: Commit**

```bash
git add code/scripts/sync_ladder_extended.jl code/test/test_sync_ladder_extended.jl code/test/runtests.jl
git commit -m "Add sync_ladder_extended.jl: SDK captures -> ladder_extended/

Renames two-digit SDK year to four-digit project form. Cutoff at 2026-05-11
keeps the 15-date arm frozen and excludes the partial 04-20 capture. Refuses
to write into data/ladder."
```

---

### Task 2: Multi-root loading and the 58-date list

**Files:**
- Modify: `code/src/TemporalFolds.jl:96-108` (replace `load_split`), plus the export line at `code/src/TemporalFolds.jl:27` and the constants block near `:52`
- Create: `code/test/test_temporal_folds_roots.jl`
- Modify: `code/test/runtests.jl`

**Interfaces:**
- Consumes: `SECTORS`, `dir_to_date`, `load_ladder` — all existing in `TemporalFolds`.
- Produces: `resolve_day(roots::Vector{<:AbstractString}, day::AbstractString) -> String` (absolute path to the day directory), `load_split(roots::Vector{<:AbstractString}, day_dirs::Vector{<:AbstractString}) -> DataFrame`, and `EXTENDED_DAYS::Vector{String}` of length 58.

- [ ] **Step 1: Write the failing test**

Create `code/test/test_temporal_folds_roots.jl`:

```julia
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd code && julia --project=. test/test_temporal_folds_roots.jl`
Expected: FAIL — `UndefVarError: EXTENDED_DAYS not defined` on the first testset.

- [ ] **Step 3: Add `EXTENDED_DAYS`**

In `code/src/TemporalFolds.jl`, immediately after the `EARNINGS_WINDOW` constant (around line 55, before the `# Data loading` banner), insert:

```julia
# ============================================================================
# Extended corpus — 58 capture dates, 2026-04-14 .. 2026-07-17
#
# The first 15 entries are the frozen arm living in data/ladder; entries 16..58
# live in data/ladder_extended. Consumers resolve each name across both roots
# via `resolve_day`, so this list is layout-agnostic.
#
# Gaps are real: 04-30, 05-04, 05-05, 05-07, 06-05, 06-08 and 06-09 were not
# captured. 05-25 (Memorial Day) and 07-03 (Independence Day observed) are
# market holidays.
# ============================================================================

const EXTENDED_DAYS = [
    # --- frozen 15-date arm (data/ladder) -----------------------------------
    "options-04-14-2026", "options-04-15-2026", "options-04-16-2026",
    "options-04-17-2026", "options-04-21-2026", "options-04-22-2026",
    "options-04-23-2026", "options-04-24-2026", "options-04-27-2026",
    "options-04-28-2026", "options-04-29-2026", "options-05-01-2026",
    "options-05-06-2026", "options-05-08-2026", "options-05-11-2026",
    # --- extension (data/ladder_extended) -----------------------------------
    "options-05-12-2026", "options-05-13-2026", "options-05-14-2026",
    "options-05-15-2026", "options-05-18-2026", "options-05-19-2026",
    "options-05-20-2026", "options-05-21-2026", "options-05-22-2026",
    "options-05-26-2026", "options-05-27-2026", "options-05-28-2026",
    "options-05-29-2026", "options-06-01-2026", "options-06-02-2026",
    "options-06-03-2026", "options-06-04-2026", "options-06-10-2026",
    "options-06-11-2026", "options-06-12-2026", "options-06-15-2026",
    "options-06-16-2026", "options-06-17-2026", "options-06-18-2026",
    "options-06-22-2026", "options-06-23-2026", "options-06-24-2026",
    "options-06-25-2026", "options-06-26-2026", "options-06-29-2026",
    "options-06-30-2026", "options-07-01-2026", "options-07-02-2026",
    "options-07-06-2026", "options-07-07-2026", "options-07-08-2026",
    "options-07-09-2026", "options-07-10-2026", "options-07-13-2026",
    "options-07-14-2026", "options-07-15-2026", "options-07-16-2026",
    "options-07-17-2026",
]
@assert length(EXTENDED_DAYS) == 58
```

- [ ] **Step 4: Replace `load_split` with the multi-root version**

In `code/src/TemporalFolds.jl`, replace the whole existing function at lines 96-108 — the one beginning `function load_split(ladder_dir::AbstractString, day_dirs::Vector{<:AbstractString})` and ending at its `end` — with:

```julia
"""
    resolve_day(roots, day) → String

Locate day-directory `day` across `roots` and return its full path. Raises if
it is absent from every root, and raises if more than one root supplies it —
both are silent corpus-corruption modes, so neither is a warn-and-continue.
"""
function resolve_day(roots::Vector{<:AbstractString}, day::AbstractString)
    hits = [joinpath(r, day) for r in roots if isdir(joinpath(r, day))]
    if isempty(hits)
        error("day directory '$(day)' not found in any root: " *
              join(roots, ", "))
    elseif length(hits) > 1
        error("day directory '$(day)' found in multiple roots: " *
              join(hits, ", "))
    end
    return hits[1]
end

function load_split(roots::Vector{<:AbstractString},
                    day_dirs::Vector{<:AbstractString})
    frames = DataFrame[]
    for d in day_dirs
        full = resolve_day(roots, d)
        day_date = dir_to_date(d)
        for f in readdir(full)
            endswith(f, ".csv") && occursin("_dte_ladder_", f) || continue
            df = load_ladder(joinpath(full, f), day_date)
            nrow(df) > 0 && push!(frames, df)
        end
    end
    out = vcat(frames...)
    out[!, :sector] = [get(SECTORS, t, "Other") for t in out.ticker]
    return out
end

# Single-root callers delegate, keeping one implementation of the load body.
load_split(ladder_dir::AbstractString, day_dirs::Vector{<:AbstractString}) =
    load_split([String(ladder_dir)], day_dirs)
```

Leave the docstring block directly above the original function (lines 88-94) in place; it documents the single-root form and stays accurate.

- [ ] **Step 5: Export the new names**

At `code/src/TemporalFolds.jl:27`, the export line currently reads:

```julia
export load_earnings_calendar, load_split, attach_earnings_features!,
       near_earnings_mask, compute_standardizer
```

Change it to:

```julia
export load_earnings_calendar, load_split, resolve_day, EXTENDED_DAYS,
       attach_earnings_features!, near_earnings_mask, compute_standardizer
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd code && julia --project=. test/test_temporal_folds_roots.jl`
Expected: PASS — both testsets green.

- [ ] **Step 7: Verify existing callers still work**

The four existing callers pass a `String` root and must bind to the delegating method. Confirm dispatch resolves without loading data:

Run:
```bash
cd code && julia --project=. -e '
include("src/TemporalFolds.jl"); using .TemporalFolds
@assert hasmethod(load_split, Tuple{String, Vector{String}})
@assert hasmethod(load_split, Tuple{Vector{String}, Vector{String}})
println("both load_split methods present")'
```
Expected: `both load_split methods present`

- [ ] **Step 8: Wire into the suite and commit**

In `code/test/runtests.jl`, add to the "Corpus tooling" testset created in Task 1:

```julia
@testset "Corpus tooling" begin
    include("test_sync_ladder_extended.jl")
    include("test_temporal_folds_roots.jl")
end
```

Run: `cd code && julia --project=. test/runtests.jl`
Expected: PASS, all testsets.

```bash
git add code/src/TemporalFolds.jl code/test/test_temporal_folds_roots.jl code/test/runtests.jl
git commit -m "TemporalFolds: multi-root load_split + 58-date EXTENDED_DAYS

resolve_day maps each day dir across roots, raising on zero or multiple
matches. Single-root load_split delegates to the multi-root method so the
four existing callers are unchanged."
```

---

### Task 3: Extended walk-forward script

**Files:**
- Create: `code/examples/walk_forward_extended.jl`

**Interfaces:**
- Consumes: `EXTENDED_DAYS`, `load_split(roots, days)`, `load_earnings_calendar`, `attach_earnings_features!`, `run_fold(train_df, test_df, n_inputs; sectors_list, tickers_list, label, verbose, seed)` returning a value with `.train_rmse`, `.test_rmse`, `.gen_gap`.
- Produces: `code/figures/walk_forward_extended_summary.csv`, `walk_forward_extended_gap.pdf`, `walk_forward_extended_gap.png`.

This task creates the script but does not run the full sweep; Tasks 4-8 handle execution. There is no unit test here — the script is a driver whose correctness is established by the gated runs, and inventing a synthetic-corpus test would exercise machinery already covered by Task 2.

- [ ] **Step 1: Write the script**

Create `code/examples/walk_forward_extended.jl`:

```julia
"""
Expanding-window walk-forward across the 58-date extended ladder corpus.

For k = 5..57, train on the first k capture dates and test on the unseen date
k+1, giving 53 folds. Two configurations run per fold: the 2-input sector model
(matching `walk_forward_temporal.jl`) and the 4-input earnings-aware model.

Reads two roots — `data/ladder` for the 15-date arm and `data/ladder_extended`
for the 43 new dates. `walk_forward_temporal.jl` is untouched and still
reproduces the 15-date result independently.

Environment:
    WF_K_MAX   cap on k, for smoke runs (default 57, the full sweep)

Run:
    julia --project=. examples/walk_forward_extended.jl
    WF_K_MAX=16 julia --project=. examples/walk_forward_extended.jl
"""

using CSV
using DataFrames
using Dates
using Plots
using Plots.PlotMeasures
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "src", "TemporalFolds.jl"))
using .TemporalFolds

const LADDER_ROOTS = [
    joinpath(@__DIR__, "..", "data", "ladder"),
    joinpath(@__DIR__, "..", "data", "ladder_extended"),
]
const EARNINGS_CSV = joinpath(@__DIR__, "..", "data", "earnings",
                              "earnings_calendar.csv")
const FIG_DIR = joinpath(@__DIR__, "..", "figures")

const CONFIGS = [
    (name = "sector_2in",   n_inputs = 2),
    (name = "earnings_4in", n_inputs = 4),
]

const K_MAX = parse(Int, get(ENV, "WF_K_MAX", "57"))
const K_RANGE = 5:K_MAX
@assert K_MAX <= length(EXTENDED_DAYS) - 1

println("Loading earnings calendar from $(EARNINGS_CSV) ...")
cal = load_earnings_calendar(EARNINGS_CSV)
println("  loaded $(length(cal)) tickers")

println("Loading extended corpus across $(length(LADDER_ROOTS)) roots ...")
all_data = load_split(LADDER_ROOTS, EXTENDED_DAYS)
attach_earnings_features!(all_data, cal)
@printf("  %d obs, %d tickers, %d dates\n", nrow(all_data),
        length(unique(all_data.ticker)), length(unique(all_data.obs_date)))

universe_sectors = sort(unique(all_data.sector))
universe_tickers = sort(unique(all_data.ticker))

function _split_by_dirnames(df, dirs)
    target_dates = Set(TemporalFolds.dir_to_date(d) for d in dirs)
    return df[[d in target_dates for d in df.obs_date], :]
end

function _test_window_has_earnings(test_df)
    return any(abs.(test_df.d2e_self) .<= 3) || any(test_df.d2e_peer_min .<= 3)
end

rows = NamedTuple[]
for k in K_RANGE
    train_df = _split_by_dirnames(all_data, EXTENDED_DAYS[1:k])
    test_df  = _split_by_dirnames(all_data, EXTENDED_DAYS[k+1:k+1])
    test_date = TemporalFolds.dir_to_date(EXTENDED_DAYS[k+1])

    if nrow(test_df) < 500
        @printf("  SKIP k=%2d (%s): only %d test obs\n",
                k, test_date, nrow(test_df))
        continue
    end

    has_e = _test_window_has_earnings(test_df)

    for cfg in CONFIGS
        fold = run_fold(train_df, test_df, cfg.n_inputs;
                        sectors_list = universe_sectors,
                        tickers_list = universe_tickers,
                        label = "WF-ext k=$(k) test=$(test_date) [$(cfg.name)]",
                        verbose = false, seed = 42)

        push!(rows, (
            k = k,
            config = cfg.name,
            n_inputs = cfg.n_inputs,
            n_train_days = k,
            n_train_obs = nrow(train_df),
            test_date = test_date,
            n_test_obs = nrow(test_df),
            train_rmse = fold.train_rmse,
            test_rmse = fold.test_rmse,
            gen_gap = fold.gen_gap,
            has_earnings_in_test_window = has_e,
        ))

        @printf("  k=%2d %-13s test=%s train=%5.2f%% test=%5.2f%% gap=%+5.2f%% e=%s\n",
                k, cfg.name, test_date, fold.train_rmse*100,
                fold.test_rmse*100, fold.gen_gap*100, has_e ? "Y" : "N")
    end
end

# ============================================================================
# Write CSV
# ============================================================================
mkpath(FIG_DIR)
df = DataFrame(rows)
csv_path = joinpath(FIG_DIR, "walk_forward_extended_summary.csv")
CSV.write(csv_path, df)
@printf("\n[csv] wrote -> %s\n", csv_path)

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^80)
println("  EXTENDED WALK-FORWARD SUMMARY  ($(length(K_RANGE)) folds)")
println("="^80)
for cfg in CONFIGS
    sub = df[df.config .== cfg.name, :]
    nrow(sub) == 0 && continue
    @printf("  %-13s  median test %5.2f%%   median gap %+5.2f%%   mean gap %+5.2f%%\n",
            cfg.name, median(sub.test_rmse)*100,
            median(sub.gen_gap)*100, mean(sub.gen_gap)*100)
end

# ============================================================================
# Figure
# ============================================================================
plt = plot(size=(900, 460), left_margin=6mm, bottom_margin=6mm,
           xlabel="test date", ylabel="test RMSE (%)",
           title="Expanding-window walk-forward, 58-date corpus",
           legend=:topright)
for cfg in CONFIGS
    sub = df[df.config .== cfg.name, :]
    nrow(sub) == 0 && continue
    plot!(plt, sub.test_date, sub.test_rmse .* 100,
          label=cfg.name, marker=:circle, markersize=3, linewidth=1.5)
end
for ext in ("pdf", "png")
    p = joinpath(FIG_DIR, "walk_forward_extended_gap.$(ext)")
    savefig(plt, p)
    @printf("[fig] wrote -> %s\n", p)
end

println("\nDone.")
```

- [ ] **Step 2: Verify it parses without running**

Run: `cd code && julia --project=. -e 'include("examples/walk_forward_extended.jl")' 2>&1 | head -5`
Expected: it begins executing and fails at the data-load step with a `resolve_day` error naming `options-05-12-2026`, because `ladder_extended/` does not exist yet. That specific error confirms the file parses and the resolver is wired correctly. Any `ParseError` or `UndefVarError` instead means fix the script before continuing.

- [ ] **Step 3: Audit for the pooling-bug pattern**

Three figure and scenario scripts previously broke on multi-date corpora by taking `:S[end]` or filtering to a single date. Confirm the new script does neither:

Run: `cd code && grep -n 'S\[end\]\|\.S\[1\]\|obs_date\s*==\|first(.*obs_date' examples/walk_forward_extended.jl`
Expected: no output. `_split_by_dirnames` uses set membership over `obs_date`, which is the correct multi-date form.

- [ ] **Step 4: Commit**

```bash
git add code/examples/walk_forward_extended.jl
git commit -m "Add walk_forward_extended.jl: 53-fold expanding window, 2 configs

Reads data/ladder and data/ladder_extended via multi-root load_split. Runs
sector 2-input and earnings-aware 4-input per fold. WF_K_MAX caps k for
smoke runs. walk_forward_temporal.jl is untouched."
```

---

### Task 4: Sync the data and verify it landed

**Files:**
- Create: `code/data/ladder_extended/` (43 directories, data only)

**Interfaces:**
- Consumes: `sync_extended` from Task 1.
- Produces: the populated `ladder_extended/` root that Tasks 7 and 8 read.

- [ ] **Step 1: Dry-run the sync**

Run:
```bash
cd code && julia --project=. -e '
include("scripts/sync_ladder_extended.jl"); using .SyncLadderExtended
c = sync_extended(sdk_dir="/Users/jdv27/Desktop/julia_work/alpaca-markets-sdk/data",
                  dest_dir=joinpath("data","ladder_extended"), dry_run=true)
println(length(c), " dirs would sync"); println(first(c)); println(last(c))'
```
Expected:
```
43 dirs would sync
options-05-12-2026
options-07-17-2026
```
If the count is not 43, stop and reconcile before copying anything.

- [ ] **Step 2: Run the sync**

Run: `cd code && julia --project=. scripts/sync_ladder_extended.jl`
Expected: `synced 43 capture directories into .../data/ladder_extended` followed by 43 names.

- [ ] **Step 3: Verify counts and the frozen root**

Run:
```bash
cd code/data && echo "extended dirs: $(ls -d ladder_extended/options-* | wc -l)" &&
echo "frozen dirs:   $(ls -d ladder/options-* | wc -l)" &&
echo "dirs not holding 31 CSVs:" &&
for d in ladder_extended/options-*; do
  n=$(ls "$d"/*_dte_ladder_*.csv 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -ne 31 ] && echo "  $d has $n"
done; echo "(none above means all 31)"
```
Expected:
```
extended dirs: 43
frozen dirs:   15
dirs not holding 31 CSVs:
(none above means all 31)
```
The frozen count of 15 is the guard: if it reads anything else, the sync wrote to the wrong root — revert immediately with `git checkout code/data/ladder`.

- [ ] **Step 4: Verify idempotency**

Run: `cd code && julia --project=. scripts/sync_ladder_extended.jl`
Expected: `synced 0 capture directories into .../data/ladder_extended`

- [ ] **Step 5: Commit the data**

This adds roughly 275 MB. Confirm the size first, then commit.

```bash
cd code/data && du -sh ladder_extended
git add code/data/ladder_extended
git commit -m "Add 43 extended capture dates, 2026-05-12 .. 2026-07-17

31 tickers on every date. Sibling of data/ladder so the 15-date arm and every
script resolving the literal data/ladder path are unaffected."
```

---

### Task 5: Refresh the earnings calendar

**Files:**
- Modify: `code/data/earnings/earnings_calendar.csv`

**Interfaces:**
- Consumes: the existing `code/scripts/fetch_earnings_calendar.py`.
- Produces: a calendar covering 2026-05-12 through 2026-07-17, read by the 4-input configuration in Task 3's script.

The current file was fetched 2026-04-26 and holds only 10 events in the new window against 24-27 in comparable nine-week windows, every one with an empty `eps_actual`. The 4-input features are proximity-based (`TemporalFolds.jl:117` and `:131` call `days_to_earnings`), so missing scheduled dates would silently make earnings-adjacent days look quiet.

- [ ] **Step 1: Record the pre-refresh baseline**

Run:
```bash
cd code/data/earnings && echo "total rows: $(($(wc -l < earnings_calendar.csv) - 1))" &&
echo "new-window events: $(awk -F, 'NR>1 && $2>="2026-05-12" && $2<="2026-07-17"' earnings_calendar.csv | wc -l)"
```
Expected: `new-window events: 10`

- [ ] **Step 2: Refresh**

Run: `cd /Users/jdv27/Desktop/julia_work/heston_implied_volatility_model && code/venv/bin/python code/scripts/fetch_earnings_calendar.py`
Expected: the script reports tickers fetched and writes `code/data/earnings/earnings_calendar.csv`.

If `code/venv/` is absent, create it per `SETUP.md` before continuing.

- [ ] **Step 3: Verify the window filled in**

Run:
```bash
cd code/data/earnings && echo "new-window events: $(awk -F, 'NR>1 && $2>="2026-05-12" && $2<="2026-07-17"' earnings_calendar.csv | wc -l)" &&
echo "of those, with actuals: $(awk -F, 'NR>1 && $2>="2026-05-12" && $2<="2026-07-17" && $4!=""' earnings_calendar.csv | wc -l)"
```
Expected: roughly 25 events, most carrying actuals.

This number is an extrapolation from historical window density, not a verified count. If it comes back near 10-12 rather than ~25, stop and report: the 4-input configuration's premise is that the new window carries an independent set of earnings events, and a sparse result undercuts running it at all.

- [ ] **Step 4: Diff the historical rows**

A refresh can revise dates the 15-date arm depends on. Surface any such change rather than absorbing it.

Run:
```bash
cd /Users/jdv27/Desktop/julia_work/heston_implied_volatility_model &&
git show HEAD:code/data/earnings/earnings_calendar.csv > /tmp/cal_old.csv &&
awk -F, 'NR>1 && $2<="2026-05-11"' /tmp/cal_old.csv | sort > /tmp/old_hist.txt &&
awk -F, 'NR>1 && $2<="2026-05-11"' code/data/earnings/earnings_calendar.csv | sort > /tmp/new_hist.txt &&
diff /tmp/old_hist.txt /tmp/new_hist.txt && echo "IDENTICAL: no historical rows changed"
```
Expected: `IDENTICAL: no historical rows changed`. If rows differ, report exactly which tickers and dates moved before proceeding — that affects the 15-date arm's earnings features.

- [ ] **Step 5: Commit**

```bash
git add code/data/earnings/earnings_calendar.csv
git commit -m "Refresh earnings calendar through 2026-07-17

Previous file was a 2026-04-26 snapshot holding 10 events in the 05-12..07-17
window against 24-27 in comparable windows; yfinance returns only ~4 upcoming
dates per ticker beyond its history window."
```

---

### Task 6: Control-arm check

**Files:** none modified. This task runs an existing script and confirms an unchanged result.

**Interfaces:**
- Consumes: the populated `ladder_extended/` from Task 4.
- Produces: empirical confirmation that the two-root split insulates the 15-date arm.

Static inspection already showed all 20 scripts resolve the literal path `data/ladder` with no wildcard. This step confirms it by execution rather than by reading.

- [ ] **Step 1: Re-run a pooled-fit script**

Run: `cd code && julia --project=. examples/calibrate_ladders_sector_nn.jl 2>&1 | tee /tmp/control_arm.log | tail -30`

- [ ] **Step 2: Confirm the corpus size is unchanged**

Run: `grep -E '[0-9]{6} (total )?obs|observations' /tmp/control_arm.log | head -3`
Expected: `234549` appears. If it reports roughly 900,000, the extended data is being picked up by the walkdir and the two-root separation has failed — stop and investigate before running anything else.

- [ ] **Step 3: Confirm the headline RMSE is unchanged**

Run: `grep -iE 'rmse|10\.2' /tmp/control_arm.log | tail -10`
Expected: the sector-NN RMSE still reads 10.24%. Small last-digit drift from nondeterministic training is acceptable; a move of more than about 0.1 percentage point is not, and means the corpus changed.

- [ ] **Step 4: Confirm the frozen root is still 15 directories**

Run: `ls -d code/data/ladder/options-* | wc -l`
Expected: `15`

No commit — this task produces no file changes.

---

### Task 7: Smoke test across the root boundary

**Files:** none modified. Produces throwaway output overwritten by Task 8.

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: a measured wall-clock figure that sizes Task 8.

`K_MAX=16` is chosen deliberately. Fold k=15 trains on days 1-15, all in `ladder/`, and tests on day 16, `options-05-12-2026`, in `ladder_extended/`. That is the one fold exercising the cross-root resolver on a real boundary, and it must run before the full sweep is worth starting.

- [ ] **Step 1: Run the smoke test with timing**

Run: `cd code && time WF_K_MAX=16 julia --project=. examples/walk_forward_extended.jl 2>&1 | tee /tmp/wf_smoke.log | tail -40`

- [ ] **Step 2: Confirm the full corpus loaded**

Run: `grep -E 'obs, [0-9]+ tickers' /tmp/wf_smoke.log`
Expected: roughly 900,000 obs across 31 tickers and 58 dates. If it reports 234,549 the extended root was not read; if a `resolve_day` error appears, a day directory is missing or duplicated and the message names which.

- [ ] **Step 3: Confirm the seam fold ran under both configurations**

Run: `grep 'k=15' /tmp/wf_smoke.log`
Expected: two lines, one `sector_2in` and one `earnings_4in`, both with `test=2026-05-12` and finite RMSE values.

- [ ] **Step 4: Confirm the earnings configuration is distinguishable**

Run: `grep -c 'earnings_4in' /tmp/wf_smoke.log`
Expected: 12 — one per fold for k=5..16. If the 4-input rows show RMSE identical to their 2-input counterparts to every digit, the earnings features are not reaching the model; investigate before the full sweep.

- [ ] **Step 5: Extrapolate the full-sweep cost and decide**

The smoke run covers 12 of 53 folds, and its folds are the cheapest — training sets grow monotonically, so the back half of the sweep costs far more per fold. Multiply the measured wall-clock by roughly 8-10 for an order-of-magnitude estimate of Task 8.

Report the estimate before starting Task 8. If it exceeds a few hours, raise it rather than launching: dropping to one configuration, or striding k, are both cheaper answers than a multi-day run.

No commit — output files are overwritten by Task 8.

---

### Task 8: Full sweep

**Files:**
- Create: `code/figures/walk_forward_extended_summary.csv`, `walk_forward_extended_gap.pdf`, `walk_forward_extended_gap.png`
- Create: `code/logs/walk_forward_extended.log`

**Interfaces:**
- Consumes: everything above.
- Produces: the drift curve this plan exists to generate.

- [ ] **Step 1: Run the sweep**

Run: `cd code && mkdir -p logs && time julia --project=. examples/walk_forward_extended.jl 2>&1 | tee logs/walk_forward_extended.log | tail -50`

- [ ] **Step 2: Confirm fold coverage**

Run:
```bash
cd code && julia --project=. -e '
using CSV, DataFrames
df = CSV.read("figures/walk_forward_extended_summary.csv", DataFrame)
println("rows: ", nrow(df))
println("folds: ", length(unique(df.k)))
println("configs: ", unique(df.config))
println("any non-finite: ", any(!isfinite, df.test_rmse))'
```
Expected: 106 rows, 53 folds, both config names, and `any non-finite: false`. Fewer than 53 folds means some were skipped for low test-observation counts — the run log names which and why.

- [ ] **Step 3: Read out the headline comparison**

Run: `grep -A 4 'EXTENDED WALK-FORWARD SUMMARY' code/logs/walk_forward_extended.log`
Expected: median test RMSE and median gap per configuration.

The reference points are the 15-date walk-forward's median test RMSE of 11.08% and median gap of +2.02%. Whether the extended sweep lands above or below those is the result; do not treat either direction as the expected answer.

- [ ] **Step 4: Confirm no paper directory was touched**

Run: `cd /Users/jdv27/Desktop/julia_work/heston_implied_volatility_model && git status --porcelain paper-arxiv paper-jcf`
Expected: no output.

- [ ] **Step 5: Commit results**

```bash
git add code/figures/walk_forward_extended_summary.csv \
        code/figures/walk_forward_extended_gap.pdf \
        code/figures/walk_forward_extended_gap.png \
        code/logs/walk_forward_extended.log
git commit -m "Extended walk-forward results: 53 folds, 58-date corpus

Sector 2-input and earnings-aware 4-input across an expanding window from
2026-04-14 to 2026-07-17. Figures are not promoted to either paper directory."
```

- [ ] **Step 6: Report**

Summarize for review: median test RMSE and gap per configuration against the 15-date baseline of 11.08% / +2.02%, whether the gap trends over the fourteen weeks, whether the 4-input configuration separates from the 2-input one, and which folds if any were skipped. Whether any of this belongs in a paper is a separate decision.

---

## Notes for the implementer

**What this plan deliberately does not do:** per-ticker and LoRA configurations across the folds, SABR and Horvath baselines, consolidating the 15-date list and `SECTORS` dictionary duplicated across five files, and any paper edit. The duplication cleanup is worth doing — `code/src/TemporalFolds.jl` was an attempt at centralizing that the `lora_*`, `calibrate_*`, and `price_error_figure.jl` scripts never adopted — but it belongs in its own task where a perturbed number is the only thing under review.

**The failure mode to watch.** Tasks 4 and 6 both check that `code/data/ladder/` still holds exactly 15 directories. That is not redundant. The entire design rests on the extended captures being invisible to the walkdir-based scripts, and a single stray write into that root converts every subsequent number in the project into a silent comparison against a different corpus.
