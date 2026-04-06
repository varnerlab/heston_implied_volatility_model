"""
Fit the ψ Term Structure from Cross-Sectional IV Data

Loads NVDA option chains across 7 DTEs (0, 2, 4, 7, 14, 25, 46 days),
all captured on the same date (2026-04-06), and fits the ψ surface:

    σ_model(K, T) = σ_base · ψ(β, DTE, K/S)

where ψ = exp(β₁·ln(DTE) + β₂·ln(K/S) + β₃·ln(DTE)·ln(K/S) + β₄·(ln(K/S))² + β₅·(ln(DTE))²)

The quadratic DTE term (β₅) captures the U-shaped ATM term structure:
short-dated IV is elevated (gamma compression), dips at intermediate DTEs,
and rises again at longer maturities (uncertainty premium).

This isolates the term structure geometry (β₁–β₅) from regime/mood effects.
The fitted β values plug directly into the full θ(t) for synthetic generation.
"""

using CSV
using TOML
using DataFrames
using Statistics
using Optim
using Plots
using Printf

# ============================================================================
# Step 1: Load all DTE slices
# ============================================================================

const DATA_DIR = joinpath(@__DIR__, "..", "..", "data", "time")
const PLOT_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(PLOT_DIR)

const DTE_FILES = [
    "nvda_0dte",
    "nvda_2dte",
    "nvda_4dte",
    "nvda_7dte",
    "nvda_14dte",
    "nvda_25dte",
    "nvda_46dte",
]

"""Load all DTE slices into a single DataFrame with DTE and spot price columns."""
function load_all_slices()
    frames = DataFrame[]
    for f in DTE_FILES
        csv_path = joinpath(DATA_DIR, "$(f).csv")
        toml_path = joinpath(DATA_DIR, "$(f).toml")

        df = CSV.read(csv_path, DataFrame)
        meta = TOML.parsefile(toml_path)["metadata"]

        S = parse(Float64, meta["underlying_share_price"])
        dte = parse(Int, meta["DTE"])

        df[!, :DTE] .= dte
        df[!, :S] .= S
        df[!, :Moneyness] = df.Strike ./ S

        push!(frames, df)
    end
    return vcat(frames...)
end

println("Loading cross-sectional IV data...")
all_data = load_all_slices()
println("  Total observations: $(nrow(all_data))")
println("  DTEs: $(sort(unique(all_data.DTE)))")
println("  Strikes per DTE:")
for dte in sort(unique(all_data.DTE))
    slice = all_data[all_data.DTE .== dte, :]
    n_calls = sum(slice.Type .== "Call")
    n_puts = sum(slice.Type .== "Put")
    println("    DTE=$dte: $n_calls calls, $n_puts puts (S=\$$(slice.S[1]))")
end

# ============================================================================
# Step 2: Filter to reliable observations
# ============================================================================

# Use calls primarily (more liquid near ATM for most DTEs)
# Filter: moneyness in [0.90, 1.10], IV > 0, Volume > 0
calls = all_data[(all_data.Type .== "Call") .&
                 (all_data.Moneyness .>= 0.90) .&
                 (all_data.Moneyness .<= 1.10) .&
                 (all_data.IV .> 0.0) .&
                 (all_data.Volume .> 0), :]

puts = all_data[(all_data.Type .== "Put") .&
                (all_data.Moneyness .>= 0.90) .&
                (all_data.Moneyness .<= 1.10) .&
                (all_data.IV .> 0.0) .&
                (all_data.Volume .> 0), :]

# Combine calls and puts for fitting
fit_data = vcat(calls, puts)
println("\nFiltered for fitting: $(nrow(fit_data)) observations ($(nrow(calls)) calls, $(nrow(puts)) puts)")

# ============================================================================
# Step 3: Fit ψ surface
# ============================================================================

println("\n" * "="^60)
println("  FITTING ψ(β, DTE, K/S)")
println("="^60)

"""
    eval_psi(β, log_dte, log_m) → Float64

Evaluate ψ = exp(β₁·ln(DTE) + β₂·ln(K/S) + β₃·ln(DTE)·ln(K/S) + β₄·(ln(K/S))² + β₅·(ln(DTE))²)
"""
function eval_psi(β::Vector{Float64}, log_dte::Float64, log_m::Float64)::Float64
    return exp(β[1] * log_dte + β[2] * log_m + β[3] * log_dte * log_m +
               β[4] * log_m^2 + β[5] * log_dte^2)
end

"""
    fit_psi(df) → (σ_base, β, rmse, model_ivs)

Fit σ_model = σ_base · ψ(β, DTE, K/S) to observed IVs.

Parameters: σ_base (ATM base vol), β₁ (DTE decay), β₂ (skew),
            β₃ (DTE×skew interaction), β₄ (smile curvature),
            β₅ (DTE curvature — captures U-shaped term structure)
"""
function fit_psi(df::DataFrame)
    n = nrow(df)
    ivs = df.IV
    dtes = Float64.(df.DTE)
    moneyness = df.Moneyness

    # Parameter vector: [log(σ_base²), β₁, β₂, β₃, β₄, β₅]
    # Initialize σ_base from the mid-DTE ATM options (the trough region)
    mid_dte_mask = (dtes .>= 4) .& (dtes .<= 14) .& (abs.(moneyness .- 1.0) .< 0.05)
    σ_mid = any(mid_dte_mask) ? mean(ivs[mid_dte_mask]) : mean(ivs)
    x0 = [log(σ_mid^2), 0.3, -1.0, 0.1, 2.0, -0.15]

    function objective(x)
        θ_base = exp(x[1])  # σ_base² stored in log space
        β = x[2:6]

        err = 0.0
        for i in 1:n
            log_dte = log(max(dtes[i], 1.0))
            log_m = log(moneyness[i])
            ψ = eval_psi(β, log_dte, log_m)
            σ_model = sqrt(max(θ_base * ψ, 1e-10))
            err += (σ_model - ivs[i])^2
        end
        return err / n
    end

    # Two-pass optimization: Nelder-Mead for global search, then polish
    result = optimize(objective, x0, NelderMead(),
                      Optim.Options(iterations=100000, g_tol=1e-14))
    result = optimize(objective, Optim.minimizer(result), NelderMead(),
                      Optim.Options(iterations=100000, g_tol=1e-14))

    x_opt = Optim.minimizer(result)
    θ_base = exp(x_opt[1])
    σ_base = sqrt(θ_base)
    β = x_opt[2:6]

    # Compute model IVs
    model_ivs = Vector{Float64}(undef, n)
    for i in 1:n
        log_dte = log(max(dtes[i], 1.0))
        log_m = log(moneyness[i])
        model_ivs[i] = sqrt(max(θ_base * eval_psi(β, log_dte, log_m), 1e-10))
    end

    rmse = sqrt(mean((model_ivs .- ivs).^2))
    return σ_base, β, rmse, model_ivs
end

σ_base, β, rmse, model_ivs = fit_psi(fit_data)

println("\nFitted parameters:")
println("  σ_base = $(round(σ_base * 100, digits=2))% (base ATM IV level)")
println("  β₁ (DTE decay)         = $(round(β[1], digits=4))")
println("  β₂ (skew)              = $(round(β[2], digits=4))")
println("  β₃ (DTE × skew)        = $(round(β[3], digits=4))")
println("  β₄ (smile curvature)   = $(round(β[4], digits=4))")
println("  β₅ (DTE curvature)     = $(round(β[5], digits=4))")
println("  RMSE = $(round(rmse * 100, digits=2))% IV")

# Interpretation
println("\nInterpretation:")
println("  β₁ = $(round(β[1], digits=3)): ", β[1] > 0 ?
    "IV increases with DTE (contango / normal term structure)" :
    "IV decreases with DTE (backwardation / inverted term structure)")
println("  β₂ = $(round(β[2], digits=3)): ", β[2] < 0 ?
    "negative skew (OTM puts have higher IV than OTM calls)" :
    "positive skew (unusual — OTM calls have higher IV)")
println("  β₃ = $(round(β[3], digits=3)): ", β[3] > 0 ?
    "skew flattens at longer maturities" :
    "skew steepens at longer maturities")
println("  β₄ = $(round(β[4], digits=3)): ", β[4] > 0 ?
    "positive curvature (smile / U-shape)" :
    "negative curvature (frown — unusual)")
println("  β₅ = $(round(β[5], digits=3)): ", β[5] < 0 ?
    "U-shaped ATM term structure (elevated short + long dated IV)" :
    "inverted-U ATM term structure")

# ============================================================================
# Step 4: ATM term structure
# ============================================================================

println("\n" * "="^60)
println("  ATM TERM STRUCTURE")
println("="^60)

# Model-predicted ATM IV at each DTE (moneyness = 1.0 → log_m = 0)
# At ATM: ψ = exp(β₁·ln(DTE) + β₅·(ln(DTE))²)  [log_m terms vanish]
println("\n  DTE → ATM IV (model)     ATM IV (market avg)")
for dte in sort(unique(fit_data.DTE))
    log_dte = log(max(dte, 1.0))
    ψ_atm = exp(β[1] * log_dte + β[5] * log_dte^2)
    iv_model = σ_base * sqrt(ψ_atm) * 100

    # Market ATM: average IV of near-ATM options at this DTE
    slice = fit_data[(fit_data.DTE .== dte) .& (abs.(fit_data.Moneyness .- 1.0) .< 0.03), :]
    iv_market = nrow(slice) > 0 ? mean(slice.IV) * 100 : NaN

    @printf("  %3d → %5.1f%%              %5.1f%%\n", dte, iv_model, iv_market)
end

# Continuous curve
dte_range = 1:50
iv_atm_curve = [σ_base * sqrt(exp(β[1] * log(d) + β[5] * log(d)^2)) * 100 for d in dte_range]

# Find the DTE that minimizes ATM IV (trough of the U)
dte_min_idx = argmin(iv_atm_curve)
println("\n  ATM term structure shape:")
println("    IV at DTE=1:  $(round(iv_atm_curve[1], digits=1))%")
println("    IV trough at DTE≈$dte_min_idx: $(round(iv_atm_curve[dte_min_idx], digits=1))%")
println("    IV at DTE=46: $(round(iv_atm_curve[end-4], digits=1))%")
println("    Short/trough ratio: $(round(iv_atm_curve[1] / iv_atm_curve[dte_min_idx], digits=2))x")
println("    Long/trough ratio:  $(round(iv_atm_curve[end-4] / iv_atm_curve[dte_min_idx], digits=2))x")

# ============================================================================
# Step 5: Per-DTE residual analysis
# ============================================================================

println("\n" * "="^60)
println("  PER-DTE FIT QUALITY")
println("="^60)

fit_data[!, :ModelIV] = model_ivs
fit_data[!, :Residual] = model_ivs .- fit_data.IV

for dte in sort(unique(fit_data.DTE))
    slice = fit_data[fit_data.DTE .== dte, :]
    rmse_dte = sqrt(mean(slice.Residual .^ 2)) * 100
    bias_dte = mean(slice.Residual) * 100
    @printf("  DTE=%3d: RMSE=%5.2f%%, bias=%+5.2f%%, n=%d\n",
            dte, rmse_dte, bias_dte, nrow(slice))
end

# ============================================================================
# Step 6: Plots
# ============================================================================

println("\nGenerating plots...")

# --- Plot 1: IV vs Moneyness, colored by DTE ---
p1 = plot(title="NVDA IV Surface — Market vs Model",
          xlabel="Moneyness (K/S)", ylabel="Implied Volatility (%)",
          legend=:outerright, size=(900, 500), dpi=150)

dte_colors = Dict(0 => :red, 2 => :orange, 4 => :gold,
                  7 => :green, 14 => :teal, 25 => :blue, 46 => :purple)

for dte in sort(unique(fit_data.DTE))
    slice = fit_data[fit_data.DTE .== dte, :]
    c = get(dte_colors, dte, :black)

    scatter!(p1, slice.Moneyness, slice.IV .* 100,
             label="$(dte)d market", marker=:circle, ms=4, color=c, alpha=0.7)

    # Model curve across moneyness range for this DTE
    m_range = range(0.90, 1.10, length=50)
    log_dte = log(max(dte, 1.0))
    iv_curve = [σ_base * sqrt(eval_psi(β, log_dte, log(m))) * 100
                for m in m_range]
    plot!(p1, m_range, iv_curve, label="$(dte)d model", lw=2, color=c, ls=:solid)
end
vline!(p1, [1.0], label=nothing, ls=:dash, color=:gray, alpha=0.5)
savefig(p1, joinpath(PLOT_DIR, "iv_surface_term_structure.png"))
println("  → saved iv_surface_term_structure.png")

# --- Plot 2: ATM term structure ---
p2 = plot(title="NVDA ATM IV Term Structure",
          xlabel="Days to Expiration", ylabel="ATM Implied Volatility (%)",
          legend=:bottomright, size=(700, 450), dpi=150)

# Market ATM points
atm_dtes = Int[]
atm_ivs_market = Float64[]
for dte in sort(unique(fit_data.DTE))
    slice = fit_data[(fit_data.DTE .== dte) .& (abs.(fit_data.Moneyness .- 1.0) .< 0.03), :]
    if nrow(slice) > 0
        push!(atm_dtes, dte)
        push!(atm_ivs_market, mean(slice.IV) * 100)
    end
end
scatter!(p2, atm_dtes, atm_ivs_market,
         label="Market ATM", marker=:circle, ms=6, color=:blue)

# Model curve
plot!(p2, dte_range, iv_atm_curve,
      label="Model: σ·exp((β₁·ln(DTE) + β₅·ln(DTE)²)/2)", lw=2, color=:red)

savefig(p2, joinpath(PLOT_DIR, "atm_term_structure.png"))
println("  → saved atm_term_structure.png")

# --- Plot 3: Residuals by DTE ---
p3 = plot(title="IV Residuals by DTE (Model - Market)",
          xlabel="Moneyness (K/S)", ylabel="IV Residual (%)",
          legend=:outerright, size=(900, 400), dpi=150)
hline!(p3, [0.0], color=:gray, ls=:dash, label=nothing)

for dte in sort(unique(fit_data.DTE))
    slice = fit_data[fit_data.DTE .== dte, :]
    c = get(dte_colors, dte, :black)
    scatter!(p3, slice.Moneyness, slice.Residual .* 100,
             label="$(dte)d", marker=:circle, ms=4, color=c, alpha=0.7)
end
savefig(p3, joinpath(PLOT_DIR, "term_structure_residuals.png"))
println("  → saved term_structure_residuals.png")

# --- Plot 4: IV surface heatmap ---
p4 = plot(title="NVDA Model IV Surface",
          xlabel="Moneyness (K/S)", ylabel="Days to Expiration",
          size=(700, 500), dpi=150)

m_grid = range(0.90, 1.10, length=60)
dte_grid = range(1, 46, length=60)
iv_surface = [σ_base * sqrt(eval_psi(β, log(d), log(m))) * 100
              for d in dte_grid, m in m_grid]

heatmap!(p4, collect(m_grid), collect(dte_grid), iv_surface,
         colorbar_title="IV (%)", color=:viridis)

# Overlay market points
scatter!(p4, fit_data.Moneyness, Float64.(fit_data.DTE),
         marker=:circle, ms=3, color=:white, alpha=0.8, label="Market obs")
savefig(p4, joinpath(PLOT_DIR, "iv_surface_heatmap.png"))
println("  → saved iv_surface_heatmap.png")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^60)
println("  SUMMARY")
println("="^60)
println("\nFitted ψ parameters for θ(t) = θ_base · ψ(β, DTE, K/S):")
println("  σ_base = $(round(σ_base, digits=4)) ($(round(σ_base*100, digits=1))%)")
println("  β₁ = $(round(β[1], digits=4))  (DTE decay)")
println("  β₂ = $(round(β[2], digits=4))  (skew)")
println("  β₃ = $(round(β[3], digits=4))  (DTE × skew interaction)")
println("  β₄ = $(round(β[4], digits=4))  (smile curvature)")
println("  β₅ = $(round(β[5], digits=4))  (DTE curvature)")
println("  Overall RMSE = $(round(rmse*100, digits=2))% IV")
println("\nThese β values can be plugged directly into ThetaHybrid")
println("for synthetic data generation with JumpHMM.")
