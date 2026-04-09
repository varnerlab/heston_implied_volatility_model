"""
    HestonIV

Modified Heston implied volatility model for American option pricing.

Pipeline: JumpHMM synthetic prices → Heston dv variance process → IV → CRR binomial tree → American option prices

The key innovation is that the Heston mean-reversion target θ is a hybrid function
of the JumpHMM state, days to expiration, moneyness, and aggregate market mood:

    θ(t) = θ_{s_t} · (1 + γ·M_t) · ψ(DTE, K/S_t)

where ψ = exp(β₁·ln(DTE) + β₂·ln(K/S) + β₃·ln(DTE)·ln(K/S))
"""
module HestonIV

using Statistics
using Random
using LinearAlgebra
using DataFrames
using Distributions
using Optim

import JumpHMM

# Source files
include("Types.jl")
include("ThetaFunction.jl")
include("HestonVariance.jl")
include("CRRTree.jl")
include("Pipeline.jl")
include("Calibration.jl")

# --- Exports ---

# Types
export HestonParameters, ThetaHybrid, OptionContract, IVPoint, ScenarioResult
export CalibrationData

# θ-function
export compute_theta, ψ, compute_mood, compute_mood_path, auto_calibrate_theta_states

# Heston variance
export simulate_variance, simulate_variance_ensemble
export variance_to_iv, variance_path_to_iv

# CRR tree
export crr_american_price, crr_european_price, price_contract

# Pipeline
export run_scenario, run_single_asset_scenario, summarize

# Calibration
export prepare_calibration_data, calibrate

end # module
