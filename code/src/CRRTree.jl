"""
    CRRTree.jl

Cox-Ross-Rubinstein binomial tree for American option pricing.
"""

"""
    crr_american_price(S, K, σ, r, T, n_steps, option_type; q=0.0) → Float64

Price an American option using the CRR binomial tree.

# Arguments
- `S::Float64`: current underlying price
- `K::Float64`: strike price
- `σ::Float64`: implied volatility (annualized)
- `r::Float64`: risk-free rate (annualized)
- `T::Float64`: time to expiration (in years)
- `n_steps::Int`: number of tree steps
- `option_type::Symbol`: `:call` or `:put`
- `q::Float64`: continuous dividend yield (annualized; default 0.0)

# Returns
The American option price.
"""
function crr_american_price(S::Float64, K::Float64, σ::Float64,
                            r::Float64, T::Float64, n_steps::Int,
                            option_type::Symbol;
                            q::Float64=0.0)::Float64
    @assert option_type in (:call, :put) "option_type must be :call or :put"
    @assert σ > 0.0 "volatility must be positive"
    @assert T > 0.0 "time to expiration must be positive"
    @assert n_steps > 0 "n_steps must be positive"

    Δt = T / n_steps
    u = exp(σ * sqrt(Δt))
    d = 1.0 / u
    R = exp((r - q) * Δt)
    p = (R - d) / (u - d)
    qprob = 1.0 - p

    # Terminal payoffs at expiration (step n_steps)
    prices = Vector{Float64}(undef, n_steps + 1)
    values = Vector{Float64}(undef, n_steps + 1)

    for i in 0:n_steps
        prices[i+1] = S * u^(n_steps - i) * d^i
        values[i+1] = _payoff(prices[i+1], K, option_type)
    end

    # Backward induction with early exercise
    disc = 1.0 / R
    for step in (n_steps-1):-1:0
        for i in 0:step
            S_node = S * u^(step - i) * d^i
            hold_value = disc * (p * values[i+1] + qprob * values[i+2])
            exercise_value = _payoff(S_node, K, option_type)
            values[i+1] = max(hold_value, exercise_value)
        end
    end

    return values[1]
end

"""
    crr_european_price(S, K, σ, r, T, n_steps, option_type; q=0.0) → Float64

Price a European option using the CRR binomial tree (no early exercise).
Useful for validation against Black-Scholes.

Accepts a continuous dividend yield `q` (default 0.0). The risk-neutral drift
becomes `(r − q)Δt` so non-zero dividends shift the up-state probability the
same way Black–Scholes-Merton does.
"""
function crr_european_price(S::Float64, K::Float64, σ::Float64,
                            r::Float64, T::Float64, n_steps::Int,
                            option_type::Symbol;
                            q::Float64=0.0)::Float64
    @assert option_type in (:call, :put) "option_type must be :call or :put"
    @assert σ > 0.0 "volatility must be positive"
    @assert T > 0.0 "time to expiration must be positive"
    @assert n_steps > 0 "n_steps must be positive"

    Δt = T / n_steps
    u = exp(σ * sqrt(Δt))
    d = 1.0 / u
    R = exp((r - q) * Δt)
    p = (R - d) / (u - d)
    qprob = 1.0 - p

    values = Vector{Float64}(undef, n_steps + 1)
    for i in 0:n_steps
        S_terminal = S * u^(n_steps - i) * d^i
        values[i+1] = _payoff(S_terminal, K, option_type)
    end

    disc = 1.0 / R
    for step in (n_steps-1):-1:0
        for i in 0:step
            values[i+1] = disc * (p * values[i+1] + qprob * values[i+2])
        end
    end

    return values[1]
end

"""
    price_contract(S, contract, σ, r; n_steps=200, q=0.0) → Float64

Price an OptionContract given current underlying price and IV.

`contract.DTE` is calendar days to expiration, converted to years with a
365-day calendar — the same convention the ladder corpus and ScenarioTemplate
use throughout. (Trading-step counts are a separate concept; see `n_sim_steps`
in Pipeline.jl.)
"""
function price_contract(S::Float64, contract::OptionContract, σ::Float64,
                        r::Float64; n_steps::Int=200, q::Float64=0.0)::Float64
    T = contract.DTE / 365.0  # calendar DTE → years
    if contract.style == :american
        return crr_american_price(S, contract.K, σ, r, T, n_steps,
                                  contract.option_type; q=q)
    else
        return crr_european_price(S, contract.K, σ, r, T, n_steps,
                                  contract.option_type; q=q)
    end
end

# --- Internal helpers ---

function _payoff(S::Float64, K::Float64, option_type::Symbol)::Float64
    if option_type == :call
        return max(S - K, 0.0)
    else
        return max(K - S, 0.0)
    end
end
