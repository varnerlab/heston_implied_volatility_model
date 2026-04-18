using Test
using HestonIV
using Distributions

# Black-Scholes reference for European-option sanity checks
const Φ = let N = Normal(0, 1); x -> cdf(N, x); end

function bs_call(S, K, σ, r, T)
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    return S * Φ(d1) - K * exp(-r * T) * Φ(d2)
end

function bs_put(S, K, σ, r, T)
    d1 = (log(S / K) + (r + 0.5 * σ^2) * T) / (σ * sqrt(T))
    d2 = d1 - σ * sqrt(T)
    return K * exp(-r * T) * Φ(-d2) - S * Φ(-d1)
end

@testset "CRRTree" begin
    @testset "European call — converges to Black-Scholes" begin
        S, K, σ, r, T = 100.0, 100.0, 0.20, 0.05, 0.5
        crr_price = crr_european_price(S, K, σ, r, T, 500, :call)
        bs_price = bs_call(S, K, σ, r, T)
        @test isapprox(crr_price, bs_price; atol=0.05)
    end

    @testset "European put — converges to Black-Scholes" begin
        S, K, σ, r, T = 100.0, 100.0, 0.20, 0.05, 0.5
        crr_price = crr_european_price(S, K, σ, r, T, 500, :put)
        bs_price = bs_put(S, K, σ, r, T)
        @test isapprox(crr_price, bs_price; atol=0.05)
    end

    @testset "European put-call parity: C - P = S - K·e^(-rT)" begin
        S, K, σ, r, T = 100.0, 105.0, 0.25, 0.04, 0.75
        c = crr_european_price(S, K, σ, r, T, 400, :call)
        p = crr_european_price(S, K, σ, r, T, 400, :put)
        @test isapprox(c - p, S - K * exp(-r * T); atol=0.05)
    end

    @testset "American ≥ European (early exercise premium ≥ 0)" begin
        S, K, σ, r, T = 100.0, 100.0, 0.30, 0.05, 1.0
        amer_put = crr_american_price(S, K, σ, r, T, 200, :put)
        euro_put = crr_european_price(S, K, σ, r, T, 200, :put)
        @test amer_put >= euro_put - 1e-9
        # No-dividend American call equals European call
        amer_call = crr_american_price(S, K, σ, r, T, 200, :call)
        euro_call = crr_european_price(S, K, σ, r, T, 200, :call)
        @test isapprox(amer_call, euro_call; atol=1e-2)
    end

    @testset "Deep ITM call ≈ S - K·e^(-rT) (intrinsic + carry)" begin
        S, K, σ, r, T = 200.0, 100.0, 0.20, 0.05, 0.25
        c = crr_european_price(S, K, σ, r, T, 300, :call)
        @test isapprox(c, S - K * exp(-r * T); atol=0.5)
    end

    @testset "Deep OTM call → near zero" begin
        S, K, σ, r, T = 50.0, 200.0, 0.20, 0.05, 0.25
        c = crr_european_price(S, K, σ, r, T, 300, :call)
        @test 0.0 <= c < 0.01
    end

    @testset "American put — deep ITM ≥ intrinsic value" begin
        S, K, σ, r, T = 50.0, 100.0, 0.20, 0.05, 1.0
        p = crr_american_price(S, K, σ, r, T, 200, :put)
        @test p >= K - S - 1e-9
    end

    @testset "Monotonic in volatility (call price increases with σ)" begin
        S, K, r, T = 100.0, 100.0, 0.05, 0.5
        prices = [crr_european_price(S, K, σ, r, T, 200, :call)
                  for σ in [0.10, 0.20, 0.30, 0.40]]
        @test issorted(prices)
    end

    @testset "Monotonic in time-to-expiration (American call)" begin
        S, K, σ, r = 100.0, 100.0, 0.25, 0.05
        prices = [crr_american_price(S, K, σ, r, T, 200, :call)
                  for T in [0.1, 0.5, 1.0, 2.0]]
        @test issorted(prices)
    end

    @testset "Input validation" begin
        @test_throws AssertionError crr_american_price(100.0, 100.0, 0.0, 0.05, 0.5, 100, :call)
        @test_throws AssertionError crr_american_price(100.0, 100.0, 0.2, 0.05, 0.0, 100, :call)
        @test_throws AssertionError crr_american_price(100.0, 100.0, 0.2, 0.05, 0.5, 0, :call)
        @test_throws AssertionError crr_american_price(100.0, 100.0, 0.2, 0.05, 0.5, 100, :forward)
    end

    @testset "price_contract — dispatch on style" begin
        contract_amer = OptionContract(100.0, 252, :put, :american)  # 1 trading year
        contract_euro = OptionContract(100.0, 252, :put, :european)
        S, σ, r = 100.0, 0.25, 0.05
        amer = price_contract(S, contract_amer, σ, r; n_steps=200)
        euro = price_contract(S, contract_euro, σ, r; n_steps=200)
        @test amer >= euro - 1e-9
    end
end
