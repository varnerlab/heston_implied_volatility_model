using Test, JumpHMM, Distributions, Random, Statistics

@testset "Conditional Student-t price emissions" begin
    model=JumpHiddenMarkovModel(LaplacePartition(0.,1.,2),[.7 .3;.2 .8],
        [StudentTEmission(-1.,.4,5.,20,false),StudentTEmission(1.,.3,5.,20,false)],
        [.4,.6],JumpParameters(.2,.7;p_neg=.6,N_tail=1),5.,.03,1/252)
    # A narrow test bound forces many rejections and distinguishes conditional
    # sampling from clipping, which would create mass at the two endpoints.
    spec=TruncatedStudentT(.8)
    raw=JumpHMM.simulate(model,3;n_paths=20000,seed=128)
    bounded=simulate_truncated(model,3;n_paths=20000,seed=128,emissions=spec)
    again=simulate_truncated(model,3;n_paths=20000,seed=128,emissions=spec)
    residuals=Float64[];rejected=0
    same_states=true;same_draws=true;same_accepted=true
    for (a,b,c) in zip(raw.paths,bounded.paths,again.paths)
        same_states &= a.states==b.states && a.jumps==b.jumps
        same_draws &= b.observations==c.observations
        for t in eachindex(a.states)
            e=model.emissions[a.states[t]]
            z=(b.observations[t]-e.μ)/e.σ
            push!(residuals,z)
            if abs((a.observations[t]-e.μ)/e.σ)<=spec.cutoff
                same_accepted &= a.observations[t]==b.observations[t]
            else
                rejected+=1
            end
        end
    end
    @test same_states
    @test same_draws
    @test same_accepted
    @test rejected>10000
    @test maximum(abs,residuals)<spec.cutoff
    distribution=truncated(TDist(5.),-.8,.8)
    for x in [-.5,0.,.5]
        @test abs(mean(residuals .<= x)-cdf(distribution,x))<.012
    end
    # Under a one-state model, compare the exponential moment against an
    # independent midpoint quadrature of the normalized conditional density.
    one=JumpHiddenMarkovModel(LaplacePartition(0.,1.,1),ones(1,1),
        [StudentTEmission(.1,.4,5.,20,false)],[1.],JumpParameters(0.,.7),5.,0.,1.)
    draws=simulate_truncated(one,1;n_paths=100000,seed=99,emissions=spec)
    prices=[exp(p.observations[1]) for p in draws.paths]
    grid=range(-.8+.8/20000,.8-.8/20000;length=20000)
    expected=sum(exp(.1+.4z)*pdf(distribution,z) for z in grid)*1.6/20000
    @test abs(mean(prices)-expected)<5std(prices)/sqrt(length(prices))
    @test all(exp(.1-.4*.8).<=prices.<=exp(.1+.4*.8))
    @test_throws ArgumentError TruncatedStudentT(Inf)
    @test_throws ArgumentError TruncatedStudentT(NaN)
    @test_throws ArgumentError TruncatedStudentT(0)
    @test_throws ArgumentError simulate_truncated(one,0)
    @test_throws ArgumentError simulate_truncated(one,1;start=2)
    @test emission_metadata(TruncatedStudentT())["standardized_cutoff"]==10.
    variance=sum(z^2*pdf(distribution,z) for z in grid)*1.6/20000
    @test truncated_t_variance(5.,spec)≈variance rtol=1e-7
    portfolio=JumpHMM.PortfolioModel(["A","B"],Dict("A"=>model,"B"=>model),
        JumpHMM.GaussianCopula([1. .3;.3 1.]),Dict("A"=>1,"B"=>2),"A")
    paths=simulate_truncated(portfolio,4;n_paths=100,seed=7,emissions=spec)
    @test paths.tickers==["A","B"]
    @test all(length(v.paths)==100 for v in values(paths.results))
    lower=minimum(e.μ-spec.cutoff*e.σ for e in model.emissions)
    upper=maximum(e.μ+spec.cutoff*e.σ for e in model.emissions)
    @test all(lower<=x<=upper for v in values(paths.results) for p in v.paths for x in p.observations)
end
