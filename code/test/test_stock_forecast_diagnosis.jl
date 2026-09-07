using Test, JumpHMM, Distributions, LinearAlgebra, Statistics
include(joinpath(@__DIR__,"..","src","StockForecastDiagnosis.jl"))
using .StockForecastDiagnosis

@testset "Jump duration filtering and forecast boundary" begin
    m=JumpHiddenMarkovModel(LaplacePartition(0.0,1.0,2),[.7 .3;.2 .8],
        [StudentTEmission(-1.,.4,5.,20,false),StudentTEmission(1.,.3,5.,20,false)],
        [.4,.6],JumpParameters(.2,.7;p_neg=.6,N_tail=1),5.,0.,1/252)
    c=jump_components(m);a=initial_mass(m,c)
    a=condition(m,propagate(m,a,c),.8)
    # Enumerate every latent source state and duration independently.
    brute=zeros(size(a));p=c.probabilities
    for r in 0:size(a,2)-1,s in 1:2,k in 1:2
        weight=a[s,r+1]
        if r>0
            brute[k,r]+=weight*c.tail[k]
        else
            brute[k,1]+=weight*(1-m.jump.ϵ+m.jump.ϵ*p[1])*m.transition[s,k]
            for duration in 1:length(p)-1
                brute[k,duration]+=weight*m.jump.ϵ*p[duration+1]*c.tail[k]
            end
        end
    end
    @test propagate(m,a,c)≈brute atol=1e-14
    @test sum(brute)≈1
    likelihood=[pdf(TDist(e.ν),(.3-e.μ)/e.σ)/e.σ for e in m.emissions]
    expected=brute.*likelihood;expected./=sum(expected)
    @test condition(m,brute,.3)≈expected
    # A forced alternating chain must transition before the first future return.
    alternate=JumpHiddenMarkovModel(m.partition,[0. 1.;1. 0.],m.emissions,
        [.5,.5],JumpParameters(0.,.7;p_neg=.6,N_tail=1),5.,0.,1/252)
    start=initial_mass(alternate);start.=0;start[1,1]=1
    paths=simulate_forward(alternate,start,20000,2,42)
    @test abs(mean(paths[1,:])*252-1)<.02
    @test abs(mean(paths[2,:])*252+1)<.02
    @test paths==simulate_forward(alternate,start,20000,2,42)
    # The Monte Carlo first-emission mean agrees with exact latent propagation.
    simulated=simulate_forward(m,a,40000,1,43)
    expected_mean=sum(vec(sum(brute;dims=2)).*[e.μ for e in m.emissions])/252
    @test abs(mean(simulated)-expected_mean)<5std(simulated)/sqrt(length(simulated))
    prices=JumpHMM.prices_from_growth_rates([-3.,2.,.5],100.;rf=.03,dt=1/252)
    @test JumpHMM.excess_growth_rates(prices;rf=.03,dt=1/252)≈[-3.,2.,.5]
end
