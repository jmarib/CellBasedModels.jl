using DifferentialEquations

@testset "CachedArrays integration" begin

    # 1) Declare the nested sizes (leaf counts)
    ns = [3, 2]  # e.g. fluid (ρ,u), solid (x,y), particles (x,y,z,v)
    params_cache = [ones(n+10) for n in ns]  # e.g.

    # 2) Build the store (creates flat buffer + view proxy)
    P = CellBasedModels.IntegratorParams(ns, params_cache)

    @test P._ns == ns
    @test P._params_cache == params_cache
    @test length(P._params) == length(ns)
    @test all(size(P._params.u[i]) == (ns[i],) for i in 1:length(ns))
    @test all(P._params.u[i] === @views params_cache[i][1:ns[i]] for i in 1:length(ns))

    # 3) Update sizes (mutating)
    ns = [5, 5]
    P = CellBasedModels.update(P, ns)  # change sizes
    @test P._ns == ns
    @test all(size(P._params.u[i]) == (P._ns[i],) for i in 1:length(ns))
    @test all(size(P._params.u[i]) == (ns[i],) for i in 1:length(ns))
    @test all(P._params.u[i] === @views params_cache[i][1:ns[i]] for i in 1:length(ns))

    ns = [3, 2]

    cache = [zeros(10), zeros(10)]
    x = CellBasedModels.IntegratorParams(ns, cache)
    x._params.u[1] .= 1; x._params.u[2] .= 2

    cache = [zeros(10), zeros(10)]
    u = CellBasedModels.IntegratorParams(ns, cache)
    u._params.u[1] .= 10; u._params.u[2] .= 20;

    cache = [zeros(10), zeros(10)]
    v = CellBasedModels.IntegratorParams(ns, cache)

    for ns in ([3,2], [5,5])
        x._ns .= ns; x = CellBasedModels.update(x); x._params.u[1] .= 1; x._params.u[2] .= 2
        u._ns .= ns; u = CellBasedModels.update(u); u._params.u[1] .= 10; u._params.u[2] .= 20
        v._ns .= ns; v = CellBasedModels.update(v); v._params.u[1] .= 0.0; v._params.u[2] .= 0.0

        v .= x .* 0.1# .+ u .+ 1.   # calls your * and + on IntegratorParams
        @test all(v._params.u[1] .== 0.1) && all(v._params.u[2] .== 0.2)    
        @test all(v._params_cache.u[1][1:ns[1]] .== 0.1) && all(v._params_cache.u[2][1:ns[2]] .== 0.2)

        v .= 0.1 .* x# .+ u .+ 1.   # calls your * and + on IntegratorParams
        @test all(v._params.u[1] .== 0.1) && all(v._params.u[2] .== 0.2) 
        @test all(v._params_cache.u[1][1:ns[1]] .== 0.1) && all(v._params_cache.u[2][1:ns[2]] .== 0.2)   

        v .= x .+ 0.1# .+ u .+ 1.   # calls your * and + on IntegratorParams
        @test all(v._params.u[1] .== 1.1) && all(v._params.u[2] .== 2.1)    
        @test all(v._params_cache.u[1][1:ns[1]] .== 1.1) && all(v._params_cache.u[2][1:ns[2]] .== 2.1)

        v .= 0.1 .+ x# .+ u .+ 1.   # calls your * and + on IntegratorParams
        @test all(v._params.u[1] .== 1.1) && all(v._params.u[2] .== 2.1)
        @test all(v._params_cache.u[1][1:ns[1]] .== 1.1) && all(v._params_cache.u[2][1:ns[2]] .== 2.1)

        v .= x .- 0.1# .+ u .+ 1.   # calls your * and + on IntegratorParams
        @test all(v._params.u[1] .== 0.9) && all(v._params.u[2] .== 1.9)
        @test all(v._params_cache.u[1][1:ns[1]] .== 0.9) && all(v._params_cache.u[2][1:ns[2]] .== 1.9)

        v .= 0.1 .- x# .+ u .+ 1.   # calls your * and + on IntegratorParams
        @test all(v._params.u[1] .== -0.9) && all(v._params.u[2] .== -1.9)
        @test all(v._params_cache.u[1][1:ns[1]] .== -0.9) && all(v._params_cache.u[2][1:ns[2]] .== -1.9)

        v .= x .* u
        @test all(v._params.u[1] .== 10.0) && all(v._params.u[2] .== 40.0)
        @test all(v._params_cache.u[1][1:ns[1]] .== 10.0) && all(v._params_cache.u[2][1:ns[2]] .== 40.0)

        v .= x .+ u   # calls your * and + on IntegratorParams
        @test all(v._params.u[1] .== 11.0) && all(v._params.u[2] .== 22.0)
        @test all(v._params_cache.u[1][1:ns[1]] .== 11.0) && all(v._params_cache.u[2][1:ns[2]] .== 22.0)

        v .= x .- u
        @test all(v._params.u[1] .== -9.0) && all(v._params.u[2] .== -18.0)
        @test all(v._params_cache.u[1][1:ns[1]] .== -9.0) && all(v._params_cache.u[2][1:ns[2]] .== -18.0)
    end


    function f!(du, u, p, t)
        @. du = 1
    end

    ns = [3, 2]
    cache = [zeros(10), zeros(10)]
    u = CellBasedModels.IntegratorParams(ns, cache)
    u._params.u[1] .= 10; u._params.u[2] .= 20;

    problem = ODEProblem(f!, u, (0.0, 1.0))
    integrator = init(problem, Euler(), dt=0.1)  # just to test init works
    step!(integrator)  # just to test step! works
    @test  all(integrator.u._params.u[1][1:ns[1]] .== 10.1) && all(integrator.u._params.u[2][1:ns[2]] .== 20.1)
    step!(integrator)  # just to test step! works
    @test  all(integrator.u._params.u[1][1:ns[1]] .≈ 10.2) && all(integrator.u._params.u[2][1:ns[2]] .≈ 20.2)

    ns = [5, 5]
    CellBasedModels.update(integrator, ns)
    println(integrator.u._params.u)
    @test  all(integrator.u._params.u[1][1:ns[1]] .≈ [10.2,10.2,10.2,0.0,0.0]) && all(integrator.u._params.u[2][1:ns[2]] .≈ [20.2,20.2,0.0,0.0,0.0])


    # # 3) Access by properties (always zero-copy views into the flat buffer)
    # ρ = ParamViews.params(P).fluid.rho   # SubArray view
    # u = ParamViews.params(P).fluid.u
    # @views ρ .= 1.0
    # @views u .= 2.0

    # # 4) Arithmetic (returns new ParamStore with same layout)
    # Q = 0.5 * P + P
    # @assert ParamViews.ndofs(Q) == ParamViews.ndofs(P)

    # # 5) Use as u0 in DifferentialEquations.jl
    # using DifferentialEquations

    # f(u, p, t) = u - 0.1u          # works via +, *
    # prob = ODEProblem((u,p,t)->f(u,p,t), P, (0.0, 1.0))
    # sol  = solve(prob, Tsit5())

end