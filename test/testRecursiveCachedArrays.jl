using BenchmarkTools
using CUDA

@testset "RecursiveCachedArrays" begin

    N = 1000000
    ns = [100, 20]

    f!(x,y,v) = v .= x .* 0.1 .+ y

    # Basic usage
    for fzeros in [zeros, CUDA.zeros]

        cache = (fzeros(N), fzeros(N))
        xt = CellBasedModels.TupleOfCachedArrays(cache, ns)
        xt.u[1] .= 1; xt.u[2] .= 1

        cache = (fzeros(N), fzeros(N))
        yt = CellBasedModels.TupleOfCachedArrays(cache, ns)
        yt.u[1] .= 10; yt.u[2] .= 10;

        cache = (fzeros(N).+1, fzeros(N).+1)
        vt = CellBasedModels.TupleOfCachedArrays(cache, ns)

        @test all(xt.ns .== ns) && all(yt.ns .== ns) && all(vt.ns .== ns)
        @test all(xt.u[1] .== 1) && all(xt.u[2] .== 1)
        @test all(yt.u[1] .== 10) && all(yt.u[2] .== 10)
        @test all(vt.u[1] .== 1) && all(vt.u[2] .== 1)

        f!(xt,yt,vt)

        @test all(vt.u[1][1:ns[1]] .≈ 10.1) && all(vt.u[2][1:ns[2]] .≈ 10.1)
        @test all(vt.u[1][ns[1]+1:end] .== 1) && all(vt.u[2][ns[2]+1:end] .== 1)
    
    end

    # Kernels working with TupleOfCachedArrays structure
    ns = CUDA.CuArray{Int32}([100,20])
    cache = (CUDA.zeros(N), CUDA.ones(N))
    vt = CellBasedModels.TupleOfCachedArrays(cache, ns)
    println(typeof(vt.u))
    println(typeof(vt.ns))
    function f!(vt)
        index = threadIdx().x    # this example only requires linear indexing, so just use `x`
        stride = blockDim().x
        @inbounds for i in index:stride:length(vt.u[1])
            vt.u[1][i] = 1.0 * vt.u[2][i]
        end
    end
    @cuda threads=254 f!(vt)
    @test all(vt.u[1][1:ns[1]] .≈ 1) && all(vt.u[2][1:ns[2]] .≈ 1)
    @test all(vt.u[1][ns[1]+1:end] .== 0) && all(vt.u[2][ns[2]+1:end] .== 1)

end

    # function kernel!(x, y, v)
    #     index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    #     stride = blockDim().x
    #     for i in index:stride:length(v)
    #         v[i] = x[i] * 0.1 + y[i]
    #     end
    #     return nothing
    # end

    # function kernelt!(s)
    #     index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    #     stride = blockDim().x
    #     for i in index:stride:length(v)
    #         s.u.v[i] = s.u.x[i] * 0.1 + s.u.y[i]
    #     end
    #     return nothing
    # end

    # cache = (CUDA.zeros(N), CUDA.zeros(N))
    # xt = CellBasedModels.TupleOfCachedArrays(cache, ns)
    # xt.u[1] .= 1; xt.u[2] .= 1
    
    # cache = (CUDA.zeros(N), CUDA.zeros(N))
    # yt = CellBasedModels.TupleOfCachedArrays(cache, ns)
    # yt.u[1] .= 10; yt.u[2] .= 10;

    # cache = (CUDA.zeros(N), CUDA.zeros(N))
    # vt = CellBasedModels.TupleOfCachedArrays(cache, ns)

    # f!(x,y,v) = v .= x .* 0.1 .+ y

    # x = CUDA.zeros(2*N) .+ 1; y = CUDA.zeros(2*N) .+ 10; v = CUDA.zeros(2*N);

    # @cuda threads=254 kernel!(x, y, v)
    # @cuda threads=254 kernelt!(s)
    # f!(x,y,v); f!(xt,yt,vt)
    # println(vt.u)
    # println(v)

# using DifferentialEquations

# @testset "CachedArrays integration" begin

#     # 1) Declare the nested sizes (leaf counts)
#     n = [3, 2]  # e.g. fluid (ρ,u), solid (x,y), particles (x,y,z,v)
#     params = (ones(n+10) for n in n)  # e.g.

#     # 2) Build the store (creates flat buffer + view proxy)
#     P = CellBasedModels.IntegratorParams(n, params)

#     @test P.n == n
#     @test length(P.u) == length(n)
#     # @test all(size(P[i]) == (n[i],) for i in 1:length(n))

#     # # 3) Update sizes (mutating)
#     # ns = [5, 5]
#     # CellBasedModels.update!(P, ns)  # change sizes
#     # @test all(size(P[i]) == (n[i],) for i in 1:length(n))

#     ns = [3, 2]

#     cache = [zeros(10), zeros(10)]
#     x = CellBasedModels.IntegratorParams(ns, cache)
#     x.u[1] .= 1; x.u[2] .= 2

#     cache = [zeros(10), zeros(10)]
#     u = CellBasedModels.IntegratorParams(ns, cache)
#     u.u[1] .= 10; u.u[2] .= 20;

#     cache = [zeros(10), zeros(10)]
#     v = CellBasedModels.IntegratorParams(ns, cache)

#     for ns in ([3,2], [5,5])
#         CellBasedModels.update!(x, n); x.u[1] .= 1;   x.u[2] .= 2
#         CellBasedModels.update!(u, n); u.u[1] .= 10;  u.u[2] .= 20
#         CellBasedModels.update!(v, n); v.u[1] .= 0.0; v.u[2] .= 0.0

#         v .= x .* 0.1# .+ u .+ 1.   # calls your * and + on IntegratorParams
#         @test all(v[1] .== 0.1) && all(v[2] .== 0.2)    

#         # v .= 0.1 .* x# .+ u .+ 1.   # calls your * and + on IntegratorParams
#         # @test all(v._params.u[1] .== 0.1) && all(v._params.u[2] .== 0.2) 
#         # @test all(v._params_cache.u[1][1:ns[1]] .== 0.1) && all(v._params_cache.u[2][1:ns[2]] .== 0.2)   

#         # v .= x .+ 0.1# .+ u .+ 1.   # calls your * and + on IntegratorParams
#         # @test all(v._params.u[1] .== 1.1) && all(v._params.u[2] .== 2.1)    
#         # @test all(v._params_cache.u[1][1:ns[1]] .== 1.1) && all(v._params_cache.u[2][1:ns[2]] .== 2.1)

#         # v .= 0.1 .+ x# .+ u .+ 1.   # calls your * and + on IntegratorParams
#         # @test all(v._params.u[1] .== 1.1) && all(v._params.u[2] .== 2.1)
#         # @test all(v._params_cache.u[1][1:ns[1]] .== 1.1) && all(v._params_cache.u[2][1:ns[2]] .== 2.1)

#         # v .= x .- 0.1# .+ u .+ 1.   # calls your * and + on IntegratorParams
#         # @test all(v._params.u[1] .== 0.9) && all(v._params.u[2] .== 1.9)
#         # @test all(v._params_cache.u[1][1:ns[1]] .== 0.9) && all(v._params_cache.u[2][1:ns[2]] .== 1.9)

#         # v .= 0.1 .- x# .+ u .+ 1.   # calls your * and + on IntegratorParams
#         # @test all(v._params.u[1] .== -0.9) && all(v._params.u[2] .== -1.9)
#         # @test all(v._params_cache.u[1][1:ns[1]] .== -0.9) && all(v._params_cache.u[2][1:ns[2]] .== -1.9)

#         # v .= x .* u
#         # @test all(v._params.u[1] .== 10.0) && all(v._params.u[2] .== 40.0)
#         # @test all(v._params_cache.u[1][1:ns[1]] .== 10.0) && all(v._params_cache.u[2][1:ns[2]] .== 40.0)

#         # v .= x .+ u   # calls your * and + on IntegratorParams
#         # @test all(v._params.u[1] .== 11.0) && all(v._params.u[2] .== 22.0)
#         # @test all(v._params_cache.u[1][1:ns[1]] .== 11.0) && all(v._params_cache.u[2][1:ns[2]] .== 22.0)

#         # v .= x .- u
#         # @test all(v._params.u[1] .== -9.0) && all(v._params.u[2] .== -18.0)
#         # @test all(v._params_cache.u[1][1:ns[1]] .== -9.0) && all(v._params_cache.u[2][1:ns[2]] .== -18.0)
#     end


#     # function f!(du, u, p, t)
#     #     @. du = 1
#     # end

#     # ns = [3, 2]
#     # cache = [zeros(10), zeros(10)]
#     # u = CellBasedModels.IntegratorParams(ns, cache)
#     # u._params.u[1] .= 10; u._params.u[2] .= 20;

#     # problem = ODEProblem(f!, u, (0.0, 1.0))
#     # integrator = init(problem, Euler(), dt=0.1)  # just to test init works
#     # step!(integrator)  # just to test step! works
#     # @test  all(integrator.u._params.u[1][1:ns[1]] .== 10.1) && all(integrator.u._params.u[2][1:ns[2]] .== 20.1)
#     # step!(integrator)  # just to test step! works
#     # @test  all(integrator.u._params.u[1][1:ns[1]] .≈ 10.2) && all(integrator.u._params.u[2][1:ns[2]] .≈ 20.2)

#     # ns = [5, 5]
#     # CellBasedModels.update(integrator, ns)
#     # println(integrator.u._params.u)
#     # @test  all(integrator.u._params.u[1][1:ns[1]] .≈ [10.2,10.2,10.2,0.0,0.0]) && all(integrator.u._params.u[2][1:ns[2]] .≈ [20.2,20.2,0.0,0.0,0.0])


#     # # 3) Access by properties (always zero-copy views into the flat buffer)
#     # ρ = ParamViews.params(P).fluid.rho   # SubArray view
#     # u = ParamViews.params(P).fluid.u
#     # @views ρ .= 1.0
#     # @views u .= 2.0

#     # # 4) Arithmetic (returns new ParamStore with same layout)
#     # Q = 0.5 * P + P
#     # @assert ParamViews.ndofs(Q) == ParamViews.ndofs(P)

#     # # 5) Use as u0 in DifferentialEquations.jl
#     # using DifferentialEquations

#     # f(u, p, t) = u - 0.1u          # works via +, *
#     # prob = ODEProblem((u,p,t)->f(u,p,t), P, (0.0, 1.0))
#     # sol  = solve(prob, Tsit5())

# end