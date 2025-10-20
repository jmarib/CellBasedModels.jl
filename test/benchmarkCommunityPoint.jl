using BenchmarkTools
using Adapt
using DifferentialEquations
# using FastBroadcast

N = 10^6
n = 10^6

agent = AgentPoint(2)

community = CommunityPoint(agent, n, N)

f!(x) = x .= 5 .* x .+ 1

# @btime f!(community)

# function to_gpu(cp::CommunityPoint{D, P, N, NC}) where {D, P, N, NC}
#     CommunityPoint{D, P, N, NC}(
#         Adapt.adapt(CuArray, cp._propertiesAgent),
#         Adapt.adapt(CuArray, cp._N),
#         Adapt.adapt(CuArray, cp._NCache),
#         Adapt.adapt(CuArray, cp._NNew),
#         Adapt.adapt(CuArray, cp._idMax),
#         Adapt.adapt(CuArray, cp._NFlag),
#     )
# end
# community_gpu = to_gpu(community)
# community_gpu = Adapt.adapt(CuArray, community)

# function f_gpu!(community) 
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = gridDim().x * blockDim().x
#     x = community.x
#     for i in index:stride:community._N[1]
#         @inbounds x[i] = 5 * (x[i] + 1.0)
#     end
# end

# CUDA.@cuda threads=32 f_gpu!(community_gpu)

# Base.@propagate_inbounds function FastBroadcast.fast_materialize!(
#         dst::A, bc::Broadcasted{S}) where {S, A}
#     if S === Base.Broadcast.DefaultArrayStyle{0}
#         fill!(dst, bc[1])
#     elseif S <: Base.Broadcast.DefaultArrayStyle
#         println("holi")
#         DifferentialEquations.DiffEqBase.FastBroadcast._fast_materialize!(dst, Val(indices_do_not_alias(A)), bc)
#     else
#         materialize!(dst, bc)
#     end
# end

f2(u, uprev, dt, fsalfirst) = @. u=uprev + dt * fsalfirst
f3(u, uprev, dt, fsalfirst) = DifferentialEquations.DiffEqBase.@.. broadcast=false u=uprev + dt * fsalfirst

agent = AgentPoint(1)
community = CommunityPoint(agent, n, N)

community_gpu = Adapt.adapt(CuArray, community)
# community_gpu = community

# community_gpu = zeros(10000)
fODE!(du, u, p, t) = @. du = 5 * (u + 1.0)

prob = ODEProblem(fODE!, community, (0.0, 1.0))
integrator = init(prob, Euler(), dt=0.1, save_everystep=false)

step!(integrator)
@btime step!(integrator)

prob = ODEProblem(fODE!, community_gpu, (0.0, 1.0))
integrator = init(prob, Euler(), dt=1.0, save_everystep=false)
step!(integrator)
@btime step!(integrator)

t = integrator.t
dt = integrator.dt
uprev = integrator.uprev
u = integrator.u
f = integrator.f
p = integrator.p
fsalfirst = integrator.fsalfirst
# println(uprev, dt, fsalfirst)
# @btime f2(u, uprev, dt, fsalfirst)
# @btime f3(u, uprev, dt, fsalfirst)
