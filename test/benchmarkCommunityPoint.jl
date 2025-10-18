using BenchmarkTools
using Adapt

N = 100000
n = 100

agent = AgentPoint(2)

community = CommunityPoint(agent, n, N)

f!(x) = x .= 5 .* x .+ 1

@btime f!(community)

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
community_gpu = Adapt.adapt(CuArray, community)

function f_gpu!(community) 
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    x = community.x
    for i in index:stride:community._N[1]
        @inbounds x[i] = 5 * (x[i] + 1.0)
    end
end

CUDA.@cuda threads=32 f_gpu!(community_gpu)