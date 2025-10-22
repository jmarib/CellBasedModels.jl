using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector
import CellBasedModels: CommunityPointMeta
using Logging

agent = AgentPoint(3)
community = CommunityPoint(agent, 1000000)
community_gpu = toGPU(community)
x = CUDA.zeros(Float64, 1000000)

kernel_iterate_gpu!(community, x) = begin
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x     
    stride = gridDim().x * blockDim().x
    @inbounds for i in index:stride:community.N
        x[i] = 1
    end
end

@btime CUDA.@cuda threads=252 kernel_iterate_gpu!(community_gpu, x)

function Base.iterate(
        community::CommunityPoint{<:CommunityPointMeta{S}}, 
        state = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ) where {S<:CUDA.CuDeviceArray}
    stride = gridDim().x * blockDim().x
    # CUDA.@cuprintln("Iterate state: ", (state, state + stride))
    state >= community.N + 1 ? nothing : (state, state + stride)
end

kernel_iterate_gpu!(community, x) = @inbounds for i in community
    x[i] = 1
end

@btime CUDA.@cuda threads=252 kernel_iterate_gpu!(community_gpu, x)

function Base.iterate(
        community::CommunityPoint{<:CommunityPointMeta{S}}, 
        state = (gridDim().x * blockDim().x, (blockIdx().x - 1) * blockDim().x + threadIdx().x)
    ) where {S<:CUDA.CuDeviceArray}
    # stride = gridDim().x * blockDim().x
    # CUDA.@cuprintln("Iterate state: ", (state, state + stride))
    state[2] >= community.N + 1 ? nothing : (state[2], (state[1], state[1] + state[2]))
end

kernel_iterate_gpu!(community, x) = @inbounds for i in community
    x[i] = 1
end

@btime CUDA.@cuda threads=252 kernel_iterate_gpu!(community_gpu, x)

struct CommunityPointIteratorTest{B}
    N::Int
end
Adapt.@adapt_structure CommunityPointIteratorTest

function loopOverAgents(community::CommunityPoint{B, D, P, T, NP, N, NC}) where {B, D, P, T, NP, N, NC}
    CommunityPointIteratorTest{B}(N)
end

function Base.iterate(iterator::CommunityPointIteratorTest, state = 1)
    state >= iterator.N + 1 ? nothing : (state, state + 1)
end

function Base.iterate(
        community::CommunityPointIteratorTest{<:CommunityPointMeta{S}}, 
        state = (
                gridDim().x * blockDim().x,                         #Stride
                (blockIdx().x - 1) * blockDim().x + threadIdx().x  #Index
            )
    ) where {S<:CUDA.CuDeviceArray}
    state[2] >= community.N + 1 ? nothing : (state[2], (state[1], state[1] + state[2]))
end

kernel_iterate_gpu!(community, x) = @inbounds for i in loopOverAgents(community)
    x[i] = 1
end

@btime CUDA.@cuda threads=252 kernel_iterate_gpu!(community_gpu, x)

