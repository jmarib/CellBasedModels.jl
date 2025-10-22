CellBasedModels.toCPU(cp::CommunityPoint{CommunityPointMeta}) = Adapt.adapt(SizedVector, cp)
CellBasedModels.toCPU(cp::CommunityPoint{CommunityPointMeta{<:SizedVector}}) = cp

CellBasedModels.toGPU(cp::CommunityPoint{CommunityPointMeta}) = cp
CellBasedModels.toGPU(cp::CommunityPoint{B}) where {B<:CommunityPointMeta{<:SizedVector}} = Adapt.adapt(CuArray, cp)

function Base.iterate(
        community::CommunityPoint{<:CommunityPointMeta{S}}, 
        state = (
                gridDim().x * blockDim().x,                         #Stride
                (blockIdx().x - 1) * blockDim().x + threadIdx().x  #Index
            )
    ) where {S<:CUDA.CuDeviceArray}
    state[2] >= community.N + 1 ? nothing : (state[2], (state[1], state[1] + state[2]))
end

######################################################################################################
# Iterator
######################################################################################################
function Base.iterate(
        community::CommunityPointIterator{<:CommunityPointMeta{S}}, 
        state = (
                gridDim().x * blockDim().x,                         #Stride
                (blockIdx().x - 1) * blockDim().x + threadIdx().x  #Index
            )
    ) where {S<:CUDA.CuDeviceArray}
    state[2] >= community.N + 1 ? nothing : (state[2], (state[1], state[1] + state[2]))
end

######################################################################################################
# Kernel functions
######################################################################################################
function removeAgent!(community::CommunityPoint{<:CommunityPointMeta{S}}, id::Int) where {S<:CUDA.CuDeviceArray}
    community._meta._reorderedFlag[1] = true
    community._meta._removedIDs[id] = true
end

@generated function addAgent!(community::CommunityPoint{CommunityPoint{<:CommunityPointMeta{S}}, D, P, T, NP, N, NC}, kwargs::NamedTuple{P, T2}) where {S<:CUDA.CuDeviceArray, D, P, T, NP, N, NC, T2}

    cases = [
        :(community.$name[newPos] = kwargs.$name)
        for name in P
    ]

    quote 
        newPos = CUDA.atomic_add!(community._meta._NNew, 1, 1)
        if newPos > community._meta._NCache[1]
                community._meta._overflowFlag[1] = true
                community._meta._NNew[1] = community._meta._NCache[1]
                return
        else
                newId = CUDA.atomic_add!(community._meta._idMax, 1, 1)
                community._meta._id[newPos] = newId
                $(cases...)
        end
     end
end