import CellBasedModels: CommunityPoint, CommunityPointMeta, CommunityPointIterator, toCPU, toGPU, removeAgent!, addAgent!

toCPU(cp::CommunityPoint{CommunityPointMeta}) = cp
toGPU(cp::CommunityPoint{CommunityPointMeta}) = cp

function toCPU(cp::CommunityPoint{CommunityPointMeta{<:Union{CUDA.CuArray, CUDA.CuDeviceArray}}})
    CommunityPoint(
        Adapt.adapt(Vector, cp._pa),
        Adapt.adapt(
            CommunityPointMeta{Threads.Atomic{Int}, SizedVector{length(cp._m._id), Int}, SizedVector{length(cp._m._removed), Bool}, Threads.Atomic{1, Bool}},
            cp._m,
        ),
        cp._paCopy,
        cp.N,
        cp.NCache,
    )
end

function toGPU(cp::CommunityPoint{B}) where {B<:CommunityPointMeta{<:Threads.Atomic}}
    CommunityPoint(
        Adapt.adapt(CUDA.CuArray, cp._pa),
        CommunityPointMeta(
            CUDA.CuArray([cp._m._N[]]),
            CUDA.CuArray([cp._m._NCache[]]),
            CUDA.CuArray([cp._m._NNew[]]),
            CUDA.CuArray([cp._m._idMax[]]),
            Adapt.adapt(CUDA.CuArray, cp._m._id),
            Adapt.adapt(CUDA.CuArray, cp._m._removed),
            CUDA.CuArray([cp._m._reorderedFlag[]]),
            CUDA.CuArray([cp._m._overflowFlag[]])
        ),
        cp._paCopy,
        cp.N,
        cp.NCache,
    )
end

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
function removeAgent!(community::CommunityPoint{<:CommunityPointMeta{S}}, pos::Int) where {S<:CUDA.CuArray}
    error("removeAgent! for GPU CommunityPoint structures is not allowed outside kernels. Do it inside a kernel or on CPU before passing it to the gpu.")
    return
end

function removeAgent!(community::CommunityPoint{<:CommunityPointMeta{S}}, pos::Int) where {S<:CUDA.CuDeviceArray}
    if pos < 1 || pos > community.N
        CUDA.@cuprintln "Position $pos is out of bounds for CommunityPoint with N=$(community.N). No agent removed."
    else
        community._m._reorderedFlag[1] = true
        community._m._removed[pos] = true
    end
    return
end

function addAgent!(community::CommunityPoint{<:CommunityPointMeta{S}}, kwargs) where {S<:CUDA.CuArray}
    error("addAgent! for GPU CommunityPoint structures is not allowed outside kernels. Do it inside a kernel or on CPU before passing it to the gpu.")
    return
end

@generated function addAgent!(community::CommunityPoint{<:CommunityPointMeta{S}, D, P, T, NP, N, NC}, kwargs::NamedTuple{P2, T2}) where {S<:CUDA.CuDeviceArray, D, P, T, NP, N, NC, P2, T2}

    for i in P2
        if !(i in P)
            error("Property $i not found in CommunityPoint. Properties that have to be provided to addAgent! are: $(P).")
        end
    end

    for i in P
        if !(i in P2)
            error("Property $i not provided in addAgent! arguments. You must provide all properties. Provided properties are: $(P2). Properties that have to be provided are: $(P).")
        end
    end

    cases = [
        :(community._pa.$name[newPos] = kwargs.$name)
        for name in P
    ]

    quote 
        newPos = CUDA.atomic_add!(pointer(community._m._NNew, 1), 1) + 1
        if newPos > community._m._NCache[1]
                community._m._overflowFlag[1] = true
                community._m._NNew[1] = community._m._NCache[1]
                return
        else
                newId = CUDA.atomic_add!(pointer(community._m._idMax, 1), 1) + 1
                community._m._id[newPos] = newId
                $(cases...)
        end
    end

end