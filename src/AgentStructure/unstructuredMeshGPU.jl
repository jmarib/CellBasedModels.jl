import CellBasedModels: UnstructuredMeshObjectField

CellBasedModels.toCPU(field::UnstructuredMeshObjectField{TR, IDVI, IDAI, AI}) where {TR, IDVI, IDAI, AI<:Threads.Atomic} = field
CellBasedModels.toGPU(field::UnstructuredMeshObjectField{TR, IDVI, IDAI, AI}) where {TR, IDVI, IDAI, AI<:CUDA.CuArray} = field

function toCPU(field::UnstructuredMeshObjectField{TR, IDVI, IDAI, AI}) where {TR, IDVI, IDAI, AI<:CUDA.CuArray}
    UnstructuredMeshObjectField(
        field._id             === nothing ? nothing : Vector{Int}(field._id),
        field._idMax          === nothing ? nothing : Threads.Atomic{Int}(Array(field._idMax)[1]),
        field._N              === nothing ? nothing : Threads.Atomic{Int}(Array(field._N)[1]),
        field._NCache         === nothing ? nothing : Threads.Atomic{Int}(Array(field._NCache)[1]),
        field._FlagsRemoved   === nothing ? nothing : Vector{Bool}(field._FlagsRemoved),
        field._NRemoved       === nothing ? nothing : Threads.Atomic{Int}(0),
        field._NRemovedThread === nothing ? nothing : SizedVector{Threads.nthreads(), Int}(zeros(Int, Threads.nthreads())),
        field._NAdded         === nothing ? nothing : Threads.Atomic{Int}(0),
        field._NAddedThread   === nothing ? nothing : SizedVector{Threads.nthreads(), Int}(zeros(Int, Threads.nthreads())),
        field._AddedAgents    === nothing ? nothing : [Vector{NamedTuple{keys(field._p), Tuple{[CellBasedModels.standardDataType(eltype(i)) for i in values(field._p)]...}}}() for _ in 1:Threads.nthreads()],
        field._FlagOverflow   === nothing ? nothing : Threads.Atomic{Bool}(false),
        field._p              === nothing ? nothing : Adapt.adapt(Array, field._p),
        field._NP             === nothing ? nothing : field._NP,
        field._pReference     === nothing ? nothing : SizedVector{field._NP, Bool}([field._pReference...])
    )
end

function toGPU(field::UnstructuredMeshObjectField{TR, IDVI, IDAI, AI}) where {TR, IDVI, IDAI, AI<:Threads.Atomic}
    UnstructuredMeshObjectField(
        field._id             === nothing ? nothing : CUDA.CuArray(field._id),
        field._idMax          === nothing ? nothing : CUDA.CuArray([field._idMax[]]),
        field._N              === nothing ? nothing : CUDA.CuArray([field._N[]]),
        field._NCache         === nothing ? nothing : CUDA.CuArray([field._NCache[]]),
        field._FlagsRemoved   === nothing ? nothing : CUDA.CuArray(field._FlagsRemoved),
        field._NRemoved       === nothing ? nothing : CUDA.CuArray([0]),
        field._NRemovedThread === nothing ? nothing : CUDA.zeros(0),
        field._NAdded         === nothing ? nothing : CUDA.CuArray([0]),
        field._NAddedThread   === nothing ? nothing : CUDA.zeros(0),
        field._AddedAgents    === nothing ? nothing : CUDA.zeros(0),
        field._FlagOverflow   === nothing ? nothing : CUDA.CuArray([false]),
        field._p              === nothing ? nothing : Adapt.adapt(CUDA.CuArray, field._p),
        field._NP             === nothing ? nothing : field._NP,
        field._pReference     === nothing ? nothing : tuple(field._pReference...),
    )
end

Base.length(field::UnstructuredMeshObjectField{TR, IDVI, IDAI, AI}) where {TR, IDVI, IDAI, AI<:CUDA.CuArray} = CUDA.@allowscalar field._N[1]
Base.length(field::UnstructuredMeshObjectField{TR, IDVI, IDAI, AI}) where {TR, IDVI, IDAI, AI<:CUDA.CuDeviceArray} = field._N[1]
lengthCache(field::UnstructuredMeshObjectField{TR, IDVI, IDAI, AI}) where {TR, IDVI, IDAI, AI<:CUDA.CuArray} = CUDA.@allowscalar field._NCache[1]
lengthCache(field::UnstructuredMeshObjectField{TR, IDVI, IDAI, AI}) where {TR, IDVI, IDAI, AI<:CUDA.CuDeviceArray} = field._NCache[1]

# function Base.iterate(
#         community::CommunityPoint{<:CommunityPointMeta{TR, S}}, 
#         state = (
#                 gridDim().x * blockDim().x,                         #Stride
#                 (blockIdx().x - 1) * blockDim().x + threadIdx().x  #Index
#             )
#     ) where {TR, S<:CUDA.CuDeviceArray}
#     state[2] >= community._m._N[1] + 1 ? nothing : (state[2], (state[1], state[1] + state[2]))
# end

# ######################################################################################################
# # Iterator
# ######################################################################################################
# function Base.iterate(
#         iterator::CommunityPointIterator{<:CommunityPointMeta{TR, S}}, 
#         state = (
#                 gridDim().x * blockDim().x,                         #Stride
#                 (blockIdx().x - 1) * blockDim().x + threadIdx().x  #Index
#             )
#     ) where {TR, S<:CUDA.CuDeviceArray}
#     state[2] >= iterator.N + 1 ? nothing : (state[2], (state[1], state[1] + state[2]))
# end

# ######################################################################################################
# # Kernel functions
# ######################################################################################################
# function removeAgent!(community::CommunityPoint{<:CommunityPointMeta{TR, S}}, pos::Int) where {TR, S<:CUDA.CuArray}
#     error("removeAgent! for GPU CommunityPoint structures is not allowed outside kernels. Do it inside a kernel or on CPU before passing it to the gpu.")
#     return
# end

# function removeAgent!(community::CommunityPoint{<:CommunityPointMeta{TR, S}}, pos::Int) where {TR, S<:CUDA.CuDeviceArray}
#     if pos < 1 || pos > length(community)
#         CUDA.@cuprintln "Position $pos is out of bounds for CommunityPoint with N=$(length(community)). No agent removed."
#     else
#         community._m._flagsRemoved[pos] = true
#     end
#     return
# end

# function addAgent!(community::CommunityPoint{<:CommunityPointMeta{TR, S}}, kwargs) where {TR, S<:CUDA.CuArray}
#     error("addAgent! for GPU CommunityPoint structures is not allowed outside kernels. Do it inside a kernel or on CPU before passing it to the gpu.")
#     return
# end

# @generated function addAgent!(community::CommunityPoint{<:CommunityPointMeta{TR, S}, D, P, T, NP}, kwargs::NamedTuple{P2, T2}) where {TR, S<:CUDA.CuDeviceArray, D, P, T, NP, P2, T2}

#     for i in P2
#         if !(i in P)
#             error("Property $i not found in CommunityPoint. Properties that have to be provided to addAgent! are: $(P).")
#         end
#     end

#     for i in P
#         if !(i in P2)
#             error("Property $i not provided in addAgent! arguments. You must provide all properties. Provided properties are: $(P2). Properties that have to be provided are: $(P).")
#         end
#     end

#     cases = [
#         :(community._pa.$name[newPos] = kwargs.$name)
#         for name in P
#     ]

#     quote 
#         newPos = CUDA.atomic_add!(pointer(community._m._NNew, 1), 1) + 1
#         if newPos > community._m._NCache[1]
#                 community._m._overflowFlag[1] = true
#                 community._m._NNew[1] = community._m._NCache[1]
#                 return
#         else
#                 newId = CUDA.atomic_add!(pointer(community._m._idMax, 1), 1) + 1
#                 community._m._id[newPos] = newId
#                 $(cases...)
#         end
#     end

# end