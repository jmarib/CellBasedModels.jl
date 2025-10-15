# Based on code from M. Bauman Stackexchange answer + Gitter discussion

# using DocStringExtensions
# using RecipesBase
using StaticArraysCore
# using Statistics
using ArrayInterface
# using LinearAlgebra
using SymbolicIndexingInterface
import Adapt
# using BenchmarkTools
# using CUDA

abstract type AbstractTupleOfCachedArrays{T, N, A} <: AbstractArray{T, N} end

mutable struct TupleOfCachedArrays{T, N, A} <: AbstractTupleOfCachedArrays{T, N, A}
    u::T#::NTuple{N, <:AbstractVector} # A <: AbstractArray{<: AbstractArray{T, N - 1}}
    ns::A#::AbstractArray#::SVector{N, Int} # sizes of each subarray
end
Adapt.@adapt_structure TupleOfCachedArrays
# TupleOfCachedArrays with an added series for time

### Abstract Interface
struct AllObserved
end

function Base.Array(VA::AbstractTupleOfCachedArrays{
        T,
        N,
        A
}) where {T, N,
        A <: AbstractVector{
            <:AbstractVector,
        }}
    reduce(hcat, VA.u)
end
function Base.Array(VA::AbstractTupleOfCachedArrays{
        T,
        N,
        A
}) where {T, N,
        A <:
        AbstractVector{<:Number}}
    VA.u
end
function Base.Matrix(VA::AbstractTupleOfCachedArrays{
        T,
        N,
        A
}) where {T, N,
        A <: AbstractVector{
            <:AbstractVector,
        }}
    reduce(hcat, VA.u)
end
function Base.Matrix(VA::AbstractTupleOfCachedArrays{
        T,
        N,
        A
}) where {T, N,
        A <:
        AbstractVector{<:Number}}
    Matrix(VA.u)
end
function Base.Vector(VA::AbstractTupleOfCachedArrays{
        T,
        N,
        A
}) where {T, N,
        A <: AbstractVector{
            <:AbstractVector,
        }}
    vec(reduce(hcat, VA.u))
end
function Base.Vector(VA::AbstractTupleOfCachedArrays{
        T,
        N,
        A
}) where {T, N,
        A <:
        AbstractVector{<:Number}}
    VA.u
end
function Base.Array(VA::AbstractTupleOfCachedArrays)
    vecs = vec.(VA.u)
    Array(reshape(reduce(hcat, vecs), size(VA.u[1])..., length(VA.u)))
end
function Base.Array{U}(VA::AbstractTupleOfCachedArrays) where {U}
    vecs = vec.(VA.u)
    Array(reshape(reduce(hcat, vecs), size(VA.u[1])..., length(VA.u)))
end

Base.convert(::Type{AbstractArray}, VA::AbstractTupleOfCachedArrays) = stack(VA.u)

function Adapt.adapt_structure(to, VA::AbstractTupleOfCachedArrays)
    TupleOfCachedArrays(Adapt.adapt.((to,), VA.u))
end

function TupleOfCachedArrays(vec::Tuple, n::Union{Nothing, AbstractVector{<:Int}}=nothing)
    # if n === nothing
    #     n = map(size, vec)
    #     n = SVector{length(vec), Int}(map(i -> i[1], n))
    # else
    #     n = SVector{length(vec), Int}(n)
    # end
    # @assert length(n) == length(vec)
    # n_ = Base.Array(n)
    # @assert all([length(vec[i]) for i in 1:length(n_)] .>= n_)
    T = typeof(vec)#map(eltype, vec)[1]
    N = length(vec)
    if all(x isa Union{<:AbstractArray, <:AbstractTupleOfCachedArrays} for x in vec)
        A = Tuple{Union{typeof.(vec)...}}
    else
        A = typeof(vec)
    end
    TupleOfCachedArrays{T, N, typeof(n)}(vec, n)
end
# function TupleOfCachedArrays(vec::AbstractVector{T}, ::NTuple{N}, n::Union{Nothing, AbstractVector{Int}}=nothing) where {T, N}
#     if n === nothing
#         n = map(size, vec)
#         n = SVector{N, Int}(map(i -> i[1], n))
#     else
#         n = SVector{N, Int}(n)
#     end
#     @assert length(n) == length(vec)
#     @assert all(size(vec[i], 1) >= n[i] for i in 1:length(n))
#     TupleOfCachedArrays{eltype(T), N, typeof(vec)}(n, vec)
# end
# # Assume that the first element is representative of all other elements
# function TupleOfCachedArrays(vec::AbstractVector, n::Union{Nothing, AbstractVector{Int}}=nothing)
#     if n === nothing
#         n = map(size, vec)
#         n = SVector{length(vec), Int}(map(i -> i[1], n))
#     else
#         n = SVector{length(vec), Int}(n)
#     end
#     @assert length(n) == length(vec)
#     @assert all(size(vec[i], 1) >= n[i] for i in 1:length(n))
#     T = eltype(vec[1])
#     N = ndims(vec[1])
#     if all(x isa Union{<:AbstractArray, <:AbstractTupleOfCachedArrays} for x in vec)
#         A = Vector{Union{typeof.(vec)...}}
#     else
#         A = typeof(vec)
#     end
#     TupleOfCachedArrays{T, N + 1, A}(n, vec)
# end
# function TupleOfCachedArrays(vec::AbstractVector{VT}, n::Union{Nothing, AbstractVector{Int}}=nothing) where {T, N, VT <: AbstractArray{T, N}}
#     if n === nothing
#         n = map(size, vec)
#         n = SVector{length(vec), Int}(map(i -> i[1], n))
#     else
#         n = SVector{length(vec), Int}(n)
#     end
#     @assert length(n) == length(vec)
#     @assert all(size(vec[i], 1) >= n[i] for i in 1:length(n))
#     TupleOfCachedArrays{T, N + 1, typeof(vec)}(n, vec)
# end
# # allow multi-dimensional arrays as long as they're linearly indexed. 
# # currently restricted to arrays whose elements are all the same type
# function TupleOfCachedArrays(array::AbstractArray{AT}, n::Union{Nothing, AbstractVector{Int}}=nothing) where {T, N, AT <: AbstractArray{T, N}}
#     @assert IndexStyle(typeof(array)) isa IndexLinear
#     if n === nothing
#         n = map(size, array)
#         n = SVector{length(array), Int}(map(i -> i[1], n))
#     else
#         n = SVector{length(array), Int}(n)
#     end
#     @assert length(n) == length(array)
#     @assert all(size(array[i], 1) >= n[i] for i in 1:length(n))

#     return TupleOfCachedArrays{T, N + 1, typeof(array)}(n, array)
# end

Base.parent(vec::TupleOfCachedArrays) = vec.u

get_discretes(x) = getfield(x, :discretes)

# SymbolicIndexingInterface.is_timeseries(::Type{<:AbstractTupleOfCachedArrays}) = Timeseries()

Base.IndexStyle(A::AbstractTupleOfCachedArrays) = Base.IndexStyle(typeof(A))
Base.IndexStyle(::Type{<:AbstractTupleOfCachedArrays}) = IndexCartesian()

@inline Base.length(VA::AbstractTupleOfCachedArrays) = length(VA.u)
@inline function Base.eachindex(VA::AbstractTupleOfCachedArrays)
    return eachindex(VA.u)
end
@inline function Base.eachindex(
        ::IndexLinear, VA::AbstractTupleOfCachedArrays{T, N, <:AbstractVector{T}}) where {T, N}
    return eachindex(IndexLinear(), VA.u)
end
@inline Base.IteratorSize(::Type{<:AbstractTupleOfCachedArrays}) = Base.HasLength()
@inline Base.first(VA::AbstractTupleOfCachedArrays) = first(VA.u)
@inline Base.last(VA::AbstractTupleOfCachedArrays) = last(VA.u)
function Base.firstindex(VA::AbstractTupleOfCachedArrays{T,N,A}) where {T,N,A}
    N > 1 && Base.depwarn(
        "Linear indexing of `AbstractTupleOfCachedArrays` is deprecated. Change `A[i]` to `A.u[i]` ",
        :firstindex)
    return firstindex(VA.u)
end

function Base.lastindex(VA::AbstractTupleOfCachedArrays{T,N,A}) where {T,N,A}
     N > 1 && Base.depwarn(
        "Linear indexing of `AbstractTupleOfCachedArrays` is deprecated. Change `A[i]` to `A.u[i]` ",
        :lastindex)
    return lastindex(VA.u)
end

Base.getindex(A::AbstractTupleOfCachedArrays, I::Int) = A.u[I]
Base.getindex(A::AbstractTupleOfCachedArrays, I::AbstractArray{Int}) = A.u[I]

@deprecate Base.getindex(VA::AbstractTupleOfCachedArrays{T,N,A}, I::Int) where {T,N,A<:Union{AbstractArray, AbstractTupleOfCachedArrays}} VA.u[I] false

@deprecate Base.getindex(VA::AbstractTupleOfCachedArrays{T,N,A}, I::AbstractArray{Int}) where {T,N,A<:Union{AbstractArray, AbstractTupleOfCachedArrays}} VA.u[I] false

__parameterless_type(T) = Base.typename(T).wrapper

Base.@propagate_inbounds function _getindex(
        A::AbstractTupleOfCachedArrays, ::NotSymbolic, ::Colon, I::Int)
    A.u[I]
end

Base.@propagate_inbounds function _getindex(A::AbstractTupleOfCachedArrays, ::NotSymbolic,
        I::Union{Int, AbstractArray{Int}, AbstractArray{Bool}, Colon}...)
    if last(I) isa Int
        A.u[last(I)][Base.front(I)...]
    else
        stack(getindex.(A.u[last(I)], tuple.(Base.front(I))...))
    end
end
Base.@propagate_inbounds function _getindex(
        VA::AbstractTupleOfCachedArrays, ::NotSymbolic, ii::CartesianIndex)
    ti = Tuple(ii)
    i = last(ti)
    jj = CartesianIndex(Base.front(ti))
    return VA.u[i][jj]
end

Base.@propagate_inbounds function _getindex(
        A::AbstractTupleOfCachedArrays, ::NotSymbolic, ::Colon,
        I::Union{AbstractArray{Int}, AbstractArray{Bool}})
    TupleOfCachedArrays(A.u[I])
end

struct ParameterIndexingError <: Exception
    sym::Any
end

function Base.showerror(io::IO, pie::ParameterIndexingError)
    print(io,
        "Indexing with parameters is deprecated. Use `getp(A, $(pie.sym))` for parameter indexing.")
end

# Symbolic Indexing Methods
Base.@propagate_inbounds function Base.getindex(A::AbstractTupleOfCachedArrays, _arg, args...)
    symtype = symbolic_type(_arg)
    elsymtype = symbolic_type(eltype(_arg))

    if symtype == NotSymbolic() && elsymtype == NotSymbolic()
        if _arg isa Union{Tuple, AbstractArray} &&
           any(x -> symbolic_type(x) != NotSymbolic(), _arg)
            _getindex(A, symtype, elsymtype, _arg, args...)
        else
            _getindex(A, symtype, _arg, args...)
        end
    else
        _getindex(A, symtype, elsymtype, _arg, args...)
    end
end

# Base.@propagate_inbounds function Base.getindex(
#         A::Adjoint{T, <:AbstractTupleOfCachedArrays}, idxs...) where {T}
#     return getindex(A.parent, reverse(to_indices(A, idxs))...)
# end

Base.@propagate_inbounds function Base.setindex!(VA::AbstractTupleOfCachedArrays{T, N}, v,
        ::Colon, I::Int) where {T, N}
    VA.u[I] = v
end

Base.@propagate_inbounds Base.setindex!(VA::AbstractTupleOfCachedArrays, v, I::Int) = Base.setindex!(VA.u, v, I)
@deprecate Base.setindex!(VA::AbstractTupleOfCachedArrays{T,N,A}, v, I::Int) where {T,N,A<:Union{AbstractArray, AbstractTupleOfCachedArrays}} Base.setindex!(VA.u, v, I) false

Base.@propagate_inbounds function Base.setindex!(VA::AbstractTupleOfCachedArrays{T, N}, v,
        ::Colon, I::Colon) where {T, N}
    VA.u[I] = v
end

Base.@propagate_inbounds Base.setindex!(VA::AbstractTupleOfCachedArrays, v, I::Colon) = Base.setindex!(VA.u, v, I)
@deprecate Base.setindex!(VA::AbstractTupleOfCachedArrays{T,N,A}, v, I::Colon)  where {T,N,A<:Union{AbstractArray, AbstractTupleOfCachedArrays}} Base.setindex!(
    VA.u, v, I) false

Base.@propagate_inbounds function Base.setindex!(VA::AbstractTupleOfCachedArrays{T, N}, v,
        ::Colon, I::AbstractArray{Int}) where {T, N}
    VA.u[I] = v
end

Base.@propagate_inbounds Base.setindex!(VA::AbstractTupleOfCachedArrays, v, I::AbstractArray{Int}) = Base.setindex!(VA.u, v, I)
@deprecate Base.setindex!(VA::AbstractTupleOfCachedArrays{T,N,A}, v, I::AbstractArray{Int}) where {T,N,A<:Union{AbstractArray, AbstractTupleOfCachedArrays}} Base.setindex!(
    VA, v, :, I) false

Base.@propagate_inbounds function Base.setindex!(
        VA::AbstractTupleOfCachedArrays{T, N}, v, i::Int,
        ::Colon) where {T, N}
    for j in 1:length(VA.u)
        VA.u[j][i] = v[j]
    end
    return v
end
Base.@propagate_inbounds function Base.setindex!(VA::AbstractTupleOfCachedArrays{T, N}, x,
        ii::CartesianIndex) where {T, N}
    ti = Tuple(ii)
    i = last(ti)
    jj = CartesianIndex(Base.front(ti))
    return VA.u[i][jj] = x
end

Base.@propagate_inbounds function Base.setindex!(VA::AbstractTupleOfCachedArrays{T, N},
        x,
        idxs::Union{Int, Colon, CartesianIndex, AbstractArray{Int}, AbstractArray{Bool}}...) where {
        T, N}
    v = view(VA, idxs...)
    # error message copied from Base by running `ones(3, 3, 3)[:, 2, :] = 2`
    if length(v) != length(x)
        throw(ArgumentError("indexed assignment with a single value to possibly many locations is not supported; perhaps use broadcasting `.=` instead?"))
    end
    for (i, j) in zip(eachindex(v), eachindex(x))
        v[i] = x[j]
    end
    return x
end

# Interface for the two-dimensional indexing, a more standard AbstractArray interface
@inline Base.size(VA::AbstractTupleOfCachedArrays) = (size(VA.u[1])..., length(VA.u))
@inline Base.size(VA::AbstractTupleOfCachedArrays, i) = size(VA)[i]
# @inline Base.size(A::Adjoint{T, <:AbstractTupleOfCachedArrays}) where {T} = reverse(size(A.parent))
# @inline Base.size(A::Adjoint{T, <:AbstractTupleOfCachedArrays}, i) where {T} = size(A)[i]
Base.axes(VA::AbstractTupleOfCachedArrays) = Base.OneTo.(size(VA))
Base.axes(VA::AbstractTupleOfCachedArrays, d::Int) = Base.OneTo(size(VA)[d])

Base.@propagate_inbounds function Base.setindex!(VA::AbstractTupleOfCachedArrays{T, N}, v,
        I::Int...) where {T, N}
    VA.u[I[end]][Base.front(I)...] = v
end

function Base.:(==)(A::AbstractTupleOfCachedArrays, B::AbstractTupleOfCachedArrays)
    return A.u == B.u
end
function Base.:(==)(A::AbstractTupleOfCachedArrays, B::AbstractArray)
    return A.u == B
end
Base.:(==)(A::AbstractArray, B::AbstractTupleOfCachedArrays) = B == A

# The iterator will be over the subarrays of the container, not the individual elements
# unlike an true AbstractArray
function Base.iterate(VA::AbstractTupleOfCachedArrays, state = 1)
    state >= length(VA.u) + 1 ? nothing : (VA[:, state], state + 1)
end

# Growing the array simply adds to the container vector
function _copyfield(VA, fname)
    if fname == :u
        copy(VA.u)
    elseif fname == :t
        copy(VA.t)
    else
        getfield(VA, fname)
    end
end
function Base.copy(VA::AbstractTupleOfCachedArrays)
    typeof(VA)((_copyfield(VA, fname) for fname in fieldnames(typeof(VA)))...)
end

function Base.zero(VA::AbstractTupleOfCachedArrays)
    val = copy(VA)
    val.u .= zero.(VA.u)
    return val
end

Base.sizehint!(VA::AbstractTupleOfCachedArrays{T, N}, i) where {T, N} = sizehint!(VA.u, i)

Base.reverse!(VA::AbstractTupleOfCachedArrays) = reverse!(VA.u)
Base.reverse(VA::AbstractTupleOfCachedArrays) = TupleOfCachedArrays(reverse(VA.u))

function Base.resize!(VA::AbstractTupleOfCachedArrays, i::Integer)
    if Base.hasproperty(VA, :sys) && VA.sys !== nothing
        error("resize! is not allowed on AbstractTupleOfCachedArrays with a sys")
    end
    Base.resize!(VA.u, i)
    if Base.hasproperty(VA, :t) && VA.t !== nothing
        Base.resize!(VA.t, i)
    end
end

function Base.pointer(VA::AbstractTupleOfCachedArrays)
    Base.pointer(VA.u)
end

function Base.push!(VA::AbstractTupleOfCachedArrays{T, N}, new_item::AbstractArray) where {T, N}
    push!(VA.u, new_item)
end

function Base.append!(VA::AbstractTupleOfCachedArrays{T, N},
        new_item::AbstractTupleOfCachedArrays{T, N}) where {T, N}
    for item in copy(new_item)
        push!(VA, item)
    end
    return VA
end

function Base.stack(VA::AbstractTupleOfCachedArrays; dims = :)
    stack(stack.(VA.u); dims)
end

# AbstractArray methods
function Base.view(A::AbstractTupleOfCachedArrays{T, N, <:AbstractVector{T}},
        I::Vararg{Any, M}) where {T, N, M}
    @inline
    if length(I) == 1
        J = map(i -> Base.unalias(A, i), to_indices(A, I))
    elseif length(I) == 2 && (I[1] == Colon() || I[1] == 1)
        J = map(i -> Base.unalias(A, i), to_indices(A, Base.tail(I)))
    end
    @boundscheck checkbounds(A, J...)
    SubArray(A, J)
end
function Base.view(A::AbstractTupleOfCachedArrays, I::Vararg{Any, M}) where {M}
    @inline
    J = map(i -> Base.unalias(A, i), to_indices(A, I))
    @boundscheck checkbounds(A, J...)
    SubArray(A, J)
end
function Base.SubArray(parent::AbstractTupleOfCachedArrays, indices::Tuple)
    @inline
    SubArray(IndexStyle(Base.viewindexing(indices), IndexStyle(parent)), parent,
        Base.ensure_indexable(indices), Base.index_dimsum(indices...))
end
Base.isassigned(VA::AbstractTupleOfCachedArrays, idxs...) = checkbounds(Bool, VA, idxs...)
# function Base.check_parent_index_match(
#         ::RecursiveArrayTools.AbstractTupleOfCachedArrays{T, N}, ::NTuple{N, Bool}) where {T, N}
#     nothing
# end
Base.ndims(::AbstractTupleOfCachedArrays{T, N}) where {T, N} = N
Base.ndims(::Type{<:AbstractTupleOfCachedArrays{T, N}}) where {T, N} = N

function Base.checkbounds(
        ::Type{Bool}, VA::AbstractTupleOfCachedArrays{T, N, <:AbstractVector{T}},
        idxs...) where {T, N}
    if length(idxs) == 2 && (idxs[1] == Colon() || idxs[1] == 1)
        return checkbounds(Bool, VA.u, idxs[2])
    end
    return checkbounds(Bool, VA.u, idxs...)
end
function Base.checkbounds(::Type{Bool}, VA::AbstractTupleOfCachedArrays, idx...)
    checkbounds(Bool, VA.u, last(idx)) || return false
    for i in last(idx)
        checkbounds(Bool, VA.u[i], Base.front(idx)...) || return false
    end
    return true
end
function Base.checkbounds(VA::AbstractTupleOfCachedArrays, idx...)
    checkbounds(Bool, VA, idx...) || throw(BoundsError(VA, idx))
end
function Base.copyto!(dest::TupleOfCachedArrays, src::TupleOfCachedArrays)
    @inbounds for i in 1:length(dest.u)
        @views dest.u[i][1:dest.ns[i]] .= src.u[i][1:src.ns[i]]
    end
    return dest
end
function Base.copyto!(
        dest::AbstractTupleOfCachedArrays{T, N}, src::AbstractArray{T2, N}) where {T, T2, N}
    for (i, slice) in zip(eachindex(dest.u), eachslice(src, dims = ndims(src)))
        if ArrayInterface.ismutable(dest.u[i]) || dest.u[i] isa AbstractTupleOfCachedArrays
            copyto!(dest.u[i], slice)
        else
            dest.u[i] = StaticArraysCore.similar_type(dest.u[i])(slice)
        end
    end
    dest
end
function Base.copyto!(dest::AbstractTupleOfCachedArrays{T, N, <:AbstractVector{T}},
        src::AbstractVector{T2}) where {T, T2, N}
    copyto!(dest.u, src)
    dest
end
# Required for broadcasted setindex! when slicing across subarrays
# E.g. if `va = TupleOfCachedArrays([rand(3, 3) for i in 1:5])`
# Need this method for `va[2, :, :] .= 3.0`
Base.@propagate_inbounds function Base.maybeview(A::AbstractTupleOfCachedArrays, I...)
    return view(A, I...)
end

# Operations
function Base.isapprox(A::AbstractTupleOfCachedArrays,
        B::Union{AbstractTupleOfCachedArrays, AbstractArray};
        kwargs...)
    return all(isapprox.(A, B; kwargs...))
end

function Base.isapprox(A::AbstractArray, B::AbstractTupleOfCachedArrays; kwargs...)
    return all(isapprox.(A, B; kwargs...))
end

for op in [:(Base.:-), :(Base.:+)]
    @eval function ($op)(A::AbstractTupleOfCachedArrays, B::AbstractTupleOfCachedArrays)
        ($op).(A, B)
    end
    @eval Base.@propagate_inbounds function ($op)(A::AbstractTupleOfCachedArrays,
            B::AbstractArray)
        @boundscheck length(A) == length(B)
        TupleOfCachedArrays([($op).(a, b) for (a, b) in zip(A, B)])
    end
    @eval Base.@propagate_inbounds function ($op)(
            A::AbstractArray, B::AbstractTupleOfCachedArrays)
        @boundscheck length(A) == length(B)
        TupleOfCachedArrays([($op).(a, b) for (a, b) in zip(A, B)])
    end
end

for op in [:(Base.:/), :(Base.:\), :(Base.:*)]
    if op !== :(Base.:/)
        @eval ($op)(A::Number, B::AbstractTupleOfCachedArrays) = ($op).(A, B)
    end
    if op !== :(Base.:\)
        @eval ($op)(A::AbstractTupleOfCachedArrays, B::Number) = ($op).(A, B)
    end
end

function Base.CartesianIndices(VA::AbstractTupleOfCachedArrays)
    if !allequal(size.(VA.u))
        error("CartesianIndices only valid for non-ragged arrays")
    end
    return CartesianIndices((size(VA.u[1])..., length(VA.u)))
end

# Tools for creating similar objects
Base.eltype(::Type{<:AbstractTupleOfCachedArrays{T}}) where {T} = T

@inline function Base.similar(VA::AbstractTupleOfCachedArrays, args...)
    if args[end] isa Type
        return Base.similar(eltype(VA)[], args..., size(VA))
    else
        return Base.similar(eltype(VA)[], args...)
    end
end

function Base.similar(vec::TupleOfCachedArrays{
        T, N, AT}) where {T, N, AT <: AbstractArray{<:AbstractArray{T}}}
    return TupleOfCachedArrays(similar.(Base.parent(vec)))
end

# function Base.similar(vec::TupleOfCachedArrays{
#         T, N, AT}) where {T, N, AT <: AbstractArray{<:StaticArraysCore.StaticVecOrMat{T}}}
#     # this avoids behavior such as similar(SVector) returning an MVector
#     return TupleOfCachedArrays(similar(Base.parent(vec)))
# end

@inline function Base.similar(VA::TupleOfCachedArrays, ::Type{T} = eltype(VA)) where {T}
    TupleOfCachedArrays(similar.(VA.u, T))
end

@inline function Base.similar(VA::TupleOfCachedArrays, dims::N) where {N <: Number}
    l = length(VA)
    if dims <= l
        TupleOfCachedArrays(similar.(VA.u[1:dims]))
    else
        TupleOfCachedArrays([similar.(VA.u); [similar(VA.u[end]) for _ in (l + 1):dims]])
    end
end

# fill!
function Base.fill!(VA::AbstractTupleOfCachedArrays, x)
    for i in 1:length(VA.u)
        if VA[:, i] isa AbstractArray
            fill!(VA[:, i], x)
        else
            VA[:, i] = x
        end
    end
    return VA
end

Base.reshape(A::AbstractTupleOfCachedArrays, dims...) = Base.reshape(Array(A), dims...)

# Need this for ODE_DEFAULT_UNSTABLE_CHECK from DiffEqBase to work properly
@inline Base.any(f, VA::AbstractTupleOfCachedArrays) = any(any(f, u) for u in VA.u)
@inline Base.all(f, VA::AbstractTupleOfCachedArrays) = all(all(f, u) for u in VA.u)

# conversion tools
vecarr_to_vectors(VA::AbstractTupleOfCachedArrays) = [VA[i, :] for i in eachindex(VA.u[1])]
Base.vec(VA::AbstractTupleOfCachedArrays) = vec(convert(Array, VA)) # Allocates
# stack non-ragged arrays to convert them
function Base.convert(::Type{Array}, VA::AbstractTupleOfCachedArrays)
    if !allequal(size.(VA.u))
        error("Can only convert non-ragged TupleOfCachedArrays to Array")
    end
    return Array(VA)
end

# statistics
@inline Base.sum(VA::AbstractTupleOfCachedArrays; kwargs...) = sum(identity, VA; kwargs...)
@inline function Base.sum(f, VA::AbstractTupleOfCachedArrays; kwargs...)
    mapreduce(f, Base.add_sum, VA; kwargs...)
end
@inline Base.prod(VA::AbstractTupleOfCachedArrays; kwargs...) = prod(identity, VA; kwargs...)
@inline function Base.prod(f, VA::AbstractTupleOfCachedArrays; kwargs...)
    mapreduce(f, Base.mul_prod, VA; kwargs...)
end

# @inline Statistics.mean(VA::AbstractTupleOfCachedArrays; kwargs...) = mean(Array(VA); kwargs...)
# @inline function Statistics.median(VA::AbstractTupleOfCachedArrays; kwargs...)
#     median(Array(VA); kwargs...)
# end
# @inline Statistics.std(VA::AbstractTupleOfCachedArrays; kwargs...) = std(Array(VA); kwargs...)
# @inline Statistics.var(VA::AbstractTupleOfCachedArrays; kwargs...) = var(Array(VA); kwargs...)
# @inline Statistics.cov(VA::AbstractTupleOfCachedArrays; kwargs...) = cov(Array(VA); kwargs...)
# @inline Statistics.cor(VA::AbstractTupleOfCachedArrays; kwargs...) = cor(Array(VA); kwargs...)
# @inline Base.adjoint(VA::AbstractTupleOfCachedArrays) = Adjoint(VA)

# linear algebra
# ArrayInterface.issingular(va::AbstractTupleOfCachedArrays) = ArrayInterface.issingular(Matrix(va))

# make it show just like its data
function Base.show(io::IO, m::MIME"text/plain", x::AbstractTupleOfCachedArrays)
    (println(io, summary(x), ':'); show(io, m, x.u))
end
function Base.summary(A::AbstractTupleOfCachedArrays{T, N}) where {T, N}
    string("TupleOfCachedArrays{", T, ",", N, "}")
end

# plot recipes
# @recipe function f(VA::AbstractTupleOfCachedArrays)
#     convert(Array, VA)
# end

Base.map(f, A::AbstractTupleOfCachedArrays) = map(f, A.u)

function Base.mapreduce(f, op, A::AbstractTupleOfCachedArrays; kwargs...)
    mapreduce(f, op, view(A, ntuple(_ -> :, ndims(A))...); kwargs...)
end
function Base.mapreduce(
        f, op, A::AbstractTupleOfCachedArrays{T, 1, <:AbstractVector{T}}; kwargs...) where {T}
    mapreduce(f, op, A.u; kwargs...)
end

## broadcasting

struct TupleOfCachedArraysStyle{N} <: Broadcast.AbstractArrayStyle{N} end # N is only used when voa sees other abstract arrays
TupleOfCachedArraysStyle{N}(::Val{N}) where {N} = TupleOfCachedArraysStyle{N}()

# The order is important here. We want to override Base.Broadcast.DefaultArrayStyle to return another Base.Broadcast.DefaultArrayStyle.
Broadcast.BroadcastStyle(a::TupleOfCachedArraysStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = a
function Broadcast.BroadcastStyle(::TupleOfCachedArraysStyle{N},
        a::Base.Broadcast.DefaultArrayStyle{M}) where {M, N}
    Base.Broadcast.DefaultArrayStyle(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(::TupleOfCachedArraysStyle{N},
        a::Base.Broadcast.AbstractArrayStyle{M}) where {M, N}
    typeof(a)(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(::TupleOfCachedArraysStyle{M},
        ::TupleOfCachedArraysStyle{N}) where {M, N}
    TupleOfCachedArraysStyle(Val(max(M, N)))
end
function Broadcast.BroadcastStyle(::Type{<:AbstractTupleOfCachedArrays{T, N}}) where {T, N}
    TupleOfCachedArraysStyle{N}()
end
# make TupleOfCachedArrayss broadcastable so they aren't collected
Broadcast.broadcastable(x::AbstractTupleOfCachedArrays) = x

@inline function Base.copy(bc::Broadcast.Broadcasted{<:TupleOfCachedArraysStyle})
    bc = Broadcast.flatten(bc)
    parent = find_VoA_parent(bc.args)

    u = if parent isa AbstractVector
        # this is the default behavior in v3.15.0
        N = narrays(bc)
        map(1:N) do i
            copy(unpack_voa(bc, i))
        end
    else # if parent isa AbstractArray            
        map(enumerate(Iterators.product(axes(parent)...))) do (i, _)
            copy(unpack_voa(bc, i))
        end
    end
    TupleOfCachedArrays(rewrap(parent, u))
end

rewrap(::Array, u) = u
rewrap(parent, u) = convert(typeof(parent), u)

for (type, N_expr) in [
        (Broadcast.Broadcasted{<:TupleOfCachedArraysStyle}, :(narrays(bc))),
        (Broadcast.Broadcasted{<:Broadcast.DefaultArrayStyle}, :(length(dest.u)))
    ]

    @eval @inline function Base.copyto!(dest::AbstractTupleOfCachedArrays,
            bc::$type)
        bc = Broadcast.flatten(bc)
        N = $N_expr
        @inbounds for i in 1:N
            dest_ = @views dest.u[i][1:dest.ns[i]]
            copyto!(dest_, unpack_voa(bc, i))
        end
        dest
    end
end

## broadcasting utils

"""
    narrays(A...)

Retrieve number of arrays in the AbstractTupleOfCachedArrayss of a broadcast.
"""
narrays(A) = 0
narrays(A::AbstractTupleOfCachedArrays) = length(A.u)
narrays(bc::Broadcast.Broadcasted) = _narrays(bc.args)
narrays(A, Bs...) = common_length(narrays(A), _narrays(Bs))

function common_length(a, b)
    a == 0 ? b :
    (b == 0 ? a :
     (a == b ? a :
      throw(DimensionMismatch("number of arrays must be equal"))))
end

_narrays(args::AbstractTupleOfCachedArrays) = length(args.u)
@inline _narrays(args::Tuple) = common_length(narrays(args[1]), _narrays(Base.tail(args)))
_narrays(args::Tuple{Any}) = _narrays(args[1])
_narrays(::Any) = 0

# drop axes because it is easier to recompute
@inline function unpack_voa(bc::Broadcast.Broadcasted{Style}, i) where {Style}
    Broadcast.Broadcasted{Style}(bc.f, unpack_args_voa(i, bc.args))
end
@inline function unpack_voa(bc::Broadcast.Broadcasted{<:TupleOfCachedArraysStyle}, i)
    Broadcast.Broadcasted(bc.f, unpack_args_voa(i, bc.args))
end
unpack_voa(x, ::Any) = x
unpack_voa(x::AbstractTupleOfCachedArrays, i) = @views x.u[i][1:x.ns[i]]
# function unpack_voa(x::AbstractArray{T, N}, i) where {T, N}
#     @view x[ntuple(x -> Colon(), N - 1)..., i]
# end

@inline function unpack_args_voa(i, args::Tuple)
    v = (unpack_voa(args[1], i), unpack_args_voa(i, Base.tail(args))...)
    return v
end
@inline function unpack_args_voa(i, args::NTuple{N, <:AbstractArray}) where {N}
    ntuple(k -> unpack_voa(args[k], i), N)
end
unpack_args_voa(i, args::Tuple{Any}) = (unpack_voa(args[1], i),)
unpack_args_voa(::Any, args::Tuple{}) = ()