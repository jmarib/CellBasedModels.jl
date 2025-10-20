######################################################################################################
# AGENT STRUCTURE
######################################################################################################

struct AgentPoint{D, P, T} <: AbstractAgent where {D, P, T}

    propertiesAgent::NamedTuple{}    # Dictionary to hold agent properties

end

function AgentPoint(
    dims::Int;
    propertiesAgent::NamedTuple = (;)
)
    if dims < 0 || dims > 3
        error("dims must be between 0 and 3. Found $dims")
    end
    defaultParameters = (
        x = dims >= 1 ? Parameter(AbstractFloat, description="Position in x (protected parameter)", dimensions=:L, _scope=:agent) : nothing,
        y = dims >= 2 ? Parameter(AbstractFloat, description="Position in y (protected parameter)", dimensions=:L, _scope=:agent) : nothing,
        z = dims >= 3 ? Parameter(AbstractFloat, description="Position in z (protected parameter)", dimensions=:L, _scope=:agent) : nothing,
        id = Parameter(Integer, description="Unique identifier (protected parameter)", _scope=:agent),
        # _N = dims >= 0 ? Parameter(Int, description="Number of agents (protected parameter)") : nothing,
        # _NCache = dims >= 0 ? Parameter(Int, description="Maximum number of preallocated agents (protected parameter)") : nothing,
        # _NNew = dims >= 0 ? Parameter(Int, description="Number of new agents added in the current step (protected parameter)") : nothing,
        # _idMax = dims >= 0 ? Parameter(Int, description="Maximum unique identifier assigned (protected parameter)") : nothing,
        # _NFlag = dims >= 0 ? Parameter(Bool, description="Flag indicating if the number of agents exceeded the preallocated maximum (protected parameter)") : nothing,
    )

    # Remove keys with value `nothing`
    defaultParameters = NamedTuple{Tuple(k for (k,v) in pairs(defaultParameters) if v !== nothing)}(
        (v for (k,v) in pairs(defaultParameters) if v !== nothing)
    )
    propertiesAgentNew = parameterConvert(propertiesAgent, scope=:agent)
    for (k, v) in pairs(propertiesAgentNew)
        if haskey(defaultParameters, k)
            error("Parameter $k is already defined and cannot be used as agent property.")
        elseif startswith(string(k), "_")
            error("Parameter $k is protected and cannot be used as agent property.")
        end
    end
    propertiesAgentNew = merge(defaultParameters, propertiesAgentNew)

    t = (
        n = dt for (k, dt) in pairs(propertiesAgentNew)
    )

    return AgentPoint{dims, typeof(propertiesAgentNew).parameters[1], typeof(propertiesAgentNew).parameters[2]}(
        propertiesAgentNew
    )
end

Base.ndims(x::AgentPoint{D, P, T}) where {D, P, T} = D

function Base.show(io::IO, x::AgentPoint)
    println(io, "AgentPoint with dimensions $(x._dims): \n")
    println(io, @sprintf("\t%-15s %-15s %-15s %-20s %-s", "Name", "DataType", "Dimensions", "Default_Value", "Description"))
    println(io, "\t" * repeat("-", 85))
    for (name, par) in pairs(x.propertiesAgent)
        println(io, @sprintf("\t%-15s %-15s %-15s %-20s %-s", 
            name, 
            string(par.dataType), 
            par.dimensions == nothing ? "" : string(par.dimensions),
            par.defaultValue == nothing ? "" : string(par.defaultValue), 
            par.description))
    end
    println(io)
end

function Base.show(io::IO, ::Type{AgentPoint{D, P, T}}) where {D, P, T}
    print(io, "AgentPoint{dims=", D, ", properties=(")
    for (i, (n, t)) in enumerate(zip(P, P.parameters))
        i > 1 && print(io, ", ")
        print(io, n, "::", t.parameters[1])
    end
    print(io, ")}")
end

######################################################################################################
# COMMUNITY AGENT POINT
######################################################################################################
# struct CommunityPoint{B, D, P, T, NP, N, NC} <: AbstractCommunity{T, N}
struct CommunityPointMeta{S,G}
    _N::S
    _NCache::S
    _NNew::S
    _idMax::S
    _NFlag::G
end
Adapt.@adapt_structure CommunityPointMeta

function CommunityPointMeta(
        _N,
        _NCache,
        _NNew,
        _idMax,
        _NFlag,
    )

    S = typeof(_N)
    G = typeof(_NFlag)

    CommunityPointMeta{S, G}(
        _N,
        _NCache,
        _NNew,
        _idMax,
        _NFlag,
    )
end
struct CommunityPoint{B, D, P, T, NP, N, NC} <: AbstractCommunity where {B, D, P, T, NP, N, NC}
    _propertiesAgent::NamedTuple{P, T}
    _meta::B
    _propertiesCopy::SVector{NP, Bool}
    _n::Int
    _ncache::Int
end
Adapt.@adapt_structure CommunityPoint

function CommunityPoint(
        agent::AgentPoint{D, P, T2},
        N::Integer=1,
        NCache::Union{Nothing, Integer}=nothing
    ) where {D, P, T2}

    if N < 1
        error("N must be greater than 1. Found N=$N")
    end
    if NCache === nothing
        NCache = N
    elseif NCache < N
        error("NCache must be greater than or equal to N. Found NCache=$NCache and N=$N")
    end

    properties = NamedTuple{keys(agent.propertiesAgent)}(zeros(dtype(dt, isbits=true), NCache) for dt in values(agent.propertiesAgent))
    properties.id[1:N] .= collect(1:N)

    NP = length(P)

    _N = SizedVector{1, Int}([N])
    _NCache = SizedVector{1, Int}([NCache])
    _NNew = SizedVector{1, Int}([N])
    _idMax = SizedVector{1, Int}([N])
    _NFlag = SizedVector{1, Bool}([false])

    _meta = CommunityPointMeta(
        _N,
        _NCache,
        _NNew,
        _idMax,
        _NFlag,
    )

    _propertiesCopy = SizedVector{NP, Bool}([false for _ in 1:NP])    # Dictionary to hold agent properties for copying

    return CommunityPoint{typeof(_meta), D, P, typeof(values(properties)), NP, N, NCache}(
        properties,
        _meta,
        _propertiesCopy,
        N,
        NCache,
    )
end

function CommunityPoint(
        _propertiesAgent,
        _meta,
        _propertiesCopy,
        _n,
        _ncache,
    )

    B = typeof(_meta)
    D = length(filter(k -> k in (:x, :y, :z), keys(_propertiesAgent)))
    P = keys(_propertiesAgent)
    T = typeof(values(_propertiesAgent))
    NP = length(P)
    N = _n
    NC = _ncache

    CommunityPoint{B, D, P, T, NP, N, NC}(
        _propertiesAgent,
        _meta,
        _propertiesCopy,
        N,
        NC
    )
end

Base.length(community::CommunityPoint{B, D, P, T, NP, N, NC}) where {B, D, P, T, NP, N, NC} = N
@inline Base.size(community::CommunityPoint{B, D, P, T, NP, N, NC}) where {B, D, P, T, NP, N, NC} = (NP, N)

# Base.axes(community::CommunityPoint{B, D, P, T, NP, N, NC}) where {B, D, P, T, NP, N, NC} = (tuple(i for (i, dt) in enumerate(T) if eltype(dt) <: AbstractFloat), 1:N)
Base.ndims(community::CommunityPoint) = 2
Base.ndims(community::Type{<:CommunityPoint}) = 2
Base.IndexStyle(CommunityPoint::CommunityPoint) = Base.IndexStyle(typeof(CommunityPoint))
Base.IndexStyle(::Type{<:CommunityPoint}) = IndexCartesian()
Base.CartesianIndices(community::CommunityPoint{B, D, P, T, NP, N, NC}) where {B, D, P, T, NP, N, NC} =
    CartesianIndices((NP, N))
Base.getindex(cp::CommunityPoint, i::Symbol, j::Int) = cp._propertiesAgent[i][j]
Base.getindex(cp::CommunityPoint, i::Int, j::Int) = cp._propertiesAgent[i][j]
Base.getindex(cp::CommunityPoint, i::Symbol, :) = cp._propertiesAgent[i]
Base.getindex(cp::CommunityPoint, i::Int, :) = cp._propertiesAgent[i]
Base.getindex(cp::CommunityPoint{B, D, P, T, NP, N, NC}, :, j::Int) where {B, D, P, T, NP, N, NC} = NamedTuple{P}(
    (cp._propertiesAgent[k][j] for k in keys(cp._propertiesAgent))
)
Base.setindex!(cp::CommunityPoint, value, i::Symbol, j::Int) = (cp._propertiesAgent[i][j] = value)
Base.setindex!(cp::CommunityPoint, value, i::Int, j::Int) = (cp._propertiesAgent[i][j] = value)
Base.setindex!(cp::CommunityPoint, value, index::CartesianIndex{2}) = (cp._propertiesAgent[index[1]][index[2]] = value)
# Base.keys(cp::CommunityPoint) = keys(cp._propertiesAgent)

function Base.getindex(community::CommunityPoint, i::Integer)
    community._propertiesAgent[i]
end

function Base.iterate(community::CommunityPoint, state = 1)
    state >= length(community._propertiesAgent) + 1 ? nothing : (community[state], state + 1)
end

@generated function Base.getproperty(VA::CommunityPoint{B, D, P, T, NP, N, NC}, s::Symbol) where {B, D, P, T, NP, N, NC}
    names = P
    # build a clause for each fieldname in T
    casesmeta = [
        :(s === $(QuoteNode(name)) && return getfield(getfield(VA, :_meta), $(QuoteNode(name))))
        for name in fieldnames(CommunityPointMeta)
    ]
    cases = [
        :(s === $(QuoteNode(name)) && return @views getfield(getfield(VA, :_propertiesAgent), $(QuoteNode(name)))[1:N])
        for name in names
    ]

    quote
        if s === :_propertiesAgent; return getfield(VA, :_propertiesAgent); end
        if s === :_meta; return getfield(VA, :_meta); end
        if s === :_propertiesCopy; return getfield(VA, :_propertiesCopy); end
        if s === :_n; return getfield(VA, :_n); end
        if s === :_ncache; return getfield(VA, :_ncache); end
        $(casesmeta...)
        $(cases...)
        error("Unknown property: $s for $VA")
    end
end

## broadcasting

struct CommunityPointStyle{N} <: Broadcast.AbstractArrayStyle{N} end

# Allow constructing the style from Val or Int
CommunityPointStyle(::Val{N}) where {N} = CommunityPointStyle{N}()
CommunityPointStyle(N::Int) = CommunityPointStyle{N}()

# Your CommunityPoint acts like a 2D array
Base.BroadcastStyle(::Type{<:CommunityPoint}) = CommunityPointStyle{2}()

# Combine styles safely
Base.Broadcast.result_style(::CommunityPointStyle{M}) where {M} =
    CommunityPointStyle{M}()
Base.Broadcast.result_style(::CommunityPointStyle{M}, ::CommunityPointStyle{N}) where {M,N} =
    CommunityPointStyle{max(M,N)}()
Base.Broadcast.result_style(::CommunityPointStyle{M}, ::Base.Broadcast.AbstractArrayStyle{N}) where {M,N} =
    CommunityPointStyle{max(M,N)}()
Base.Broadcast.result_style(::Base.Broadcast.AbstractArrayStyle{M}, ::CommunityPointStyle{N}) where {M,N} =
    CommunityPointStyle{max(M,N)}()

# Base.length(::CellBasedModels.CommunityPointStyle{N}) where {N} = N

Broadcast.broadcastable(x::CommunityPoint) = x

## Copyto
@eval @inline function Base.copy(dest::CommunityPoint{B, D, P, T, NP, N, NC}) where {B, D, P, T, NP, N, NC}

    deepcopy(dest)

end

@eval @inline function Base.copy(dest::CommunityPoint{B, D, P, T, NP, N, NC}, subset::NTuple{P2, Symbol}) where {B, D, P, T, NP, N, NC, P2}

    for s in subset
        if !(s in P)
            error("Property $s not found in CommunityPoint.")
        end
    end

    CommunityPoint{B, D, P, T, NP, N, NC}(
        NamedTuple{P}(
            i in names(subset) ? copy(dest._propertiesAgent[i]) : dest._propertiesAgent[i] for i in P
        ),
        dest._meta,
        SizedVector{NP, Bool}([i in names(subset) ? true : false for i in P])    # Dictionary to hold agent properties for copying
    )

end

@eval @inline function Base.copyto!(dest::CommunityPoint{B, D, P, T, NP, N, NC},
        bc::CommunityPoint{B, D, P, T, NP, N, NC2}) where {B, D, P, T, NP, N, NC, NC2}
    @inbounds for i in 1:NP
        if bc._propertiesCopy[i]
            @views dest._propertiesAgent[i][1:N] .= bc._propertiesAgent[i][1:N]
        end
    end
    dest
end

for type in [
        Broadcast.Broadcasted{<:CommunityPoint},
        Broadcast.Broadcasted{<:CommunityPointStyle},
    ]

    @eval @inline function Base.copyto!(dest::CommunityPoint{B, D, P, T, NP, N, NC},
            bc::$type) where {B, D, P, T, NP, N, NC}
        bc = Broadcast.flatten(bc)
        @inbounds for i in 1:NP
            if dest._propertiesCopy[i]
                dest_ = @views dest._propertiesAgent[i][1:N]
                copyto!(dest_, unpack_voa(bc, i))
            end
        end
        dest
    end
end

# # drop axes because it is easier to recompute
@inline function unpack_voa(bc::Broadcast.Broadcasted{<:CommunityPoint}, i)
    Broadcast.Broadcasted(bc.f, unpack_args_voa(i, bc.args))
end

function unpack_voa(x::CommunityPoint{B, D, P, T, NP, N, NC}, i) where {B, D, P, T, NP, N, NC}
    @views x._propertiesAgent[i][1:N]
end

# Integrator
function Base.eltype(::Type{<:CommunityPoint{B, D, P, T, NP, N, NC}}) where {B, D, P, T, NP, N, NC}
    for i in T.parameters
        if i <: AbstractFloat
            return eltype(i)
        end
    end
    return Float64
end

function Base.zero(community::CommunityPoint{B, D, P, T, NP, N, NC}) where {B, D, P, T, NP, N, NC}

    communityZero = CommunityPoint{B, D, P, T, NP, N, NC}(
            NamedTuple{P, T}(
                c ? similar(dt, eltype(dt)) : dt for (dt, c) in zip(values(community._propertiesAgent), community._propertiesCopy)               
            ),
            community._meta,
            community._propertiesCopy,
            N,
            NC
        )

    for (dt, dtOld, c) in zip(values(communityZero._propertiesAgent), values(community._propertiesAgent), community._propertiesCopy)
        if c
            @views dt[1:N] .= dtOld[1:N]
        end
    end
    
    return communityZero
end

function setCopyParameters(community::CommunityPoint{B, D, P, T, NP, N, NC}, params::NTuple{P2, Symbol}) where {B, D, P, T, NP, N, NC, P2}

    for s in params
        if !(s in P)
            error("Property $s not found in CommunityPoint.")
        end
    end

    _propertiesCopy = [i in params ? true : false for i in P]

    CommunityPoint{B, D, P, T, NP, N, NC}(
        community._propertiesAgent,
        community._meta,
        _propertiesCopy,
        N,
        NC,
    )
end

function addCopyParameter!(community::CommunityPoint{B, D, P, T, NP, N, NC}, param::Symbol) where {B, D, P, T, NP, N, NC}

    if !(param in P)
        error("Property $param not found in CommunityPoint.")
    end

    idx = findfirst(==(param), P)
    _propertiesCopy = [j || i == idx ? true : false for (i,j) in enumerate(P)]

    community = CommunityPoint{B, D, P, T, NP, N, NC}(
        community._propertiesAgent,
        community._meta,
        _propertiesCopy,
        N,
        NC,
    )

    return
end

# ######################################################################################################
# # MACROS
# ######################################################################################################
# macro loopOverAgentPoint(name::Symbol, iterator::Symbol, code::Expr)

#     N__ = Symbol(name, "__N")

#     if COMPILE_PLATFORM == :CPU

#         return quote
#             Threads.@inloops @inbounds for $iterator in 1:$N__
#                 $(code)
#             end
#         end

#     elseif COMPILE_PLATFORM == :GPU

#         return quote
#             @inbounds for $iterator in stride_:stride_:$N
#                 $(code)
#             end
#         end

#     end

# end

# macro addAgentPoint(name::Symbol, args...)

#     N = Symbol(name, "__N")
#     NCache = Symbol(name, "__NCache")
#     NNew = Symbol(name, "__NNew")
#     idMax = Symbol(name, "__idMax")
#     NFlag = Symbol(name, "__NFlag")

#     id = Symbol(name, "__id")

#     N__ = Symbol(name, "__N")

#     if COMPILE_PLATFORM == :CPU

#         return quote
#                 i1New_ = Threads.atomic_add!($NNew,1)
#                 idNew_ = Threads.atomic_add!($idMax,1)
#                 if $NNew > $NCache
#                     $NFlag = true
#                 else
#                     $id[i1New_] = idNew_
#                     $code
#                 end
#             end

#     elseif COMPILE_PLATFORM == :GPU

#         return quote
#                 i1New_ = CUDA.atomic_add!(CUDA.pointer($NNew,1),1)
#                 idNew_ = CUDA.atomic_add!(CUDA.pointer($idMax,1),1)
#                 if $NNew > $NCache
#                     $NFlag = true
#                 else
#                     $id[i1New_] = idNew_
#                     $code
#                 end
#             end

#     end

# end

# macro removeAgentPoint(name::Symbol, iterator::Symbol, agentPos::Symbol)

#     id = Symbol(name, "__id")

#     return quote

#         $id[$agentPos] = -1

#     end

# end

# macro loopOverAgentPointNeighbors(name::Symbol, iterator::Symbol, code::Expr)

#     N__ = Symbol(name, "__N")

#     if COMPILE_PLATFORM == :CPU

#         return quote
#             Threads.@inloops @inbounds for $iterator in 1:$N__
#                 for nbr in $neighbors[$iterator]
#                     $(code)
#                 end
#             end
#         end

#     elseif COMPILE_PLATFORM == :GPU

#         return quote
#             @inbounds for $iterator in stride_:stride_:$N
#                 for nbr in $neighbors[$iterator]
#                     $(code)
#                 end
#             end
#         end

#     end

# end
