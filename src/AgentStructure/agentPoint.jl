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

struct CommunityPoint{D, P, T, N, NC} <: AbstractCommunity where {D, P, T, N, NC}

    _propertiesAgent::NamedTuple{P, T}    # Dictionary to hold agent properties
    _N::SVector{1, Int}
    _NCache::SVector{1, Int}
    _NNew::SVector{1, Int}
    _idMax::SVector{1, Int}
    _NFlag::SVector{1, Bool}

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

    _N = SVector{1, Int}([N])
    _NCache = SVector{1, Int}([NCache])
    _NNew = SVector{1, Int}([N])
    _idMax = SVector{1, Int}([N])
    _NFlag = SVector{1, Bool}([false])

    return CommunityPoint{D, P, typeof(values(properties)), N, NCache}(
        properties,
        _N,
        _NCache,
        _NNew,
        _idMax,
        _NFlag
    )
end

function CommunityPoint(
        _propertiesAgent,
        _N,
        _NCache,
        _NNew,
        _idMax,
        _NFlag,
    )

    D = length(filter(k -> k in (:x, :y, :z), keys(_propertiesAgent)))
    P = keys(_propertiesAgent)
    T = typeof(values(_propertiesAgent))
    N = _N[1]
    NC = _NCache[1]

    CommunityPoint{D, P, T, N, NC}(
        _propertiesAgent,
        _N,
        _NCache,
        _NNew,
        _idMax,
        _NFlag
    )
end

Base.length(community::CommunityPoint{D, P, T, N, NC}) where {D, P, T, N, NC} = N
@inline Base.size(community::CommunityPoint{D, P, T, N, NC}) where {D, P, T, N, NC} = (length(P), N)
Base.ndims(community::CommunityPoint) = 2
Base.ndims(community::Type{<:CommunityPoint}) = 2

function Base.getindex(community::CommunityPoint, i::Integer)
    community._propertiesAgent[i]
end

function Base.iterate(community::CommunityPoint, state = 1)
    state >= length(community._propertiesAgent) + 1 ? nothing : (community[state], state + 1)
end

@generated function Base.getproperty(VA::CommunityPoint{D, P, T, N, NC}, s::Symbol) where {D, P, T, N, NC}
    names = P
    # build a clause for each fieldname in T
    cases = [
        :(s === $(QuoteNode(name)) && return @views getfield(getfield(VA, :_propertiesAgent), $(QuoteNode(name)))[1:N])
        for name in names
    ]

    quote
        if s === :_propertiesAgent; return getfield(VA, :_propertiesAgent); end
        if s === :_N; return getfield(VA, :_N); end
        if s === :_NCache; return getfield(VA, :_NCache); end
        if s === :_NNew; return getfield(VA, :_NNew); end
        if s === :_idMax; return getfield(VA, :_idMax); end
        if s === :_NFlag; return getfield(VA, :_NFlag); end
        $(cases...)
        error("Unknown property: $s for $VA")
    end
end

## broadcasting

struct CommunityPointStyle{N} <: Broadcast.AbstractArrayStyle{N} end
CommunityPointStyle(::Val{N}) where {N} = CommunityPointStyle{N}()

Broadcast.broadcastable(x::CommunityPoint) = x

## Copyto

for (type, N_expr) in [
        (Broadcast.Broadcasted{<:CommunityPoint}, :(narrays(bc))),
        (Broadcast.Broadcasted{<:Broadcast.DefaultArrayStyle}, :(length(dest._propertiesAgent))),
    ]

    @eval @inline function Base.copyto!(dest::CommunityPoint,
            bc::$type)
        bc = Broadcast.flatten(bc)
        N = $N_expr
        n = dest._N[1]
        @inbounds for i in 1:N
            if eltype(dest._propertiesAgent[i]) <: AbstractFloat
                dest_ = @views dest._propertiesAgent[i][1:n]
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

function unpack_voa(x::CommunityPoint{D, P, T, N, NC}, i) where {D, P, T, N, NC}
    @views x._propertiesAgent[i][1:N]
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
