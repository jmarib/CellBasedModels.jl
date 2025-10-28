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
            par.dimensions === nothing ? "" : string(par.dimensions),
            par.defaultValue === nothing ? "" : string(par.defaultValue), 
            par.description))
    end
    println(io)
end

function Base.show(io::IO, ::Type{AgentPoint{D, P, T}}) where {D, P, T}
    print(io, "AgentPoint{dims=", D, ", properties=(")
    for (i, (n, t)) in enumerate(zip(P, T.parameters))
        i > 1 && print(io, ", ")
        print(io, n, "::", t.parameters[1])
    end
    print(io, ")}")
end

######################################################################################################
# COMMUNITY AGENT POINT
######################################################################################################
struct CommunityPointMeta{TR, AI, AB, VI, VB, SI, VVNT}
    _N::AI
    _NCache::AI

    _id::VI
    _idMax::AI
    
    _flagsRemoved::VB
    _NRemoved::AI
    _NRemovedThread::SI
    
    _NAdded::AI
    _NAddedThread::SI    
    _addedAgents::VVNT

    _flagOverflow::AB
end
Adapt.@adapt_structure CommunityPointMeta

function CommunityPointMeta(
        agent::AgentPoint,
        N::Int,
        NCache::Int,
)
    _N = Threads.Atomic{Int}(N)
    _NCache = Threads.Atomic{Int}(NCache)

    _id = Vector{Int}(zeros(Int, NCache))
    _id[1:N] = 1:N
    _idMax = Threads.Atomic{Int}(N)
    
    _flagsRemoved = Vector{Int}(zeros(Int, NCache))
    _NRemoved = Threads.Atomic{Int}(0)
    _NRemovedThread = SizedVector{Threads.nthreads(), Int}(zeros(Int, Threads.nthreads()))

    _NAdded = Threads.Atomic{Int}(0)
    _NAddedThread = SizedVector{Threads.nthreads(), Int}(zeros(Int, Threads.nthreads()))
    P = keys(agent.propertiesAgent)
    T = Tuple{(dtype(i, isbits=true) for i in values(agent.propertiesAgent))...}
    _addedAgents = [Vector{NamedTuple{P, T}}() for _ in 1:Threads.nthreads()]

    _flagOverflow = Threads.Atomic{Bool}(false)

    TR = Threads.nthreads() > 1 ? true : false
    AI = typeof(_N)
    AB = typeof(_flagOverflow)
    VI = typeof(_id)
    VB = typeof(_flagsRemoved)
    SI = typeof(_NRemovedThread)
    VVNT = typeof(_addedAgents)

    CommunityPointMeta{TR, AI, AB, VI, VB, SI, VVNT}(
        _N,
        _NCache,

        _id,
        _idMax,
        
        _flagsRemoved,
        _NRemoved,
        _NRemovedThread,

        _NAdded,
        _NAddedThread,
        _addedAgents,

        _flagOverflow,
    )
end

function CommunityPointMeta(
        _N,
        _NCache,

        _id,
        _idMax,
        
        _flagsRemoved,
        _NRemoved,
        _NRemovedThread,

        _NAdded,
        _NAddedThread,
        _addedAgents,

        _flagOverflow,
    )

    TR = Threads.nthreads() > 1 ? true : false
    AI = typeof(_N)
    AB = typeof(_flagOverflow)
    VI = typeof(_id)
    VB = typeof(_flagsRemoved)
    SI = typeof(_NRemovedThread)
    VVNT = typeof(_addedAgents)

    CommunityPointMeta{TR, AI, AB, VI, VB, SI, VVNT}(
        _N,
        _NCache,

        _id,
        _idMax,
        
        _flagsRemoved,
        _NRemoved,
        _NRemovedThread,

        _NAdded,
        _NAddedThread,
        _addedAgents,

        _flagOverflow,
    )
end
struct CommunityPoint{B, D, P, T, NP} <: AbstractCommunity where {B, D, P, T, NP}
    _pa::NamedTuple{P, T}
    _m::B
    _paCopy::SVector{NP, Bool}
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

    NP = length(P)

    _m = CommunityPointMeta(
        agent,
        N,
        NCache,
    )

    _paCopy = SizedVector{NP, Bool}([false for _ in 1:NP])    # Dictionary to hold agent properties for copying

    return CommunityPoint{typeof(_m), D, P, typeof(values(properties)), NP}(
        properties,
        _m,
        _paCopy,
    )
end

function CommunityPoint(
        _pa,
        _m,
        _paCopy,
    )

    B = typeof(_m)
    D = length(filter(k -> k in (:x, :y, :z), keys(_pa)))
    P = keys(_pa)
    T = typeof(values(_pa))
    NP = length(P)

    if NP != length(P)
        error("Length of _pa does not match NCache.")
    end

    CommunityPoint{B, D, P, T, NP}(
        _pa,
        _m,
        _paCopy,
    )
end

function Base.show(io::IO, x::CommunityPoint{B, D, P, T, NP}) where {B, D, P, T, NP}
    println(io, "CommunityPoint $D D N=$(length(x)) NCache=$(lengthCache(x)): \n")
    println(io, @sprintf("\t%-15s %-15s", "Name", "DataType"))
    println(io, "\t" * repeat("-", 85))
    for ((name, par), c) in zip(pairs(x._pa), x._paCopy)
        println(io, @sprintf("\t%-15s %-15s", 
            c ? string("*", name) : string(name),
            typeof(par)))
    end
    println(io)
end

# function Base.show(io::IO, ::Type{CommunityPoint{B, D, P, T, NP}}) where {B, D, P, T, NP}
#     print(io, "CommunityPoint{dims=", D, ", N=", N, ", NCache=", NC, ", properties=(")
#     for (i, (n, t)) in enumerate(zip(P, T.parameters))
#         i > 1 && print(io, ", ")
#         print(io, n, "::", t.parameters[1])
#     end
#     print(io, ")}")
# end

Base.length(community::CommunityPoint{B, D, P, T, NP}) where {B, D, P, T, NP} = community._m._N[]
lengthCache(community::CommunityPoint{B, D, P, T, NP}) where {B, D, P, T, NP} = community._m._NCache[]
Base.size(community::CommunityPoint{B, D, P, T, NP}) where {B, D, P, T, NP} = (NP, length(community))

# Base.axes(community::CommunityPoint{B, D, P, T, NP}) where {B, D, P, T, NP} = (tuple(i for (i, dt) in enumerate(T) if eltype(dt) <: AbstractFloat), 1:N)
Base.ndims(community::CommunityPoint) = 2
Base.ndims(community::Type{<:CommunityPoint}) = 2
Base.IndexStyle(CommunityPoint::CommunityPoint) = Base.IndexStyle(typeof(CommunityPoint))
Base.IndexStyle(::Type{<:CommunityPoint}) = IndexCartesian()
Base.CartesianIndices(community::CommunityPoint{B, D, P, T, NP}) where {B, D, P, T, NP} =
    CartesianIndices((NP, N))
Base.getindex(cp::CommunityPoint, i::Symbol, j::Int) = cp._pa[i][j]
Base.getindex(cp::CommunityPoint, i::Int, j::Int) = cp._pa[i][j]
Base.getindex(cp::CommunityPoint, i::Symbol, :) = cp._pa[i]
Base.getindex(cp::CommunityPoint, i::Int, :) = cp._pa[i]
Base.getindex(cp::CommunityPoint{B, D, P, T, NP}, :, j::Int) where {B, D, P, T, NP} = NamedTuple{P}(
    (cp._pa[k][j] for k in keys(cp._pa))
)
Base.setindex!(cp::CommunityPoint, value, i::Symbol, j::Int) = (cp._pa[i][j] = value)
Base.setindex!(cp::CommunityPoint, value, i::Int, j::Int) = (cp._pa[i][j] = value)
Base.setindex!(cp::CommunityPoint, value, index::CartesianIndex{2}) = (cp._pa[index[1]][index[2]] = value)
# Base.keys(cp::CommunityPoint) = keys(cp._pa)

function Base.getindex(community::CommunityPoint, i::Integer)
    community._pa[i]
end

@generated function Base.getproperty(community::CommunityPoint{B, D, P, T, NP}, s::Symbol) where {B, D, P, T, NP}
    names = P
    # build a clause for each fieldname in T
    cases = [
        :(s === $(QuoteNode(name)) && return @views getfield(getfield(community, :_pa), $(QuoteNode(name)))[1:length(community)])
        for name in names
    ]

    quote
        if s === :_pa; return getfield(community, :_pa); end
        if s === :_m; return getfield(community, :_m); end
        if s === :_paCopy; return getfield(community, :_paCopy); end
        $(cases...)
        error("Unknown property: $s for of the CommunityPoint.")
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

Broadcast.broadcastable(x::CommunityPoint) = x

## Copyto
@eval @inline function Base.copy(dest::CommunityPoint{B, D, P, T, NP}) where {B, D, P, T, NP}

    deepcopy(dest)

end

@eval @inline function Base.copy(dest::CommunityPoint{B, D, P, T, NP}, subset::NTuple{P2, Symbol}) where {B, D, P, T, NP, P2}

    for s in subset
        if !(s in P)
            error("Property $s not found in CommunityPoint.")
        end
    end

    CommunityPoint{B, D, P, T, NP}(
        NamedTuple{P}(
            i in names(subset) ? copy(dest._pa[i]) : dest._pa[i] for i in P
        ),
        dest._m,
        SizedVector{NP, Bool}([i in names(subset) ? true : false for i in P])    # Dictionary to hold agent properties for copying
    )

end

@eval @inline function Base.copyto!(dest::CommunityPoint{B, D, P, T, NP},
        bc::CommunityPoint{B, D, P, T, NP}) where {B, D, P, T, NP}
    N = length(dest)
    @inbounds for i in 1:NP
        if bc._paCopy[i]
            @views dest._pa[i][1:N] .= bc._pa[i][1:N]
        end
    end
    dest
end

for type in [
        Broadcast.Broadcasted{<:CommunityPoint},
        Broadcast.Broadcasted{<:CommunityPointStyle},
    ]

    @eval @inline function Base.copyto!(dest::CommunityPoint{B, D, P, T, NP},
            bc::$type) where {B, D, P, T, NP}
        bc = Broadcast.flatten(bc)
        N = length(dest)
        @inbounds for i in 1:NP
            if dest._paCopy[i]
                dest_ = @views dest._pa[i][1:N]
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

function unpack_voa(x::CommunityPoint{B, D, P, T, NP}, i) where {B, D, P, T, NP}
    @views x._pa[i][1:length(x)]
end

# Integrator
function Base.eltype(::Type{<:CommunityPoint{B, D, P, T, NP}}) where {B, D, P, T, NP}
    for i in T.parameters
        if i <: AbstractFloat
            return eltype(i)
        end
    end
    return Float64
end

function Base.zero(community::CommunityPoint{B, D, P, T, NP}) where {B, D, P, T, NP}

    communityZero = CommunityPoint{B, D, P, T, NP}(
            NamedTuple{P, T}(
                c ? similar(dt, eltype(dt)) : dt for (dt, c) in zip(values(community._pa), community._paCopy)               
            ),
            community._m,
            community._paCopy
        )

    N = length(community)
    for (dt, dtOld, c) in zip(values(communityZero._pa), values(community._pa), community._paCopy)
        if c
            @views dt[1:N] .= dtOld[1:N]
        end
    end
    
    return communityZero
end

function setCopyParameters(community::CommunityPoint{B, D, P, T, NP}, params::NTuple{P2, Symbol}) where {B, D, P, T, NP, P2}

    for s in params
        if !(s in P)
            error("Property $s not found in CommunityPoint.")
        end
    end

    _paCopy = [i in params ? true : false for i in P]

    CommunityPoint{B, D, P, T, NP}(
        community._pa,
        community._m,
        _paCopy
    )
end

function addCopyParameter!(community::CommunityPoint{B, D, P, T, NP}, param::Symbol) where {B, D, P, T, NP}

    if !(param in P)
        error("Property $param not found in CommunityPoint.")
    end

    idx = findfirst(==(param), P)
    _paCopy = [j || i == idx ? true : false for (i,j) in enumerate(P)]

    community = CommunityPoint{B, D, P, T, NP}(
        community._pa,
        community._m,
        _paCopy
    )

    return
end

function Base.iterate(community::CommunityPoint, state = (1, length(community)))
    state[1] >= state[2] ? nothing : (state, (state[1] + 1, state[2]))
end

######################################################################################################
# Iterator
######################################################################################################
struct CommunityPointIterator{B}
    N::Int
end

function loopOverAgents(community::CommunityPoint{B, D, P, T, NP}) where {B, D, P, T, NP}
    CommunityPointIterator{B}(length(community))
end

function Base.iterate(iterator::CommunityPointIterator, state = 1)
    state >= iterator.N + 1 ? nothing : (state, state + 1)
end

# Necessary for working with Threads
Base.firstindex(iterator::CommunityPointIterator) = 1
Base.lastindex(iterator::CommunityPointIterator) = iterator.N
Base.length(iterator::CommunityPointIterator) = iterator.N
Base.getindex(iterator::CommunityPointIterator, i::Int) = i


######################################################################################################
# Kernel functions
######################################################################################################
function removeAgent!(community::CommunityPoint, pos::Int)
    if pos < 1 || pos > length(community)
        @warn "Position $pos is out of bounds for CommunityPoint with N=$(length(community)). No agent removed."
    else
        community._m._flagsRemoved[pos] = true
    end
    return
end

@generated function addAgent!(community::CommunityPoint{<:CommunityPointMeta{TR, S}, D, P, T, NP}, kwargs::NamedTuple{P2, T2}) where {TR, S<:Threads.Atomic, D, P, T, NP, P2, T2}

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
        newPos = Threads.atomic_add!(community._m._NNew, 1) + 1
        if newPos > community._m._NCache[]
            community._m._overflowFlag[] = true
            return
        else
            newId = Threads.atomic_add!(community._m._idMax, 1) + 1
            community._m._id[newPos] = newId
            $(cases...)
        end
    end

end