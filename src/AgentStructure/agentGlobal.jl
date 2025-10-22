######################################################################################################
# AGENT STRUCTURE
######################################################################################################

struct AgentGlobal{P} <: AbstractAgent where {P}

    propertiesAgent::NamedTuple{}    # Dictionary to hold agent properties

end

function AgentGlobal(
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

    return AgentGlobal{typeof(propertiesAgentNew)}(
        propertiesAgentNew
    )
end

function Base.show(io::IO, x::AgentGlobal)
    println(io, "AgentGlobal with dimensions $(x._dims): \n")
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

function Base.show(io::IO, ::Type{AgentGlobal{P}}) where {P}
    print(io, "AgentGlobal{properties=(")
    for (i, (n, t)) in enumerate(zip(P.parameters[1], P.parameters[2]))
        i > 1 && print(io, ", ")
        print(io, n, "::", t.parameters[1])
    end
    print(io, ")}")
end

######################################################################################################
# COMMUNITY AGENT POINT
######################################################################################################
struct CommunityGlobal{P} <: AbstractCommunity where {P}
    _pa::NamedTuple{P, T}
    _paCopy::SVector{length(NP), Bool}
end
Adapt.@adapt_structure CommunityGlobal

function CommunityGlobal(
        agent::AgentGlobal{P},
    ) where {P}

    properties = NamedTuple{keys(agent.propertiesAgent)}(zeros(dtype(dt, isbits=true), NCache) for dt in values(agent.propertiesAgent))

    _paCopy = SizedVector{NP, Bool}([false for _ in 1:NP])    # Dictionary to hold agent properties for copying

    return CommunityGlobal{P}(
        properties,
        _paCopy,
    )
end

function CommunityGlobal(
        _pa,
        _paCopy,
    )

    P = typeof(_pa)

    CommunityGlobal{P}(
        _pa,
        _paCopy,
    )
end

@inline Base.length(community::CommunityGlobal{P}) where {P} = length(P)
@inline Base.size(community::CommunityGlobal{P, NP}) where {P, NP} = (NP, N)

# Base.axes(community::CommunityGlobal{P, NP}) where {P, NP} = (tuple(i for (i, dt) in enumerate(T) if eltype(dt) <: AbstractFloat), 1:N)
Base.ndims(community::CommunityGlobal) = 2
Base.ndims(community::Type{<:CommunityGlobal}) = 2
Base.IndexStyle(CommunityGlobal::CommunityGlobal) = Base.IndexStyle(typeof(CommunityGlobal))
Base.IndexStyle(::Type{<:CommunityGlobal}) = IndexLinear()
Base.CartesianIndices(community::CommunityGlobal{P, NP}) where {P, NP} =
    CartesianIndices((NP, N))
Base.getindex(cp::CommunityGlobal, i::Symbol) = cp._pa[i]
Base.getindex(cp::CommunityGlobal, i::Int) = cp._pa[i]
Base.setindex!(cp::CommunityGlobal, value, i::Symbol) = (cp._pa[i] = value)
Base.setindex!(cp::CommunityGlobal, value, i::Int) = (cp._pa[i] = value)
Base.setindex!(cp::CommunityGlobal, value, index::LinearIndexing) = (cp._pa[index[1]] = value)

function Base.getindex(community::CommunityGlobal, i::Integer)
    community._pa[i]
end

function Base.iterate(community::CommunityGlobal, state = 1)
    state >= length(community._pa) + 1 ? nothing : (community[state], state + 1)
end

@generated function Base.getproperty(VA::CommunityGlobal{P}, s::Symbol) where {P}
    names = P
    # build a clause for each fieldname in T
    cases = [
        :(s === $(QuoteNode(name)) && return @views getfield(getfield(VA, :_pa), $(QuoteNode(name)))[1:N])
        for name in names
    ]

    quote
        if s === :_pa; return getfield(VA, :_pa); end
        if s === :_paCopy; return getfield(VA, :_paCopy); end
        $(cases...)
        error("Unknown property: $s for $VA")
    end
end

## broadcasting

struct CommunityGlobalStyle{N} <: Broadcast.AbstractArrayStyle{N} end

# Allow constructing the style from Val or Int
CommunityGlobalStyle(::Val{N}) where {N} = CommunityGlobalStyle{1}()
CommunityGlobalStyle(N::Int) = CommunityGlobalStyle{1}()
CommunityGlobalStyle() = CommunityGlobalStyle{1}()

# Your CommunityGlobal acts like a 2D array
Base.BroadcastStyle(::Type{<:CommunityGlobal}) = CommunityGlobalStyle{1}()

# Combine styles safely
Base.Broadcast.result_style(::CommunityGlobalStyle{M}) where {M} =
    CommunityGlobalStyle{M}()
Base.Broadcast.result_style(::CommunityGlobalStyle{M}, ::CommunityGlobalStyle{N}) where {M,N} =
    CommunityGlobalStyle{max(M,N)}()
Base.Broadcast.result_style(::CommunityGlobalStyle{M}, ::Base.Broadcast.AbstractArrayStyle{N}) where {M,N} =
    CommunityGlobalStyle{max(M,N)}()
Base.Broadcast.result_style(::Base.Broadcast.AbstractArrayStyle{M}, ::CommunityGlobalStyle{N}) where {M,N} =
    CommunityGlobalStyle{max(M,N)}()

# Base.length(::CellBasedModels.CommunityGlobalStyle{N}) where {N} = N

Broadcast.broadcastable(x::CommunityGlobal) = x

## Copyto
@eval @inline function Base.copy(dest::CommunityGlobal{P, NP}) where {P, NP}

    deepcopy(dest)

end

@eval @inline function Base.copy(dest::CommunityGlobal{P}, subset::NTuple{P2}) where {P, P2}

    for s in subset
        if !(s in P)
            error("Property $s not found in CommunityGlobal.")
        end
    end

    CommunityGlobal{P}(
        NamedTuple{P}(
            i in names(subset) ? copy(dest._pa[i]) : dest._pa[i] for i in P
        ),
        SizedVector{length(P), Bool}([i in names(subset) ? true : false for i in P])    # Dictionary to hold agent properties for copying
    )

end

@eval @inline function Base.copyto!(dest::CommunityGlobal{P, NP},
        bc::CommunityGlobal{P, NP2}) where {P, NP, NC2}
    @inbounds for i in 1:NP
        if bc._paCopy[i]
            @views dest._pa[i][1:N] .= bc._pa[i][1:N]
        end
    end
    dest
end

for type in [
        Broadcast.Broadcasted{<:CommunityGlobal},
        Broadcast.Broadcasted{<:CommunityGlobalStyle},
    ]

    @eval @inline function Base.copyto!(dest::CommunityGlobal{P, NP},
            bc::$type) where {P, NP}
        bc = Broadcast.flatten(bc)
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
@inline function unpack_voa(bc::Broadcast.Broadcasted{<:CommunityGlobal}, i)
    Broadcast.Broadcasted(bc.f, unpack_args_voa(i, bc.args))
end

function unpack_voa(x::CommunityGlobal{P, NP}, i) where {P, NP}
    @views x._pa[i][1:N]
end

# Integrator
function Base.eltype(::Type{<:CommunityGlobal{P, NP}}) where {P, NP}
    for i in T.parameters
        if i <: AbstractFloat
            return eltype(i)
        end
    end
    return Float64
end

function Base.zero(community::CommunityGlobal{P, NP}) where {P, NP}

    communityZero = CommunityGlobal{P, NP}(
            NamedTuple{P, T}(
                c ? similar(dt, eltype(dt)) : dt for (dt, c) in zip(values(community._pa), community._paCopy)               
            ),
            community._m,
            community._paCopy,
            N,
            NC
        )

    for (dt, dtOld, c) in zip(values(communityZero._pa), values(community._pa), community._paCopy)
        if c
            @views dt[1:N] .= dtOld[1:N]
        end
    end
    
    return communityZero
end

function setCopyParameters(community::CommunityGlobal{P, NP}, params::NTuple{P2, Symbol}) where {P, NP, P2}

    for s in params
        if !(s in P)
            error("Property $s not found in CommunityGlobal.")
        end
    end

    _paCopy = [i in params ? true : false for i in P]

    CommunityGlobal{P, NP}(
        community._pa,
        community._m,
        _paCopy,
        N,
        NC,
    )
end

function addCopyParameter!(community::CommunityGlobal{P, NP}, param::Symbol) where {P, NP}

    if !(param in P)
        error("Property $param not found in CommunityGlobal.")
    end

    idx = findfirst(==(param), P)
    _paCopy = [j || i == idx ? true : false for (i,j) in enumerate(P)]

    community = CommunityGlobal{P, NP}(
        community._pa,
        community._m,
        _paCopy,
        N,
        NC,
    )

    return
end

# ######################################################################################################
# # MACROS
# ######################################################################################################
# macro loopOverAgentGlobal(name::Symbol, iterator::Symbol, code::Expr)

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

# macro addAgentGlobal(name::Symbol, args...)

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

# macro removeAgentGlobal(name::Symbol, iterator::Symbol, agentPos::Symbol)

#     id = Symbol(name, "__id")

#     return quote

#         $id[$agentPos] = -1

#     end

# end

# macro loopOverAgentGlobalNeighbors(name::Symbol, iterator::Symbol, code::Expr)

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
