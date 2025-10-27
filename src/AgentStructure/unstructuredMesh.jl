######################################################################################################
# AGENT STRUCTURE
######################################################################################################
struct UnstructuredMeshProperties{D, S, PA, PN, PE, PF, PV}

    propertiesAgent::Union{NamedTuple, Nothing}                   # Dictionary to hold agent properties
    propertiesNode::Union{NamedTuple, Nothing}    # Dictionary to hold agent properties
    propertiesEdge::Union{NamedTuple, Nothing}    # Dictionary to hold agent properties
    propertiesFace::Union{NamedTuple, Nothing}    # Dictionary to hold agent properties
    propertiesVolume::Union{NamedTuple, Nothing}    # Dictionary to hold agent properties

end

function _posMerge(properties, defaultParameters)
    for (k, v) in pairs(properties)
        if haskey(defaultParameters, k)
            error("Parameter $k is already defined and cannot be used as agent property.")
        elseif startswith(string(k), "_")
            error("Parameter $k is protected and cannot be used as agent property.")
        end
    end
    return merge(defaultParameters, properties)
end

function UnstructuredMeshProperties(
    dims::Int;
    propertiesAgent::Union{NamedTuple, Nothing}=nothing,
    propertiesNode::Union{NamedTuple, Nothing}=nothing,
    propertiesEdge::Union{NamedTuple, Nothing}=nothing,
    propertiesFace::Union{NamedTuple, Nothing}=nothing,
    propertiesVolume::Union{NamedTuple, Nothing}=nothing,
    scopePosition::Symbol = :propertiesAgent,
)

    if !(scopePosition in fieldnames(UnstructuredMeshProperties))
        error("scopePosition must be one of $(fieldnames(UnstructuredMeshProperties)). Found $scopePosition")
    end

    if dims < 0 || dims > 3
        error("dims must be between 0 and 3. Found $dims")
    end
    defaultParameters = (
        x = dims >= 1 ? Parameter(AbstractFloat, description="Position in x (protected parameter)", dimensions=:L, _scope=scopePosition) : nothing,
        y = dims >= 2 ? Parameter(AbstractFloat, description="Position in y (protected parameter)", dimensions=:L, _scope=scopePosition) : nothing,
        z = dims >= 3 ? Parameter(AbstractFloat, description="Position in z (protected parameter)", dimensions=:L, _scope=scopePosition) : nothing,
    )
    defaultParameters = NamedTuple{Tuple(k for (k,v) in pairs(defaultParameters) if v !== nothing)}(
        (v for (k,v) in pairs(defaultParameters) if v !== nothing)
    )

    properties = []
    for (pn, p) in zip(
        (:propertiesAgent, :propertiesNode, :propertiesEdge, :propertiesFace, :propertiesVolume),
        (propertiesAgent, propertiesNode, propertiesEdge, propertiesFace, propertiesVolume)
    )
        if scopePosition === pn && p === nothing
            error("Parameter $pn is not defined but assegned as scopePosition.")
        elseif scopePosition == pn
            push!(properties, _posMerge(parameterConvert(p, scope=pn), defaultParameters))
        elseif p === nothing
            push!(properties, nothing)
        else
            push!(properties, parameterConvert(p, scope=pn))
        end
    end

    scope = Dict(
        :propertiesAgent => 1,
        :propertiesNode => 2,
        :propertiesEdge => 3,
        :propertiesFace => 4,
        :propertiesVolume => 5,
    )[scopePosition] 

    return UnstructuredMeshProperties{dims, scope, (typeof(i) for i in properties)...}(
        properties...
    )
end

spatialDims(x::UnstructuredMeshProperties{D}) where {D} = D
scopePosition(x::UnstructuredMeshProperties{D, S}) where {D, S} = S

function Base.show(io::IO, x::UnstructuredMeshProperties{D}) where {D}
    println(io, "UnstructuredMeshProperties with dimensions $(D): \n")
    for p in propertynames(x)
        props = getfield(x, p)
        if props !== nothing
            println(io, replace(uppercase(string(p)), "PROPERTIES"=>"PROPERTIES "))
            println(io, @sprintf("\t%-15s %-15s %-15s %-20s %-s", "Name", "DataType", "Dimensions", "Default_Value", "Description"))
            println(io, "\t" * repeat("-", 85))
            for (name, par) in pairs(props)
                println(io, @sprintf("\t%-15s %-15s %-15s %-20s %-s", 
                    name, 
                    dtype(par), 
                    par.dimensions === nothing ? "" : string(par.dimensions),
                    par.defaultValue === nothing ? "" : string(par.defaultValue), 
                    par.description))
            end
            println(io)
        end
    end
end

function Base.show(io::IO, ::Type{UnstructuredMeshProperties{D, S, PA, PN, PE, PF, PV}}) where {D, S, PA, PN, PE, PF, PV}
    println(io, "UnstructuredMeshProperties{dims=", D, ", scopePosition=", S, ",")
    for (props, propsnames) in zip((PA, PN, PE, PF, PV), (:propertiesAgent, :propertiesNode, :propertiesEdge, :propertiesFace, :propertiesVolume))
        if props !== Nothing
            print(io, "\t", string(propsnames), "=(")
            for (i, (n, t)) in enumerate(zip(props.parameters[1], props.parameters[2].parameters))
                i > 1 && print(io, ", ")
                print(io, string(n), "::", t.parameters[1])
            end
            println(io, ")")
        end
    end
    println(io, "}")
end

######################################################################################################
# UnstructuredMeshObjectMeta
######################################################################################################
struct UnstructuredMeshObjectMeta{
            TR, IDVI, IDAI,    
            AI, VB, SI, VVNT, AB, 
        }
    _id::IDVI
    _idMax::IDAI

    _N::AI
    _NCache::AI
    _FlagsRemoved::VB
    _NRemoved::AI
    _NRemovedThread::SI
    _NAdded::AI
    _NAddedThread::SI    
    _AddedAgents::VVNT
    _FlagOverflow::AB
end
Adapt.@adapt_structure UnstructuredMeshObjectMeta

function UnstructuredMeshObjectMeta(
        meshProperties::Union{NamedTuple, Nothing};
        N::Int=0,
        NCache::Int=0,
        id=true
)

    if meshProperties === nothing
        return nothing
    end

    if id
        _id = Vector{Int}(zeros(Int, NCache))
        _id[1:N] = 1:N
        _idMax = Threads.Atomic{Int}(N)
    else
        _id = nothing
        _idMax = nothing
    end

    _N = meshProperties !== nothing ? Threads.Atomic{Int}(N) : nothing
    _NCache = meshProperties !== nothing ? Threads.Atomic{Int}(NCache) : nothing
    _FlagsRemoved = meshProperties !== nothing ? Vector{Int}(zeros(Int, NCache)) : nothing
    _NRemoved = meshProperties !== nothing ? Threads.Atomic{Int}(0) : nothing
    _NRemovedThread = meshProperties !== nothing ? SizedVector{Threads.nthreads(), Int}(zeros(Int, Threads.nthreads())) : nothing
    _NAdded = meshProperties !== nothing ? Threads.Atomic{Int}(0) : nothing
    _NAddedThread = meshProperties !== nothing ? SizedVector{Threads.nthreads(), Int}(zeros(Int, Threads.nthreads())) : nothing
    P = meshProperties !== nothing ? keys(meshProperties) : nothing
    T = meshProperties !== nothing ? Tuple{(dtype(i, isbits=true) for i in values(meshProperties))...} : nothing
    _AddedAgents = meshProperties !== nothing ? [Vector{NamedTuple{P, T}}() for _ in 1:Threads.nthreads()] : nothing
    _FlagOverflow = meshProperties !== nothing ? Threads.Atomic{Bool}(false) : nothing

    TR = Threads.nthreads() > 1 ? true : false
    IDVI = typeof(_id)
    IDAI = typeof(_idMax)

    AI = typeof(_N)
    AB = typeof(_FlagOverflow)
    VB = typeof(_FlagsRemoved)
    SI = typeof(_NRemovedThread)
    VVNT = typeof(_AddedAgents)

    UnstructuredMeshObjectMeta{
            TR, IDVI, IDAI,
            AI, VB, SI, VVNT, AB,
        }(
            _id,
            _idMax,

            _N,
            _NCache,
            _FlagsRemoved,
            _NRemoved,
            _NRemovedThread,
            _NAdded,
            _NAddedThread,
            _AddedAgents,
            _FlagOverflow,
        )
end

function UnstructuredMeshObjectMeta(
            _id,
            _idMax,

            _N,
            _NCache,
            _FlagsRemoved,
            _NRemoved,
            _NRemovedThread,
            _NAdded,
            _NAddedThread,
            _AddedAgents,
            _FlagOverflow,
    )

    TR = Threads.nthreads() > 1 ? true : false
    IDVI = typeof(_id)
    IDAI = typeof(_idMax)

    AI = typeof(_N)
    AB = typeof(_FlagOverflow)
    VB = typeof(_FlagsRemoved)
    SI = typeof(_NRemovedThread)
    VVNT = typeof(_AddedAgents)

    UnstructuredMeshObjectMeta{
            TR, IDVI, IDAI,
            AI, AB, VB, SI, VVNT
        }(
            _id,
            _idMax,

            _N,
            _NCache,
            _FlagsRemoved,
            _NRemoved,
            _NRemovedThread,
            _NAdded,
            _NAddedThread,
            _AddedAgents,
            _FlagOverflow,
        )
end

######################################################################################################
# UnstructuredMeshObjectMeta
######################################################################################################
struct UnstructuredMeshObject{
            D, P, S,
            AM, AP, ANP,
            NM, NP, NNP,
            EM, EP, ENP,
            FM, FP, FNP,
            VM, VP, VNP,
        }
    
    _scope::Int

    _am::AM
    _ap::AP
    _apCopy::ANP

    _nm::NM
    _np::NP
    _npCopy::NNP

    _em::EM
    _ep::EP
    _epCopy::ENP

    _fm::FM
    _fp::FP
    _fpCopy::FNP

    _vm::VM
    _vp::VP
    _vpCopy::VNP
end
Adapt.@adapt_structure UnstructuredMeshObject

function UnstructuredMeshObject(
        mesh::UnstructuredMeshProperties{D, S, PA, PN, PE, PF, PV},
        agentN::Integer=0,
        agentNCache::Union{Nothing, Integer}=nothing,
        nodeN::Integer=0,
        nodeNCache::Union{Nothing, Integer}=nothing,
        edgeN::Integer=0,
        edgeNCache::Union{Nothing, Integer}=nothing,
        faceN::Integer=0,
        faceNCache::Union{Nothing, Integer}=nothing,
        volumeN::Integer=0,
        volumeNCache::Union{Nothing, Integer}=nothing,
    ) where {D, S, PA, PN, PE, PF, PV}

    P = Threads.nthreads() > 1 ? true : false

    if agentN < 0
        error("agentN must be greater than 0. Found N=$agentN")
    end
    if agentNCache === nothing
        agentNCache = agentN
    elseif agentNCache < agentN
        error("agentNCache must be greater than or equal to agentN. Found agentNCache=$agentNCache and agentN=$agentN")
    end

    _am = UnstructuredMeshObjectMeta(mesh.propertiesAgent, N = agentN, NCache = agentNCache)
    _ap = _am !== nothing ? NamedTuple{keys(mesh.propertiesAgent)}(zeros(dtype(dt, isbits=true), agentNCache) for dt in values(mesh.propertiesAgent)) : nothing
    _apCopy = _am !== nothing ? SizedVector{length(_ap), Bool}([false for _ in 1:length(_ap)]) : nothing

    if nodeN < 0
        error("nodeN must be greater than 0. Found N=$nodeN")
    end
    if nodeNCache === nothing
        nodeNCache = nodeN
    elseif nodeNCache < nodeN
        error("nodeNCache must be greater than or equal to nodeN. Found nodeNCache=$nodeNCache and nodeN=$nodeN")
    end

    _nm = UnstructuredMeshObjectMeta(mesh.propertiesNode, N = nodeN, NCache = nodeNCache)
    _np = _nm !== nothing ? NamedTuple{keys(mesh.propertiesNode)}(zeros(dtype(dt, isbits=true), nodeNCache) for dt in values(mesh.propertiesNode)) : nothing
    _npCopy = _nm !== nothing ? SizedVector{length(_np), Bool}([false for _ in 1:length(_np)]) : nothing

    if edgeN < 0
        error("edgeN must be greater than 0. Found N=$edgeN")
    end
    if edgeNCache === nothing
        edgeNCache = edgeN
    elseif edgeNCache < edgeN
        error("edgeNCache must be greater than or equal to edgeN. Found edgeNCache=$edgeNCache and edgeN=$edgeN")
    end

    _em = UnstructuredMeshObjectMeta(mesh.propertiesEdge, N = edgeN, NCache = edgeNCache)
    _ep = _em !== nothing ? NamedTuple{keys(mesh.propertiesEdge)}(zeros(dtype(dt, isbits=true), edgeNCache) for dt in values(mesh.propertiesEdge)) : nothing
    _epCopy = _em !== nothing ? SizedVector{length(_ep), Bool}([false for _ in 1:length(_ep)]) : nothing

    if faceN < 0
        error("faceN must be greater than 0. Found N=$faceN")
    end
    if faceNCache === nothing
        faceNCache = faceN
    elseif faceNCache < faceN
        error("faceNCache must be greater than or equal to faceN. Found faceNCache=$faceNCache and faceN=$faceN")
    end

    _fm = UnstructuredMeshObjectMeta(mesh.propertiesFace, N = faceN, NCache = faceNCache)
    _fp = _fm !== nothing ? NamedTuple{keys(mesh.propertiesFace)}(zeros(dtype(dt, isbits=true), faceNCache) for dt in values(mesh.propertiesFace)) : nothing
    _fpCopy = _fm !== nothing ? SizedVector{length(_fp), Bool}([false for _ in 1:length(_fp)]) : nothing

    if volumeN < 0
        error("volumeN must be greater than 0. Found N=$volumeN")
    end
    if volumeNCache === nothing
        volumeNCache = volumeN
    elseif volumeNCache < volumeN
        error("volumeNCache must be greater than or equal to volumeN. Found volumeNCache=$volumeNCache and volumeN=$volumeN")
    end

    _vm = UnstructuredMeshObjectMeta(mesh.propertiesVolume, N = volumeN, NCache = volumeNCache)
    _vp = _vm !== nothing ? NamedTuple{keys(mesh.propertiesVolume)}(zeros(dtype(dt, isbits=true), volumeNCache) for dt in values(mesh.propertiesVolume)) : nothing
    _vpCopy = _vm !== nothing ? SizedVector{length(_vp), Bool}([false for _ in 1:length(_vp)]) : nothing

    return UnstructuredMeshObject{
            D, P, S,
            typeof(_am), typeof(_ap), typeof(_apCopy),
            typeof(_nm), typeof(_np), typeof(_npCopy),
            typeof(_em), typeof(_ep), typeof(_epCopy),
            typeof(_fm), typeof(_fp), typeof(_fpCopy),
            typeof(_vm), typeof(_vp), typeof(_vpCopy)
        }(
            S,

            _am,
            _ap,
            _apCopy,

            _nm,
            _np,
            _npCopy,

            _em,
            _ep,
            _epCopy,

            _fm,
            _fp,
            _fpCopy,

            _vm,
            _vp,
            _vpCopy,
        )
end

function UnstructuredMeshObject(
            scope,

            _am,
            _ap,
            _apCopy,

            _nm,
            _np,
            _npCopy,

            _em,
            _ep,
            _epCopy,

            _fm,
            _fp,
            _fpCopy,

            _vm,
            _vp,
            _vpCopy,
    )

    D = length(filter(k -> k in (:x, :y, :z), keys(_pa)))
    P = keys(_pa)
    S = scope

    return UnstructuredMeshObject{
            D, P, S,
            typeof(_am), typeof(_ap), typeof(_apCopy),
            typeof(_nm), typeof(_np), typeof(_npCopy),
            typeof(_em), typeof(_ep), typeof(_epCopy),
            typeof(_fm), typeof(_fp), typeof(_fpCopy),
            typeof(_vm), typeof(_vp), typeof(_vpCopy)
        }(
            scope,

            _am,
            _ap,
            _apCopy,

            _nm,
            _np,
            _npCopy,

            _em,
            _ep,
            _epCopy,

            _fm,
            _fp,
            _fpCopy,

            _vm,
            _vp,
            _vpCopy,
        )
end

# function Base.show(io::IO, x::UnstructuredMeshObject{B, D, P, T, NP}) where {B, D, P, T, NP}
#     println(io, "UnstructuredMeshObject $D D N=$(length(x)) NCache=$(lengthCache(x)): \n")
#     println(io, @sprintf("\t%-15s %-15s", "Name", "DataType"))
#     println(io, "\t" * repeat("-", 85))
#     for ((name, par), c) in zip(pairs(x._pa), x._paCopy)
#         println(io, @sprintf("\t%-15s %-15s", 
#             c ? string("*", name) : string(name),
#             typeof(par)))
#     end
#     println(io)
# end

# # function Base.show(io::IO, ::Type{UnstructuredMeshObject{B, D, P, T, NP}}) where {B, D, P, T, NP}
# #     print(io, "UnstructuredMeshObject{dims=", D, ", N=", N, ", NCache=", NC, ", properties=(")
# #     for (i, (n, t)) in enumerate(zip(P, T.parameters))
# #         i > 1 && print(io, ", ")
# #         print(io, n, "::", t.parameters[1])
# #     end
# #     print(io, ")}")
# # end

# Base.length(community::UnstructuredMeshObject{B, D, P, T, NP}) where {B, D, P, T, NP} = community._m._N[]
# lengthCache(community::UnstructuredMeshObject{B, D, P, T, NP}) where {B, D, P, T, NP} = community._m._NCache[]
# Base.size(community::UnstructuredMeshObject{B, D, P, T, NP}) where {B, D, P, T, NP} = (NP, length(community))

# # Base.axes(community::UnstructuredMeshObject{B, D, P, T, NP}) where {B, D, P, T, NP} = (tuple(i for (i, dt) in enumerate(T) if eltype(dt) <: AbstractFloat), 1:N)
# Base.ndims(community::UnstructuredMeshObject) = 2
# Base.ndims(community::Type{<:UnstructuredMeshObject}) = 2
# Base.IndexStyle(UnstructuredMeshObject::UnstructuredMeshObject) = Base.IndexStyle(typeof(UnstructuredMeshObject))
# Base.IndexStyle(::Type{<:UnstructuredMeshObject}) = IndexCartesian()
# Base.CartesianIndices(community::UnstructuredMeshObject{B, D, P, T, NP}) where {B, D, P, T, NP} =
#     CartesianIndices((NP, N))
# Base.getindex(cp::UnstructuredMeshObject, i::Symbol, j::Int) = cp._pa[i][j]
# Base.getindex(cp::UnstructuredMeshObject, i::Int, j::Int) = cp._pa[i][j]
# Base.getindex(cp::UnstructuredMeshObject, i::Symbol, :) = cp._pa[i]
# Base.getindex(cp::UnstructuredMeshObject, i::Int, :) = cp._pa[i]
# Base.getindex(cp::UnstructuredMeshObject{B, D, P, T, NP}, :, j::Int) where {B, D, P, T, NP} = NamedTuple{P}(
#     (cp._pa[k][j] for k in keys(cp._pa))
# )
# Base.setindex!(cp::UnstructuredMeshObject, value, i::Symbol, j::Int) = (cp._pa[i][j] = value)
# Base.setindex!(cp::UnstructuredMeshObject, value, i::Int, j::Int) = (cp._pa[i][j] = value)
# Base.setindex!(cp::UnstructuredMeshObject, value, index::CartesianIndex{2}) = (cp._pa[index[1]][index[2]] = value)
# # Base.keys(cp::UnstructuredMeshObject) = keys(cp._pa)

# function Base.getindex(community::UnstructuredMeshObject, i::Integer)
#     community._pa[i]
# end

# @generated function Base.getproperty(community::UnstructuredMeshObject{B, D, P, T, NP}, s::Symbol) where {B, D, P, T, NP}
#     names = P
#     # build a clause for each fieldname in T
#     cases = [
#         :(s === $(QuoteNode(name)) && return @views getfield(getfield(community, :_pa), $(QuoteNode(name)))[1:length(community)])
#         for name in names
#     ]

#     quote
#         if s === :_pa; return getfield(community, :_pa); end
#         if s === :_m; return getfield(community, :_m); end
#         if s === :_paCopy; return getfield(community, :_paCopy); end
#         $(cases...)
#         error("Unknown property: $s for of the UnstructuredMeshObject.")
#     end
# end

# ## broadcasting

# struct CommunityPointStyle{N} <: Broadcast.AbstractArrayStyle{N} end

# # Allow constructing the style from Val or Int
# CommunityPointStyle(::Val{N}) where {N} = CommunityPointStyle{N}()
# CommunityPointStyle(N::Int) = CommunityPointStyle{N}()

# # Your UnstructuredMeshObject acts like a 2D array
# Base.BroadcastStyle(::Type{<:UnstructuredMeshObject}) = CommunityPointStyle{2}()

# # Combine styles safely
# Base.Broadcast.result_style(::CommunityPointStyle{M}) where {M} =
#     CommunityPointStyle{M}()
# Base.Broadcast.result_style(::CommunityPointStyle{M}, ::CommunityPointStyle{N}) where {M,N} =
#     CommunityPointStyle{max(M,N)}()
# Base.Broadcast.result_style(::CommunityPointStyle{M}, ::Base.Broadcast.AbstractArrayStyle{N}) where {M,N} =
#     CommunityPointStyle{max(M,N)}()
# Base.Broadcast.result_style(::Base.Broadcast.AbstractArrayStyle{M}, ::CommunityPointStyle{N}) where {M,N} =
#     CommunityPointStyle{max(M,N)}()

# Broadcast.broadcastable(x::UnstructuredMeshObject) = x

# ## Copyto
# @eval @inline function Base.copy(dest::UnstructuredMeshObject{B, D, P, T, NP}) where {B, D, P, T, NP}

#     deepcopy(dest)

# end

# @eval @inline function Base.copy(dest::UnstructuredMeshObject{B, D, P, T, NP}, subset::NTuple{P2, Symbol}) where {B, D, P, T, NP, P2}

#     for s in subset
#         if !(s in P)
#             error("Property $s not found in UnstructuredMeshObject.")
#         end
#     end

#     UnstructuredMeshObject{B, D, P, T, NP}(
#         NamedTuple{P}(
#             i in names(subset) ? copy(dest._pa[i]) : dest._pa[i] for i in P
#         ),
#         dest._m,
#         SizedVector{NP, Bool}([i in names(subset) ? true : false for i in P])    # Dictionary to hold agent properties for copying
#     )

# end

# @eval @inline function Base.copyto!(dest::UnstructuredMeshObject{B, D, P, T, NP},
#         bc::UnstructuredMeshObject{B, D, P, T, NP}) where {B, D, P, T, NP}
#     N = length(dest)
#     @inbounds for i in 1:NP
#         if bc._paCopy[i]
#             @views dest._pa[i][1:N] .= bc._pa[i][1:N]
#         end
#     end
#     dest
# end

# for type in [
#         Broadcast.Broadcasted{<:UnstructuredMeshObject},
#         Broadcast.Broadcasted{<:CommunityPointStyle},
#     ]

#     @eval @inline function Base.copyto!(dest::UnstructuredMeshObject{B, D, P, T, NP},
#             bc::$type) where {B, D, P, T, NP}
#         bc = Broadcast.flatten(bc)
#         N = length(dest)
#         @inbounds for i in 1:NP
#             if dest._paCopy[i]
#                 dest_ = @views dest._pa[i][1:N]
#                 copyto!(dest_, unpack_voa(bc, i))
#             end
#         end
#         dest
#     end
# end

# # # drop axes because it is easier to recompute
# @inline function unpack_voa(bc::Broadcast.Broadcasted{<:UnstructuredMeshObject}, i)
#     Broadcast.Broadcasted(bc.f, unpack_args_voa(i, bc.args))
# end

# function unpack_voa(x::UnstructuredMeshObject{B, D, P, T, NP}, i) where {B, D, P, T, NP}
#     @views x._pa[i][1:length(x)]
# end

# # Integrator
# function Base.eltype(::Type{<:UnstructuredMeshObject{B, D, P, T, NP}}) where {B, D, P, T, NP}
#     for i in T.parameters
#         if i <: AbstractFloat
#             return eltype(i)
#         end
#     end
#     return Float64
# end

# function Base.zero(community::UnstructuredMeshObject{B, D, P, T, NP}) where {B, D, P, T, NP}

#     communityZero = UnstructuredMeshObject{B, D, P, T, NP}(
#             NamedTuple{P, T}(
#                 c ? similar(dt, eltype(dt)) : dt for (dt, c) in zip(values(community._pa), community._paCopy)               
#             ),
#             community._m,
#             community._paCopy
#         )

#     N = length(community)
#     for (dt, dtOld, c) in zip(values(communityZero._pa), values(community._pa), community._paCopy)
#         if c
#             @views dt[1:N] .= dtOld[1:N]
#         end
#     end
    
#     return communityZero
# end

# function setCopyParameters(community::UnstructuredMeshObject{B, D, P, T, NP}, params::NTuple{P2, Symbol}) where {B, D, P, T, NP, P2}

#     for s in params
#         if !(s in P)
#             error("Property $s not found in UnstructuredMeshObject.")
#         end
#     end

#     _paCopy = [i in params ? true : false for i in P]

#     UnstructuredMeshObject{B, D, P, T, NP}(
#         community._pa,
#         community._m,
#         _paCopy
#     )
# end

# function addCopyParameter!(community::UnstructuredMeshObject{B, D, P, T, NP}, param::Symbol) where {B, D, P, T, NP}

#     if !(param in P)
#         error("Property $param not found in UnstructuredMeshObject.")
#     end

#     idx = findfirst(==(param), P)
#     _paCopy = [j || i == idx ? true : false for (i,j) in enumerate(P)]

#     community = UnstructuredMeshObject{B, D, P, T, NP}(
#         community._pa,
#         community._m,
#         _paCopy
#     )

#     return
# end

# function Base.iterate(community::UnstructuredMeshObject, state = (1, length(community)))
#     state[1] >= state[2] ? nothing : (state, (state[1] + 1, state[2]))
# end

# ######################################################################################################
# # Iterator
# ######################################################################################################
# struct CommunityPointIterator{B}
#     N::Int
# end

# function loopOverAgents(community::UnstructuredMeshObject{B, D, P, T, NP}) where {B, D, P, T, NP}
#     CommunityPointIterator{B}(length(community))
# end

# function Base.iterate(iterator::CommunityPointIterator, state = 1)
#     state >= iterator.N + 1 ? nothing : (state, state + 1)
# end

# # Necessary for working with Threads
# Base.firstindex(iterator::CommunityPointIterator) = 1
# Base.lastindex(iterator::CommunityPointIterator) = iterator.N
# Base.length(iterator::CommunityPointIterator) = iterator.N
# Base.getindex(iterator::CommunityPointIterator, i::Int) = i


# ######################################################################################################
# # Kernel functions
# ######################################################################################################
# function removeAgent!(community::UnstructuredMeshObject, pos::Int)
#     if pos < 1 || pos > length(community)
#         @warn "Position $pos is out of bounds for UnstructuredMeshObject with N=$(length(community)). No agent removed."
#     else
#         community._m._flagsRemoved[pos] = true
#     end
#     return
# end

# @generated function addAgent!(community::UnstructuredMeshObject{<:UnstructuredMeshObjectMeta{TR, S}, D, P, T, NP}, kwargs::NamedTuple{P2, T2}) where {TR, S<:Threads.Atomic, D, P, T, NP, P2, T2}

#     for i in P2
#         if !(i in P)
#             error("Property $i not found in UnstructuredMeshObject. Properties that have to be provided to addAgent! are: $(P).")
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
#         newPos = Threads.atomic_add!(community._m._NNew, 1) + 1
#         if newPos > community._m._NCache[]
#             community._m._overflowFlag[] = true
#             return
#         else
#             newId = Threads.atomic_add!(community._m._idMax, 1) + 1
#             community._m._id[newPos] = newId
#             $(cases...)
#         end
#     end

# end