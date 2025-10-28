######################################################################################################
# AGENT STRUCTURE
######################################################################################################
struct UnstructuredMesh{D, S, PA, PN, PE, PF, PV}

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

function UnstructuredMesh(
    dims::Int;
    propertiesAgent::Union{NamedTuple, Nothing}=nothing,
    propertiesNode::Union{NamedTuple, Nothing}=nothing,
    propertiesEdge::Union{NamedTuple, Nothing}=nothing,
    propertiesFace::Union{NamedTuple, Nothing}=nothing,
    propertiesVolume::Union{NamedTuple, Nothing}=nothing,
    scopePosition::Symbol = :propertiesAgent,
)

    if !(scopePosition in fieldnames(UnstructuredMesh))
        error("scopePosition must be one of $(fieldnames(UnstructuredMesh)). Found $scopePosition")
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

    return UnstructuredMesh{dims, scope, (typeof(i) for i in properties)...}(
        properties...
    )
end

spatialDims(x::UnstructuredMesh{D}) where {D} = D
scopePosition(x::UnstructuredMesh{D, S}) where {D, S} = S

function Base.show(io::IO, x::UnstructuredMesh{D}) where {D}
    println(io, "UnstructuredMesh with dimensions $(D): \n")
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

function Base.show(io::IO, ::Type{UnstructuredMesh{D, S, PA, PN, PE, PF, PV}}) where {D, S, PA, PN, PE, PF, PV}
    println(io, "UnstructuredMesh{dims=", D, ", scopePosition=", S, ",")
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
# UnstructuredMeshObjectField
######################################################################################################
struct UnstructuredMeshObjectField{
            TR, IDVI, IDAI,    
            AI, VB, SI, VVNT, AB,
            PR, PRN, PRC 
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

    _p::PR
    _NP::Int
    _pReference::PRC
end
Adapt.@adapt_structure UnstructuredMeshObjectField

function UnstructuredMeshObjectField(
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

    _N = Threads.Atomic{Int}(N)
    _NCache = Threads.Atomic{Int}(NCache)
    _FlagsRemoved = Vector{Int}(zeros(Int, NCache))
    _NRemoved = Threads.Atomic{Int}(0)
    _NRemovedThread = SizedVector{Threads.nthreads(), Int}(zeros(Int, Threads.nthreads()))
    _NAdded = Threads.Atomic{Int}(0)
    _NAddedThread = SizedVector{Threads.nthreads(), Int}(zeros(Int, Threads.nthreads()))
    P = keys(meshProperties)
    T = Tuple{(dtype(i, isbits=true) for i in values(meshProperties))...}
    _AddedAgents = [Vector{NamedTuple{P, T}}() for _ in 1:Threads.nthreads()]
    _FlagOverflow = Threads.Atomic{Bool}(false)

    _p = NamedTuple{keys(meshProperties)}(zeros(dtype(dt, isbits=true), NCache) for dt in values(meshProperties))
    _NP = length(meshProperties)
    _pReference = SizedVector{length(_p), Bool}([true for _ in 1:length(meshProperties)])

    TR = Threads.nthreads() > 1 ? true : false
    IDVI = typeof(_id)
    IDAI = typeof(_idMax)

    AI = typeof(_N)
    AB = typeof(_FlagOverflow)
    VB = typeof(_FlagsRemoved)
    SI = typeof(_NRemovedThread)
    VVNT = typeof(_AddedAgents)

    PR = typeof(_p)
    PRN = _NP
    PRC = typeof(_pReference)

    UnstructuredMeshObjectField{
            TR, IDVI, IDAI,    
            AI, VB, SI, VVNT, AB,
            PR, PRN, PRC 
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

            _p,
            _NP,
            _pReference,
        )
end

function UnstructuredMeshObjectField(
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

            _p,
            _NP,
            _pReference
    )

    TR = Threads.nthreads() > 1 ? true : false
    IDVI = typeof(_id)
    IDAI = typeof(_idMax)

    AI = typeof(_N)
    AB = typeof(_FlagOverflow)
    VB = typeof(_FlagsRemoved)
    SI = typeof(_NRemovedThread)
    VVNT = typeof(_AddedAgents)

    PR = typeof(_p)
    PRN = _NP
    PRC = typeof(_pReference)

    UnstructuredMeshObjectField{
            TR, IDVI, IDAI,    
            AI, VB, SI, VVNT, AB,
            PR, PRN, PRC 
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

            _p,
            _NP,
            _pReference
        )
end

function Base.show(io::IO, x::UnstructuredMeshObjectField{
            TR, IDVI, IDAI,    
            AI, VB, SI, VVNT, AB,
            PR, PRN, PRC 
        }) where {
            TR, IDVI, IDAI,    
            AI, VB, SI, VVNT, AB,
            PR, PRN, PRC 
        } 
    
    println(io, "UnstructuredMeshObjectField: \n")
    println(io, @sprintf("\t%-25s %-15s", "Property", "DataType"))
    println(io, "\t" * repeat("-", 40))
    println(io, @sprintf("\t%-25s %-15s", "_id", IDVI))
    println(io, @sprintf("\t%-25s %-15s", "_idMax", IDAI))
    println(io, @sprintf("\t%-25s %-15s", "_N", AI))
    println(io, @sprintf("\t%-25s %-15s", "_NCache", AI))
    println(io, @sprintf("\t%-25s %-15s", "_FlagsRemoved", VB))
    println(io, @sprintf("\t%-25s %-15s", "_NRemoved", AI))
    println(io, @sprintf("\t%-25s %-15s", "_NRemovedThread", SI))
    println(io, @sprintf("\t%-25s %-15s", "_NAdded", AI))
    println(io, @sprintf("\t%-25s %-15s", "_NAddedThread", SI))
    println(io, @sprintf("\t%-25s %-15s", "_AddedAgents", VVNT))
    println(io, @sprintf("\t%-25s %-15s", "_FlagOverflow", AB))

    for ((n, t), c) in zip(pairs(x._p), x._pReference)
        if c
            println(io, @sprintf("\t%-25s %-15s", string("*p.",n), typeof(t)))
        else
            println(io, @sprintf("\t%-25s %-15s", string("p.",n),  typeof(t)))
        end
    end

end

function Base.show(io::IO, ::Type{UnstructuredMeshObjectField})
    println(io, "UnstructuredMeshObjectField{")
    CellBasedModels.show(io, x)
    println(io, "}")
end

function show(io::IO, x::Type{UnstructuredMeshObjectField{
            TR, IDVI, IDAI,
            AI, VB, SI, VVNT, AB,
            PR, PRN, PRC
        }}) where {
            TR, IDVI, IDAI,
            AI, VB, SI, VVNT, AB,
            PR, PRN, PRC
        } 
    print(io, "_id::", IDVI, ", ")
    print(io, "_idMax::", IDAI, ", ")
    print(io, "_N::", AI, ", ")
    print(io, "_NCache::", AI, ", ")
    print(io, "_FlagsRemoved::", VB, ", ")
    print(io, "_NRemoved::", AI, ", ")
    print(io, "_NRemovedThread::", SI, ", ")
    print(io, "_NAdded::", AI, ", ")
    print(io, "_NAddedThread::", SI, ", ")
    print(io, "_AddedAgents::", VVNT, ", ")
    print(io, "_FlagOverflow::", AB, ", ")
    for (n, t) in zip(PR.parameters[1], PR.parameters[2].parameters)
        print(io, "p.", string(n), "::", t, ", ")
    end
end

@generated function Base.getproperty(field::UnstructuredMeshObjectField{
            TR, IDVI, IDAI,
            AI, VB, SI, VVNT, AB,
            PR, PRN, PRC
        }, s::Symbol) where {
            TR<:Any, IDVI<:Any, IDAI<:Any,
            AI<:Any, VB<:Any, SI<:Any, VVNT<:Any, AB<:Any,
            PR, PRN<:Any, PRC<:Any
        }
    # build a clause for each fieldname in T
    general = [
        :(if s === $(QuoteNode(name)); return getfield(field, $(QuoteNode(name))); end)
        for name in fieldnames(UnstructuredMeshObjectField)
    ]
    cases = [
        :(s === $(QuoteNode(name)) && return @views getfield(getfield(field, :_p), $(QuoteNode(name)))[1:length(field)])
        for name in PR.parameters[1]
    ]

    quote
        $(general...)
        $(cases...)
        error("Unknown property: $s for of the UnstructuredMeshObject.")
    end
end

Base.length(field::UnstructuredMeshObjectField{TR, IDVI, IDAI, AI}) where {TR, IDVI, IDAI, AI<:Threads.Atomic} = field._N[]
Base.size(field::UnstructuredMeshObjectField) = (field._NP, length(field))

struct UnstructuredMeshObjectFieldStyle{N} <: Broadcast.AbstractArrayStyle{N} end

# Allow constructing the style from Val or Int
UnstructuredMeshObjectFieldStyle(::Val{N}) where {N} = UnstructuredMeshObjectFieldStyle{N}()
UnstructuredMeshObjectFieldStyle(N::Int) = UnstructuredMeshObjectFieldStyle{N}()

# Your UnstructuredMeshObjectField acts like a 2D array
Base.BroadcastStyle(::Type{<:UnstructuredMeshObjectField}) = UnstructuredMeshObjectFieldStyle{2}()

# Combine styles safely
Base.Broadcast.result_style(::UnstructuredMeshObjectFieldStyle{M}) where {M} =
    UnstructuredMeshObjectFieldStyle{M}()
Base.Broadcast.result_style(::UnstructuredMeshObjectFieldStyle{M}, ::UnstructuredMeshObjectFieldStyle{N}) where {M,N} =
    UnstructuredMeshObjectFieldStyle{max(M,N)}()
Base.Broadcast.result_style(::UnstructuredMeshObjectFieldStyle{M}, ::Base.Broadcast.AbstractArrayStyle{N}) where {M,N} =
    UnstructuredMeshObjectFieldStyle{max(M,N)}()
Base.Broadcast.result_style(::Base.Broadcast.AbstractArrayStyle{M}, ::UnstructuredMeshObjectFieldStyle{N}) where {M,N} =
    UnstructuredMeshObjectFieldStyle{max(M,N)}()

Broadcast.broadcastable(x::UnstructuredMeshObjectField) = x

## Copy
function Base.copy(field::UnstructuredMeshObjectField)

    return UnstructuredMeshObjectField(
        field._id,
        field._idMax,

        field._N,
        field._NCache,
        field._FlagsRemoved,
        field._NRemoved,
        field._NRemovedThread,
        field._NAdded,
        field._NAddedThread,
        field._AddedAgents,
        field._FlagOverflow,

        NamedTuple{keys(field._p)}(
            r ? p : copy(p) for (p, r) in zip(values(field._p), field._pReference)
        ),
        field._NP,
        field._pReference,
    )

end

## Zeros
function Base.zero(field::UnstructuredMeshObjectField{
            TR, IDVI, IDAI,
            AI, VB, SI, VVNT, AB,
            PR, PRN, PRC
        }) where {
            TR, IDVI, IDAI,
            AI, VB, SI, VVNT, AB,
            PR, PRN, PRC
        } 

    UnstructuredMeshObjectField(
        field._id,
        field._idMax,

        field._N,
        field._NCache,
        field._FlagsRemoved,
        field._NRemoved,
        field._NRemovedThread,
        field._NAdded,
        field._NAddedThread,
        field._AddedAgents,
        field._FlagOverflow,

        NamedTuple{keys(field._p)}(
            r ? p : zero(p) for (p, r) in zip(values(field._p), field._pReference)
        ),
        field._NP,
        field._pReference,
    )
end

## Copyto!
@eval @inline function Base.copyto!(
    dest::UnstructuredMeshObjectField{TR, IDVI, IDAI, AI, VB, SI, VVNT, AB, PR, PRN, PRC},
    bc::UnstructuredMeshObjectField{TR, IDVI, IDAI, AI, VB, SI, VVNT, AB, PR, PRN, PRC}) where {TR, IDVI, IDAI, AI, VB, SI, VVNT, AB, PR, PRN, PRC}
    N = length(dest)
    @inbounds for i in 1:PRN
        if !bc._pReference[i]
            @views dest._p[i][1:N] .= bc._p[i][1:N]
        end
    end
    dest
end

for type in [
        Broadcast.Broadcasted{<:UnstructuredMeshObjectField},
        Broadcast.Broadcasted{<:UnstructuredMeshObjectFieldStyle},
    ]

    @eval @inline function Base.copyto!(
            dest::UnstructuredMeshObjectField{TR, IDVI, IDAI, AI, VB, SI, VVNT, AB, PR, PRN, PRC},
            bc::$type) where {TR, IDVI, IDAI, AI, VB, SI, VVNT, AB, PR, PRN, PRC}
        bc = Broadcast.flatten(bc)
        N = length(dest)
        @inbounds for i in 1:PRN
            if !dest._pReference[i]
                dest_ = @views dest._p[i][1:N]
                copyto!(dest_, unpack_voa(bc, i))
            end
        end
        dest
    end
end

# # drop axes because it is easier to recompute
@inline function unpack_voa(bc::Broadcast.Broadcasted{<:UnstructuredMeshObjectField}, i)
    Broadcast.Broadcasted(bc.f, unpack_args_voa(i, bc.args))
end

function unpack_voa(x::UnstructuredMeshObjectField, i)
    @views x._p[i][1:length(x)]
end

######################################################################################################
# UnstructuredMeshObjectField
######################################################################################################
struct UnstructuredMeshObject{
            D, P, S,
            A, N, E, F, V
    }
    
    _scope::Int

    a::A
    n::N
    e::E
    f::F
    v::V
end
Adapt.@adapt_structure UnstructuredMeshObject

function UnstructuredMeshObject(
        mesh::UnstructuredMesh{D, S, PA, PN, PE, PF, PV};
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

    fields = []
    for (N,NC,p,name) in zip(
        (agentN, nodeN, edgeN, faceN, volumeN),
        (agentNCache, nodeNCache, edgeNCache, faceNCache, volumeNCache),
        (mesh.propertiesAgent, mesh.propertiesNode, mesh.propertiesEdge, mesh.propertiesFace, mesh.propertiesVolume),
        ("Agent", "Node", "Edge", "Face", "Volume"),
    )
        if N < 0
            error("$(name)N must be greater than 0. Found N=$N")
        end
        if NC === nothing
            NC = N
        elseif NC < N
            error("$(name)NCache must be greater than or equal to $(name)N. Found $(name)NCache=$NC and $(name)N=$N")
        end
        push!(fields, UnstructuredMeshObjectField(mesh.propertiesAgent, N = agentN, NCache = agentNCache))
    end
    
    return UnstructuredMeshObject{
            D, P, S,
            (typeof(i) for i in fields)...
        }(
            S, fields...
        )
end

function UnstructuredMeshObject(
            scope, a, n, e, f, v
    )

    D = length(filter(k -> k in (:x, :y, :z), keys(_pa)))
    P = keys(_pa)
    S = scope

    return UnstructuredMeshObject{
            D, P, S,
            typeof(a), typeof(n), typeof(e), typeof(f), typeof(v)
        }(
            scope, a, n, e, f, v
        )
end

function Base.show(io::IO, x::UnstructuredMeshObject{B, D, P}) where {B, D, P}
    println(io, "UnstructuredMeshObject $D D: \n")
    CellBasedModels.show(io, x)
    println(io)
end

function show(io::IO, x::UnstructuredMeshObject{B, D, P}) where {B, D, P}
    for (f,n,copy) in zip(
            (x.a, x.n, x.e, x.f, x.v),
            ("Agent Properties", "Node Properties", "Edge Properties", "Face Properties", "Volume Properties"),
        )
        if f !== nothing
            println(io, replace(n))
            println(io, @sprintf("\t%-15s %-15s", "Name", "DataType"))
            println(io, "\t" * repeat("-", 85))
            for ((name, par), c) in zip(pairs(f._p), f._pReference)
                println(io, @sprintf("\t%-15s %-15s", 
                    c ? string("*", name) : string(name),
                    typeof(par)))
            end
            println(io)
        end
    end
end

function Base.show(io::IO, x::Type{UnstructuredMeshObject{
            D, P, S, A, N, E, F, V,
        }}) where {
            D, P, S, A, N, E, F, V,
        }
    println(io, "UnstructuredMesh{dims=", D, ", processors=", P, ", scopePosition=", S,)
    CellBasedModels.show(io, x)
    println(io, "}")
end

function show(io::IO, x::Type{UnstructuredMeshObject{
            D, P, S, A, N, E, F, V,
        }}) where {
            D, P, S, A, N, E, F, V,
        }
    for ((propsmeta,props), propsnames) in zip(((A,), (N,), (E,), (F,), (V,)), ("PropertiesAgent", "PropertiesNode", "PropertiesEdge", "PropertiesFace", "PropertiesVolume"))
        if props !== Nothing
            print(io, "\t", string(propsnames), "Meta", "=(")
            # println(propsmeta)
            CellBasedModels.show(io, propsmeta)
            println(io, ")")
            print(io, "\t", string(propsnames), "=(")
            for (i, (n, t)) in enumerate(zip(props.parameters[1], props.parameters[2].parameters))
                i > 1 && print(io, ", ")
                print(io, string(n), "::", t)
            end
            println(io, ")")
        end
    end
end

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

# GETINDEX TO BE REMOVED
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

## broadcasting

# struct UnstructuredMeshObjectFieldStyle{N} <: Broadcast.AbstractArrayStyle{N} end

# # Allow constructing the style from Val or Int
# UnstructuredMeshObjectFieldStyle(::Val{N}) where {N} = UnstructuredMeshObjectFieldStyle{N}()
# UnstructuredMeshObjectFieldStyle(N::Int) = UnstructuredMeshObjectFieldStyle{N}()

# # Your UnstructuredMeshObject acts like a 2D array
# Base.BroadcastStyle(::Type{<:UnstructuredMeshObject}) = UnstructuredMeshObjectFieldStyle{2}()

# # Combine styles safely
# Base.Broadcast.result_style(::UnstructuredMeshObjectFieldStyle{M}) where {M} =
#     UnstructuredMeshObjectFieldStyle{M}()
# Base.Broadcast.result_style(::UnstructuredMeshObjectFieldStyle{M}, ::UnstructuredMeshObjectFieldStyle{N}) where {M,N} =
#     UnstructuredMeshObjectFieldStyle{max(M,N)}()
# Base.Broadcast.result_style(::UnstructuredMeshObjectFieldStyle{M}, ::Base.Broadcast.AbstractArrayStyle{N}) where {M,N} =
#     UnstructuredMeshObjectFieldStyle{max(M,N)}()
# Base.Broadcast.result_style(::Base.Broadcast.AbstractArrayStyle{M}, ::UnstructuredMeshObjectFieldStyle{N}) where {M,N} =
#     UnstructuredMeshObjectFieldStyle{max(M,N)}()

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
#         Broadcast.Broadcasted{<:UnstructuredMeshObjectFieldStyle},
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

# @generated function addAgent!(community::UnstructuredMeshObject{<:UnstructuredMeshObjectField{TR, S}, D, P, T, NP}, kwargs::NamedTuple{P2, T2}) where {TR, S<:Threads.Atomic, D, P, T, NP, P2, T2}

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