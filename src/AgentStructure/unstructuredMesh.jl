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
        MESHSCOPES,
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

    return UnstructuredMesh{dims, scopePosition, (typeof(i) for i in properties)...}(
        properties...
    )
end

const MESHSCOPES = (:propertiesAgent, :propertiesNode, :propertiesEdge, :propertiesFace, :propertiesVolume)
spatialDims(x::UnstructuredMesh{D}) where {D} = D
scope(x::UnstructuredMesh{D, S}) where {D, S} = S
scope2scopePosition(x::Symbol) = findfirst(==(x), MESHSCOPES)
scopePosition2scope(x::Int) = MESHSCOPES[x]

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
    for (props, propsnames) in zip((PA, PN, PE, PF, PV), MESHSCOPES)
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
            P, 
            PR, PRN, PRC,
            IDVI, IDAI,    
            AI, VB, SI, VVNT, AB            
        }
    _p::PR
    _NP::Int
    _pReference::PRC

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

    P = platform(_N)
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
            P, 
            PR, PRN, PRC, 
            IDVI, IDAI,    
            AI, VB, SI, VVNT, AB
        }(
            _p,
            _NP,
            _pReference,

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

function UnstructuredMeshObjectField(
            _p,
            _NP,
            _pReference,

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

    P = platform(_N)

    PR = typeof(_p)
    PRN = _NP
    PRC = typeof(_pReference)

    IDVI = typeof(_id)
    IDAI = typeof(_idMax)

    AI = typeof(_N)
    AB = typeof(_FlagOverflow)
    VB = typeof(_FlagsRemoved)
    SI = typeof(_NRemovedThread)
    VVNT = typeof(_AddedAgents)

    UnstructuredMeshObjectField{
            P, 
            PR, PRN, PRC,
            IDVI, IDAI,    
            AI, VB, SI, VVNT, AB,
        }(
            _p,
            _NP,
            _pReference,

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

function Base.show(io::IO, x::UnstructuredMeshObjectField{
            P, 
            PR, PRN, PRC, 
            IDVI, IDAI,    
            AI, VB, SI, VVNT, AB,
        }) where {
            P, 
            PR, PRN, PRC,
            IDVI, IDAI,    
            AI, VB, SI, VVNT, AB, 
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

function Base.show(io::IO, x::Type{UnstructuredMeshObjectField})
    println(io, "UnstructuredMeshObjectField{")
    # CellBasedModels.show(io, x)
    println(io, "}")
end

function show(io::IO, ::Type{UnstructuredMeshObjectField{
            P, 
            PR, PRN, PRC,
            IDVI, IDAI,
            AI, VB, SI, VVNT, AB,
        }}) where {
            P, 
            PR, PRN, PRC,
            IDVI, IDAI,
            AI, VB, SI, VVNT, AB,
        } 
    for (n, t) in zip(PR.parameters[1], PR.parameters[2].parameters)
        print(io, "p.", string(n), "::", t, ", ")
    end
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
end

@generated function Base.getproperty(field::UnstructuredMeshObjectField{
            P, 
            PR, PRN, PRC,
            IDVI, IDAI,
            AI, VB, SI, VVNT, AB,            
        }, s::Symbol) where {
            P, 
            PR, PRN, PRC,
            IDVI, IDAI,
            AI, VB, SI, VVNT, AB,
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

Base.length(field::UnstructuredMeshObjectField{P}) where {P<:CPU} = field._N[]
Base.size(field::UnstructuredMeshObjectField) = (field._NP, length(field))

Base.getindex(field::UnstructuredMeshObjectField, i::Int) = field._p[i]
Base.getindex(field::UnstructuredMeshObjectField, s::Symbol) = getproperty(field, s)

## Copy
function Base.copy(field::UnstructuredMeshObjectField)

    UnstructuredMeshObjectField(
        NamedTuple{keys(field._p)}(
            r ? p : copy(p) for (p, r) in zip(values(field._p), field._pReference)
        ),
        field._NP,
        field._pReference,

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
    )

end

## Zero
function Base.zero(field::UnstructuredMeshObjectField)

    UnstructuredMeshObjectField(
        NamedTuple{keys(field._p)}(
            r ? p : zero(p) for (p, r) in zip(values(field._p), field._pReference)
        ),
        field._NP,
        field._pReference,

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
    )
end

## Copyto!
@eval @inline function Base.copyto!(
    dest::UnstructuredMeshObjectField{P, PR, PRN, PRC},
    bc::UnstructuredMeshObjectField{P, PR, PRN, PRC}) where {P, PR, PRN, PRC}
    N = length(dest)
    @inbounds for i in 1:PRN
        if !bc._pReference[i]
            @views dest._p[i][1:N] .= bc._p[i][1:N]
        end
    end
    dest
end

## Broadcasting
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

for type in [
        Broadcast.Broadcasted{<:UnstructuredMeshObjectField},
        Broadcast.Broadcasted{<:UnstructuredMeshObjectFieldStyle},
    ]

    @eval @inline function Base.copyto!(
            dest::UnstructuredMeshObjectField{P, PR, PRN, PRC},
            bc::$type) where {P, PR, PRN, PRC}
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
            P, D, S,
            A, N, E, F, V
    }
    _scopePos::Int
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

    fields = []
    P = platform()
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
        field = UnstructuredMeshObjectField(p, N = N, NCache = NC)
        push!(fields, field)
    end

    scopePosition = scope2scopePosition(S)

    return UnstructuredMeshObject{
            P, D, S,
            (typeof(i) for i in fields)...
        }(
            scopePosition, fields...
        )
end

function UnstructuredMeshObject(
            scopePos, a, n, e, f, v
    )

    D = length(filter(k -> k in (:x, :y, :z), keys((agentProperties=a, nodeProperties=n, edgeProperties=e, faceProperties=f, volumeProperties=v)[scopePos]._p)))
    P = platform()
    for i in (a, n, e, f, v)
        if i !== nothing
            P = platform(i._N)
            break
        end
    end
    S = scopePosition2scope(scopePos)

    return UnstructuredMeshObject{
            P, D, S,
            typeof(a), typeof(n), typeof(e), typeof(f), typeof(v)
        }(
            scopePos, a, n, e, f, v
        )
end

function Base.show(io::IO, x::UnstructuredMeshObject{P, D, S}) where {P, D, S}
    println(io, "UnstructuredMeshObject platform=$P dimensions=$D scopePosition=$S: \n")
    CellBasedModels.show(io, x)
    println(io)
end

function show(io::IO, x::UnstructuredMeshObject{P, D, S}) where {P, D, S}
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

# function Base.show(io::IO, x::Type{UnstructuredMeshObject{
#             P, D, S, A, N, E, F, V,
#         }}) where {
#             P, D, S, A, N, E, F, V,
#         }
#     println(io, "UnstructuredMesh{platform=", P, ", dimension=", D, ", scopePosition=", S,)
#     CellBasedModels.show(io, x)
#     println(io, "}")
# end

# function show(io::IO, ::Type{UnstructuredMeshObject{
#             P, D, S, A, N, E, F, V,
#         }}) where {
#             P, D, S, A, N, E, F, V,
#         }
#     for ((propsmeta,props), propsnames) in zip(((A,), (N,), (E,), (F,), (V,)), ("PropertiesAgent", "PropertiesNode", "PropertiesEdge", "PropertiesFace", "PropertiesVolume"))
#         if props !== Nothing
#             print(io, "\t", string(propsnames), "Meta", "=(")
#             # println(propsmeta)
#             CellBasedModels.show(io, propsmeta)
#             println(io, ")")
#             print(io, "\t", string(propsnames), "=(")
#             for (i, (n, t)) in enumerate(zip(props.parameters[1], props.parameters[2].parameters))
#                 i > 1 && print(io, ", ")
#                 print(io, string(n), "::", t)
#             end
#             println(io, ")")
#         end
#     end
# end

function Base.length(::UnstructuredMeshObject{P, D, S, A, N, E, F, V}) where {P, D, S, A, N, E, F, V}
    c = 0
    A !== nothing ? c += 1 : nothing
    N !== nothing ? c += 1 : nothing
    E !== nothing ? c += 1 : nothing
    F !== nothing ? c += 1 : nothing
    V !== nothing ? c += 1 : nothing
    return c
end    

Base.size(mesh::UnstructuredMeshObject) = (length(mesh),)

Base.ndims(::UnstructuredMeshObject) = 1
Base.ndims(::Type{<:UnstructuredMeshObject}) = 1

function Base.eltype(::Type{<:UnstructuredMeshObject{P, D, S, A, N, E, F, V}}) where {P, D, S, A, N, E, F, V}
    return CellBasedModels.concreteDataType(AbstractFloat)
end

function Base.iterate(mesh::UnstructuredMeshObject, state = 1)
    state >= length(mesh) ? nothing : (state, state + 1)
end

scope(::UnstructuredMeshObject{P, D, S}) where {P, D, S} = S
scopePosition(x::UnstructuredMeshObject) = x._scopePos

#Getindex
Base.getindex(community::UnstructuredMeshObject, i::Integer) =
    getfield(community, (:a, :n, :e, :f, :v)[i])
Base.getindex(community::UnstructuredMeshObject, s::Symbol) =
    getfield(community, s)

## Copy
function Base.copy(field::UnstructuredMeshObject)

    UnstructuredMeshObject(
        field._scopePos,
        field.a === nothing ? nothing : copy(field.a),
        field.n === nothing ? nothing : copy(field.n),
        field.e === nothing ? nothing : copy(field.e),
        field.f === nothing ? nothing : copy(field.f),
        field.v === nothing ? nothing : copy(field.v),
    )

end

## Zero
function Base.zero(field::UnstructuredMeshObject)

    UnstructuredMeshObject(
        field._scopePos,
        field.a === nothing ? nothing : zero(field.a),
        field.n === nothing ? nothing : zero(field.n),
        field.e === nothing ? nothing : zero(field.e),
        field.f === nothing ? nothing : zero(field.f),
        field.v === nothing ? nothing : zero(field.v),
    )

end

## Copyto!
@eval @inline function Base.copyto!(
    dest::UnstructuredMeshObject,
    bc::UnstructuredMeshObject)
    @inbounds for i in (:a, :n, :e, :f, :v)
        if dest[i] !== nothing && bc[i] !== nothing
            copyto!(dest[i], bc[i])
        end
    end
    dest
end

## Broadcasting
struct UnstructuredMeshObjectStyle{N} <: Broadcast.AbstractArrayStyle{N} end

# Allow constructing the style from Val or Int
UnstructuredMeshObjectStyle(::Val{N}) where {N} = UnstructuredMeshObjectStyle{N}()
UnstructuredMeshObjectStyle(N::Int) = UnstructuredMeshObjectStyle{N}()

# Your UnstructuredMeshObject acts like a 2D array
Base.BroadcastStyle(::Type{<:UnstructuredMeshObject}) = UnstructuredMeshObjectStyle{1}()

# Combine styles safely
Base.Broadcast.result_style(::UnstructuredMeshObjectStyle{M}) where {M} =
    UnstructuredMeshObjectStyle{M}()
Base.Broadcast.result_style(::UnstructuredMeshObjectStyle{M}, ::UnstructuredMeshObjectStyle{N}) where {M,N} =
    UnstructuredMeshObjectStyle{max(M,N)}()
Base.Broadcast.result_style(::UnstructuredMeshObjectStyle{M}, ::Base.Broadcast.AbstractArrayStyle{N}) where {M,N} =
    UnstructuredMeshObjectStyle{max(M,N)}()
Base.Broadcast.result_style(::Base.Broadcast.AbstractArrayStyle{M}, ::UnstructuredMeshObjectStyle{N}) where {M,N} =
    UnstructuredMeshObjectStyle{max(M,N)}()

Broadcast.broadcastable(x::UnstructuredMeshObject) = x

for type in [
        Broadcast.Broadcasted{<:UnstructuredMeshObject},
        Broadcast.Broadcasted{<:UnstructuredMeshObjectStyle},
    ]

    @eval @inline function Base.copyto!(
            dest::UnstructuredMeshObject{P, A, N, E, F, V},
            bc::$type) where {P, A, N, E, F, V}
        bc = Broadcast.flatten(bc)
        @inbounds for i in 1:length(dest)
            d = dest[i]
            if d !== nothing
                np, n = size(d)
                for j in 1:np
                    if !d._pReference[j]
                        dest_ = @views dest[i]._p[j][1:n]
                        copyto!(dest_, unpack_voa(bc, i, j, n))
                    end
                end
            end
        end
        dest
    end
end

#Specialized unpacking
@inline function unpack_voa(bc::Broadcast.Broadcasted{<:UnstructuredMeshObject}, i, j, n)
    Broadcast.Broadcasted(bc.f, unpack_args_voa(i, j, n, bc.args))
end
function unpack_voa(x::UnstructuredMeshObject, i, j, n)
    @views x[i][j][1:n]
end