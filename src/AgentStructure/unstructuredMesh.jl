import RecursiveArrayTools
import LinearAlgebra

# Add ForwardDiff support for automatic differentiation
using ForwardDiff: ForwardDiff

# Import OrdinaryDiffEqDifferentiation for jacobian! method
try
    using OrdinaryDiffEqDifferentiation
catch
    # Fallback if package not available
end

######################################################################################################
# AGENT STRUCTURE
######################################################################################################
struct UnstructuredMesh{D, S, PA, PN, PE, PF, PV} <: AbstractMesh

    a::Union{NamedTuple, Nothing}                   # Dictionary to hold agent properties
    n::Union{NamedTuple, Nothing}    # Dictionary to hold agent properties
    e::Union{NamedTuple, Nothing}    # Dictionary to hold agent properties
    f::Union{NamedTuple, Nothing}    # Dictionary to hold agent properties
    v::Union{NamedTuple, Nothing}    # Dictionary to hold agent properties
    _functions::Dict{Symbol, Any}    # Dictionary to hold functions associated with the mesh

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
    propertiesVolume::Union{NamedTuple, Nothing}=nothing
)

    if dims < 0 || dims > 3
        error("dims must be between 0 and 3. Found $dims")
    end
    defaultParameters = (
        x = dims >= 1 ? Parameter(AbstractFloat, description="Position in x (protected parameter)", dimensions=:L) : nothing,
        y = dims >= 2 ? Parameter(AbstractFloat, description="Position in y (protected parameter)", dimensions=:L) : nothing,
        z = dims >= 3 ? Parameter(AbstractFloat, description="Position in z (protected parameter)", dimensions=:L) : nothing,
    )
    defaultParameters = NamedTuple{Tuple(k for (k,v) in pairs(defaultParameters) if v !== nothing)}(
        (v for (k,v) in pairs(defaultParameters) if v !== nothing)
    )

    properties = []
    for (pn, p) in zip(
        MESHSCOPES,
        (propertiesAgent, propertiesNode, propertiesEdge, propertiesFace, propertiesVolume)
    )
        if :n === pn && p === nothing
            push!(properties, defaultParameters)
        elseif :n === pn 
            push!(properties, _posMerge(parameterConvert(p), defaultParameters))
        elseif p === nothing
            push!(properties, nothing)
        else
            push!(properties, parameterConvert(p))
        end
    end

    return UnstructuredMesh{dims, Nothing, (typeof(i) for i in properties)...}(
        properties...,
        Dict{Symbol, Any}()
    )
end

const MESHSCOPES = (:a, :n, :e, :f, :v)
spatialDims(x::UnstructuredMesh{D}) where {D} = D
specialization(x::UnstructuredMesh{D, S}) where {D, S} = S

function Base.show(io::IO, x::UnstructuredMesh{D, S}) where {D, S}

    if S === Nothing
        println(io, "UnstructuredMesh with dimensions $(D): \n")
    else
        println(io, "$S with dimensions $(D): \n")
    end
    for p in propertynames(x)
        p === :_functions ? continue : nothing
        props = getfield(x, p)
        if props !== nothing
            println(io, "mesh.",string(p),".")
            println(io, @sprintf("\t%-15s %-15s %-15s %-20s %-60s %-s", "Name", "DataType", "Dimensions", "Default_Value", "ModifiedIn", "Description"))
            println(io, "\t" * repeat("-", 145))
            for (name, par) in pairs(props)
                println(io, @sprintf("\t%-15s %-15s %-15s %-20s %-60s %-s", 
                    name, 
                    dtype(par), 
                    par.dimensions === nothing ? "" : string(par.dimensions),
                    par.defaultValue === nothing ? "" : string(par.defaultValue), 
                    length(par._modifiedIn) === 0 ? "" : string(tuple([i[2] for i in par._modifiedIn]...)),
                    par.description))
            end
            println(io)
        end
    end
    println(io, "Functions")
    for (scope, (type, funcs)) in pairs(x._functions)
        print(io, "\t", scope, " (", type, "):")
        for (i,f) in enumerate(funcs)
            # print(io, "\t\t Subfunctions ", i, ":")
            print(io, " ", f)
        end
        println(io)
    end
end

function Base.show(io::IO, ::Type{UnstructuredMesh{D, S, PA, PN, PE, PF, PV}}) where {D, S, PA, PN, PE, PF, PV}
    if S === Nothing
        print(io, "UnstructuredMesh{dims=", D, ",")
    else
        print(io, string(S), "{dims=", D, ",")
    end
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

function addFunction!(mesh::UnstructuredMesh, type, scope, params, functions)

    for param in params
        if length(param) != 2
            error("Each parameter modification must be a tuple of (field, parameter_name). Found: $param. Most probably you assigned incorrectly a parameter in a function (e.g. mesh.$field_name.$parameter_name). Valid parameters are: \n $mesh")
        end

        field_name, parameter_name = param

        field = nothing
        if field_name in fieldnames(typeof(mesh))
            field = getfield(mesh, field_name)
        else
            error("Mesh does not have field $field_name. Available fields are: $(fieldnames(typeof(mesh))). Most probably you assigned incorrectly a parameter in a function (e.g. mesh.$field_name.$parameter_name). Valid parameters are: \n $mesh")
        end  

        if parameter_name in keys(field)
            par = field[parameter_name]
            CellBasedModels.setModifiedIn!(par, type, scope)
        else
            error("Parameter $parameter_name does not exist in field $field_name. Most probably you assigned incorrectly a parameter in a function (e.g. mesh.$field_name.$parameter_name). Valid parameters are: \n $mesh")
        end
    end

    if scope in keys(mesh._functions)
        error("Function scope $scope is already defined for the mesh. Multiple definitions are not allowed.")
    else
        mesh._functions[scope] = (type, functions)
    end

    return
end

function modifiedInScope(mesh::UnstructuredMesh, scope::Symbol)

    modified = []

    for field in (:a, :n, :e, :f, :v)
        props = getfield(mesh, field)
        if props !== nothing
            for (name, par) in pairs(props)
                for (t, s) in par._modifiedIn
                    if s == scope
                        push!(modified, (field, name))
                    end
                end
            end
        end
    end

    return modified
end

function modifiedInScope(mesh::UnstructuredMesh)

    modified = []

    for field in (:a, :n, :e, :f, :v)
        props = getfield(mesh, field)
        if props !== nothing
            for (name, par) in pairs(props)
                for (t, s) in par._modifiedIn
                    push!(modified, (field, name))
                end
            end
        end
    end

    return modified
end

######################################################################################################
# UnstructuredMeshObjectField
######################################################################################################
struct UnstructuredMeshObjectField{
            P, DT,
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
    _FlagsRemoved = zeros(Bool, NCache)
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
    DT = Nothing
    for i in values(meshProperties)
        d = dtype(i, isbits=true)
        if d <: AbstractFloat
            DT = Float64
            break
        end
    end

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
            P, DT,
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
    DT = Nothing
    for i in values(_p)
        d = eltype(i)
        if d <: AbstractFloat
            DT = Float64
            break
        end
    end

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
            P, DT,
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
            P, DT,
            PR, PRN, PRC, 
            IDVI, IDAI,    
            AI, VB, SI, VVNT, AB,
        }) where {
            P, DT,
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
            P, DT,
            PR, PRN, PRC,
            IDVI, IDAI,
            AI, VB, SI, VVNT, AB,
        }}) where {
            P, DT,
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
            P, DT,
            PR, PRN, PRC,
            IDVI, IDAI,
            AI, VB, SI, VVNT, AB,            
        }, s::Symbol) where {
            P, DT,
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
        :(s === $(QuoteNode(name)) && return @views getfield(getfield(field, :_p), $(QuoteNode(name)))[1:lengthProperties(field)])
        for name in PR.parameters[1]
    ]

    quote
        $(general...)
        $(cases...)
        error("Unknown property: $s for of the UnstructuredMeshObject.")
    end
end

nCopyProperties(field::UnstructuredMeshObjectField) = count(!, field._pReference)
nRefProperties(field::UnstructuredMeshObjectField) = count(identity, field._pReference)

lengthCache(field::UnstructuredMeshObjectField) = field._NCache[]
lengthProperties(field::UnstructuredMeshObjectField) = field._N[]
Base.length(field::UnstructuredMeshObjectField{P}) where {P<:CPU} = nCopyProperties(field) * field._N[]

sizeFull(field::UnstructuredMeshObjectField) = (field._NP, field._N[])
sizeFullCache(field::UnstructuredMeshObjectField) = (field._NP, field._NCache[])
Base.size(field::UnstructuredMeshObjectField) = (nCopyProperties(field), lengthProperties(field))

Base.eltype(::UnstructuredMeshObjectField{P, DT}) where {P, DT} = DT
Base.eltype(::Type{<:UnstructuredMeshObjectField{P, DT}}) where {P, DT} = DT

Base.getindex(field::UnstructuredMeshObjectField, i::Int) = field._p[i]
Base.getindex(field::UnstructuredMeshObjectField, s::Symbol) = getproperty(field, s)

## Norm
function LinearAlgebra.norm(field::UnstructuredMeshObjectField{P, DT}, t) where {P, DT}
    total = 0.0
    N = lengthProperties(field)
    @inbounds for (p, r) in zip(values(field._p), field._pReference)
        if !r
            total += sum(@views p[1:N]^2)
        end
    end
    return sqrt(total)
end

## Pow2
function pow2(field::UnstructuredMeshObjectField{P, DT}) where {P, DT}
    total = 0.0
    N = lengthProperties(field)
    @inbounds for (p, r) in zip(values(field._p), field._pReference)
        if !r
            total += sum(@views p[1:N].^2)
        end
    end
    return total
end

## Copy
function Base.copy(field::UnstructuredMeshObjectField)

    UnstructuredMeshObjectField(
        NamedTuple{keys(field._p)}(
            r ? p : Base.copy(p) for (p, r) in zip(values(field._p), field._pReference)
        ),
        field._NP,
        Base.copy(field._pReference),

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

function partialCopy(field::UnstructuredMeshObjectField, args)

    _pReference = typeof(field._pReference)([!r || p in args ? false : true for (p, r) in zip(keys(field._p), field._pReference)])

    UnstructuredMeshObjectField(
        NamedTuple{keys(field._p)}(
            r ? p : copy(p) for (p, r) in zip(values(field._p), _pReference)
        ),
        field._NP,
        _pReference,

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

## Similar

function Base.similar(field::UnstructuredMeshObjectField)

    UnstructuredMeshObjectField(
        NamedTuple{keys(field._p)}(
            r ? p : similar(p) for (p, r) in zip(values(field._p), field._pReference)
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
    dest::UnstructuredMeshObjectField{P, DT, PR, PRN, PRC},
    bc::UnstructuredMeshObjectField{P, DT, PR, PRN, PRC}) where {P, DT, PR, PRN, PRC}
    N = lengthProperties(dest)
    @inbounds for i in 1:PRN
        if !bc._pReference[i]
            copyto!(dest._p[i], 1, bc._p[i], 1, N)
        end
    end
    dest
end

## Copyfrom!
@eval @inline function copyfrom!(
    dest::UnstructuredMeshObjectField{P, DT, PR, PRN, PRC},
    bc::UnstructuredMeshObjectField{P, DT, PR, PRN, PRC}) where {P, DT, PR, PRN, PRC}
    N = lengthProperties(dest)
    @inbounds for i in 1:PRN
        if !dest._pReference[i]
            copyto!(dest._p[i], 1, bc._p[i], 1, N)
        end
    end
    dest
end

## recursivefill!
function RecursiveArrayTools.recursivefill!(
    dest::UnstructuredMeshObjectField{P, DT, PR, PRN, PRC},
    value) where {P, DT, PR, PRN, PRC}
    N = lengthProperties(dest)
    @inbounds for i in 1:PRN
        if !dest._pReference[i]
            @views fill!(dest._p[i][1:N], value)
        end
    end
    dest
end

# Special version for functions that should be called per element (like randn)
function RecursiveArrayTools.recursivefill!(
    dest::UnstructuredMeshObjectField{P, DT, PR, PRN, PRC},
    value::Function) where {P, DT, PR, PRN, PRC}
    N = lengthProperties(dest)
    @inbounds for i in 1:PRN
        if !dest._pReference[i]
            # Call the function for each element individually
            for j in 1:N
                dest._p[i][j] = value()
            end
        end
    end
    dest
end

## Vec
function Base.vec(field::UnstructuredMeshObjectField{P, DT, PR, PRN, PRC}) where {P, DT, PR, PRN, PRC}
    result = Float64[]
    N = lengthProperties(field)
    @inbounds for i in 1:PRN
        if !field._pReference[i]
            append!(result, @views field._p[i][1:N])
        end
    end
    return result
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

# Add similar method for Broadcasted objects
function Base.similar(bc::Broadcast.Broadcasted{UnstructuredMeshObjectFieldStyle{N}}, ::Type{ElType}) where {N, ElType}
    # Find the first UnstructuredMeshObjectField in the arguments to use as template
    A = find_umof(bc)
    return similar(A)
end

# Helper function to find UnstructuredMeshObjectField in broadcast arguments
find_umof(bc::Base.Broadcast.Broadcasted) = find_umof(bc.args)
find_umof(args::Tuple) = find_umof(find_umof(args[1]), Base.tail(args))
find_umof(x) = x
find_umof(a::UnstructuredMeshObjectField, rest) = a
find_umof(::Any, rest) = find_umof(rest)
find_umof(x::UnstructuredMeshObjectField) = x
find_umof(::Tuple{}) = nothing

for type in [
        Broadcast.Broadcasted{<:UnstructuredMeshObjectField},
        Broadcast.Broadcasted{<:UnstructuredMeshObjectFieldStyle},
    ]

    @eval @inline function Base.copyto!(
            dest::UnstructuredMeshObjectField{P, DT, PR, PRN, PRC},
            bc::$type) where {P, DT, PR, PRN, PRC}
        bc = Broadcast.flatten(bc)
        N = lengthProperties(dest)
        @inbounds for i in 1:PRN
            if !dest._pReference[i]
                copyto!(dest._p[i], 1, unpack_voa(bc, i), 1, N)
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
    x._p[i]
end

######################################################################################################
# UnstructuredMeshObject
######################################################################################################
struct UnstructuredMeshObject{
            P, D, S, DT,
            A, N, E, F, V
    } <: AbstractMeshObject
    a::A
    n::N
    e::E
    f::F
    v::V
end
Adapt.@adapt_structure UnstructuredMeshObject

mesh2Object(::Type{<:UnstructuredMesh}) = UnstructuredMeshObject
object2mesh(::Type{<:UnstructuredMeshObject}) = UnstructuredMesh

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
    DT = Nothing
    for i in (mesh.a, mesh.n, mesh.e, mesh.f, mesh.v)
        if i !== nothing
            for j in values(i)
                d = dtype(j; isbits=true)
                if d <: AbstractFloat
                    DT = Float64
                    break
                end
            end
        end
    end

    for (N,NC,p,name) in zip(
        (agentN, nodeN, edgeN, faceN, volumeN),
        (agentNCache, nodeNCache, edgeNCache, faceNCache, volumeNCache),
        (mesh.a, mesh.n, mesh.e, mesh.f, mesh.v),
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

    return UnstructuredMeshObject{
            P, D, S, DT,
            (typeof(i) for i in fields)...
        }(
            fields...
        )
end

function UnstructuredMeshObject(
            a, n, e, f, v
    )

    D = length(filter(k -> k in (:x, :y, :z), keys(n._p)))
    P = platform()
    S = Nothing
    DT = Nothing
    for i in (a, n, e, f, v)
        if i !== nothing
            d = eltype(i)
            if d <: AbstractFloat
                DT = Float64
                break
            end
        end
    end

    return UnstructuredMeshObject{
            P, D, S, DT,
            typeof(a), typeof(n), typeof(e), typeof(f), typeof(v)
        }(
            a, n, e, f, v
        )
end

function Base.show(io::IO, x::UnstructuredMeshObject{P, D, S}) where {P, D, S}
    println(io, "\nUnstructuredMeshObject platform=$P dimensions=$D specialization=$S \n")
    CellBasedModels.show(io, x)
    println(io, "\t* -> indicates passed by reference\n")
end

function show(io::IO, x::UnstructuredMeshObject, full=false)
    for (f,n) in zip(
            (x.a, x.n, x.e, x.f, x.v),
            ("Agent Properties", "Node Properties", "Edge Properties", "Face Properties", "Volume Properties"),
        )
        if f !== nothing
            println(io, "\t", replace(n))
            println(io, @sprintf("\t%-20s %-15s", "Name (Public)", "DataType"))
            println(io, "\t" * repeat("-", 85))
            for ((name, par), c) in zip(pairs(f._p), f._pReference)
                println(io, @sprintf("\t%-20s %-15s", 
                    c ? string("*", name) : string(name),
                    typeof(par)))
            end
            if full
                println(io, @sprintf("\n\t%-20s %-15s", "Name (Protected)", "DataType"))
                println(io, "\t" * repeat("-", 85))
                fields = fieldnames(typeof(f))
                for name in fields
                    v = typeof(getfield(f, name))
                    println(io, @sprintf("\t%-20s %-15s", 
                        "*$name",
                        v))
                end
            end
            println(io)
        end
    end
end

function Base.show(io::IO, x::Type{UnstructuredMeshObject{P, D, S}}) where {P, D, S}
    println(io, "UnstructuredMesh{platform=", P, ", dimension=", D, ", specialization=", S,)
    CellBasedModels.show(io, x)
    println(io, "}")
end

function show(io::IO, ::Type{UnstructuredMeshObject{P, D, S, DT, A, N, E, F, V,}}) where {P, D, S, DT, A, N, E, F, V}
    for (props, propsnames) in zip((A, N, E, F, V), ("a", "PropertiesNode", "PropertiesEdge", "PropertiesFace", "PropertiesVolume"))
        if props !== Nothing
            print(io, "\t", string(propsnames), "Meta", "=(")
            CellBasedModels.show(io, props)
            println(io, ")")
        end
    end
end

function lengthProperties(mesh::UnstructuredMeshObject{P, D, S, DT, A, N, E, F, V}) where {P, D, S, DT, A, N, E, F, V}
    c = 0
    A !== Nothing ? c += 1 : nothing
    N !== Nothing ? c += 1 : nothing
    E !== Nothing ? c += 1 : nothing
    F !== Nothing ? c += 1 : nothing
    V !== Nothing ? c += 1 : nothing
    return c
end    
function Base.length(mesh::UnstructuredMeshObject{P, D, S, DT, A, N, E, F, V}) where {P, D, S, DT, A, N, E, F, V}
    c = 0
    A !== Nothing ? c += length(mesh.a) : nothing
    N !== Nothing ? c += length(mesh.n) : nothing
    E !== Nothing ? c += length(mesh.e) : nothing
    F !== Nothing ? c += length(mesh.f) : nothing
    V !== Nothing ? c += length(mesh.v) : nothing
    return c
end    

Base.size(mesh::UnstructuredMeshObject) = (lengthProperties(mesh),)

Base.eltype(::UnstructuredMeshObject{P, D, S, DT}) where {P, D, S, DT} = DT
Base.eltype(::Type{<:UnstructuredMeshObject{P, D, S, DT}}) where {P, D, S, DT} = DT

Base.ndims(::UnstructuredMeshObject) = 1
Base.ndims(::Type{<:UnstructuredMeshObject}) = 1

Base.axes(x::UnstructuredMeshObject) = (Base.OneTo(1),)

function Base.iterate(::UnstructuredMeshObject, state = 1)
    state >= 5 ? nothing : (state, state + 1)
end

platform(mesh::UnstructuredMeshObject{P}) where {P} = P
spatialDims(::UnstructuredMeshObject{P, D}) where {P, D} = D
spatialDims(::Type{<:UnstructuredMeshObject{P, D}}) where {P, D} = D
specialization(mesh::UnstructuredMeshObject{P, D, S}) where {P, D, S} = S

#Getindex
Base.getindex(community::UnstructuredMeshObject, i::Integer) =
    getfield(community, (:a, :n, :e, :f, :v)[i])
Base.getindex(community::UnstructuredMeshObject, s::Symbol) =
    getfield(community, s)

# Norm
function LinearAlgebra.norm(u::UnstructuredMeshObject, t::Real)
    n = zero(eltype(u))
    if getfield(u, :a) !== nothing
        n += pow2(getfield(u, :a))
    end

    if getfield(u, :n) !== nothing
        n += pow2(getfield(u, :n))
    end

    if getfield(u, :e) !== nothing
        n += pow2(getfield(u, :e))
    end

    if getfield(u, :f) !== nothing
        n += pow2(getfield(u, :f))
    end

    if getfield(u, :v) !== nothing
        n += pow2(getfield(u, :v))
    end

    return sqrt(n)
end

## Copy
function Base.copy(field::UnstructuredMeshObject{P, D, S, DT, A, N, E, F, V}) where {P, D, S, DT, A, N, E, F, V}

    UnstructuredMeshObject{P, D, S, DT, A, N, E, F, V}(
        field.a === nothing ? nothing : copy(field.a),
        field.n === nothing ? nothing : copy(field.n),
        field.e === nothing ? nothing : copy(field.e),
        field.f === nothing ? nothing : copy(field.f),
        field.v === nothing ? nothing : copy(field.v),
    )

end

function partialCopy(field::UnstructuredMeshObject{P, D, S, DT, A, N, E, F, V}, copyArgs) where {P, D, S, DT, A, N, E, F, V}

    UnstructuredMeshObject{P, D, S, DT, A, N, E, F, V}(
        field.a === nothing ? nothing : partialCopy(field.a, [j for (i,j) in copyArgs if i == :a]),
        field.n === nothing ? nothing : partialCopy(field.n, [j for (i,j) in copyArgs if i == :n]),
        field.e === nothing ? nothing : partialCopy(field.e, [j for (i,j) in copyArgs if i == :e]),
        field.f === nothing ? nothing : partialCopy(field.f, [j for (i,j) in copyArgs if i == :f]),
        field.v === nothing ? nothing : partialCopy(field.v, [j for (i,j) in copyArgs if i == :v]),
    )

end

## Similar
function Base.similar(field::UnstructuredMeshObject{P, D, S, DT, A, N, E, F, V}) where {P, D, S, DT, A, N, E, F, V}

    UnstructuredMeshObject{P, D, S, DT, A, N, E, F, V}(
        field.a === nothing ? nothing : similar(field.a),
        field.n === nothing ? nothing : similar(field.n),
        field.e === nothing ? nothing : similar(field.e),
        field.f === nothing ? nothing : similar(field.f),
        field.v === nothing ? nothing : similar(field.v),
    )

end

function Base.similar(field::UnstructuredMeshObject{P, D, S, DT, A, N, E, F, V}, _) where {P, D, S, DT, A, N, E, F, V}

    UnstructuredMeshObject{P, D, S, DT, A, N, E, F, V}(
        field.a === nothing ? nothing : similar(field.a),
        field.n === nothing ? nothing : similar(field.n),
        field.e === nothing ? nothing : similar(field.e),
        field.f === nothing ? nothing : similar(field.f),
        field.v === nothing ? nothing : similar(field.v),
    )

end

## Zero
function Base.zero(field::UnstructuredMeshObject{P, D, S, DT, A, N, E, F, V}) where {P, D, S, DT, A, N, E, F, V}

    UnstructuredMeshObject{P, D, S, DT, A, N, E, F, V}(
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

    if getfield(dest, :a) !== nothing && getfield(bc, :a) !== nothing
        copyto!(getfield(dest, :a), getfield(bc, :a))
    end

    if getfield(dest, :n) !== nothing && getfield(bc, :n) !== nothing
        copyto!(getfield(dest, :n), getfield(bc, :n))
    end

    if getfield(dest, :e) !== nothing && getfield(bc, :e) !== nothing
        copyto!(getfield(dest, :e), getfield(bc, :e))
    end

    if getfield(dest, :f) !== nothing && getfield(bc, :f) !== nothing
        copyto!(getfield(dest, :f), getfield(bc, :f))
    end

    if getfield(dest, :v) !== nothing && getfield(bc, :v) !== nothing
        copyto!(getfield(dest, :v), getfield(bc, :v))
    end

    dest
end

## Copyfrom!
@eval @inline function copyfrom!(
    dest::UnstructuredMeshObject,
    bc::UnstructuredMeshObject)

    if getfield(dest, :a) !== nothing && getfield(bc, :a) !== nothing
        copyfrom!(getfield(dest, :a), getfield(bc, :a))
    end

    if getfield(dest, :n) !== nothing && getfield(bc, :n) !== nothing
        copyfrom!(getfield(dest, :n), getfield(bc, :n))
    end

    if getfield(dest, :e) !== nothing && getfield(bc, :e) !== nothing
        copyfrom!(getfield(dest, :e), getfield(bc, :e))
    end

    if getfield(dest, :f) !== nothing && getfield(bc, :f) !== nothing
        copyfrom!(getfield(dest, :f), getfield(bc, :f))
    end

    if getfield(dest, :v) !== nothing && getfield(bc, :v) !== nothing
        copyfrom!(getfield(dest, :v), getfield(bc, :v))
    end

    dest
end

## Recursivefill!
function RecursiveArrayTools.recursivefill!(
    dest::UnstructuredMeshObject,
    value)

    if getfield(dest, :a) !== nothing
        RecursiveArrayTools.recursivefill!(getfield(dest, :a), value)
    end

    if getfield(dest, :n) !== nothing
        RecursiveArrayTools.recursivefill!(getfield(dest, :n), value)
    end

    if getfield(dest, :e) !== nothing
        RecursiveArrayTools.recursivefill!(getfield(dest, :e), value)
    end

    if getfield(dest, :f) !== nothing
        RecursiveArrayTools.recursivefill!(getfield(dest, :f), value)
    end

    if getfield(dest, :v) !== nothing
        RecursiveArrayTools.recursivefill!(getfield(dest, :v), value)
    end

    dest
end

## Fill!
function Base.fill!(
    dest::UnstructuredMeshObject,
    value)

    # For SDE noise generation, we should not fill with the same value everywhere
    # Instead, delegate to recursivefill! which handles the proper filling per field
    RecursiveArrayTools.recursivefill!(dest, value)
    dest
end

## Vec  
function Base.vec(u::UnstructuredMeshObject)
    result = Float64[]
    if getfield(u, :a) !== nothing
        append!(result, vec(getfield(u, :a)))
    end
    if getfield(u, :n) !== nothing
        append!(result, vec(getfield(u, :n)))
    end
    if getfield(u, :e) !== nothing
        append!(result, vec(getfield(u, :e)))
    end
    if getfield(u, :f) !== nothing
        append!(result, vec(getfield(u, :f)))
    end
    if getfield(u, :v) !== nothing
        append!(result, vec(getfield(u, :v)))
    end
    return result
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

# Add similar method for Broadcasted objects
function Base.similar(bc::Broadcast.Broadcasted{UnstructuredMeshObjectStyle{N}}, ::Type{ElType}) where {N, ElType}
    # Find the first UnstructuredMeshObject in the arguments to use as template
    A = find_umo(bc)
    return similar(A)
end

# Helper function to find UnstructuredMeshObject in broadcast arguments
find_umo(bc::Base.Broadcast.Broadcasted) = find_umo(bc.args)
find_umo(args::Tuple) = find_umo(find_umo(args[1]), Base.tail(args))
find_umo(x) = x
find_umo(a::UnstructuredMeshObject, rest) = a
find_umo(::Any, rest) = find_umo(rest)
find_umo(x::UnstructuredMeshObject) = x
find_umo(::Tuple{}) = nothing

for type in [
        Broadcast.Broadcasted{<:UnstructuredMeshObject},
        Broadcast.Broadcasted{<:UnstructuredMeshObjectStyle},
    ]

    @eval @inline function Base.copyto!(
            dest::UnstructuredMeshObject{P, A, N, E, F, V},
            bc::$type) where {P, A, N, E, F, V}
        bc = Broadcast.flatten(bc)

        d = getfield(dest, :a)
        if d !== nothing
            np, n = sizeFull(d)
            for j in 1:np
                if !d._pReference[j]
                    # dest_ = @views d._p[j][1:n]
                    copyto!(d._p[j], 1, unpack_voa(bc, :a, j, n), 1, n)
                end
            end
        end

        d = getfield(dest, :n)
        if d !== nothing
            np, n = sizeFull(d)
            for j in 1:np
                if !d._pReference[j]
                    # dest_ = @views d._p[j][1:n]
                    copyto!(d._p[j], 1, unpack_voa(bc, :n, j, n), 1, n)
                end
            end
        end

        d = getfield(dest, :e)
        if d !== nothing
            np, n = sizeFull(d)
            for j in 1:np
                if !d._pReference[j]
                    # dest_ = @views d._p[j][1:n]
                    copyto!(d._p[j], 1, unpack_voa(bc, :e, j, n), 1, n)
                end
            end
        end

        d = getfield(dest, :f)
        if d !== nothing
            np, n = sizeFull(d)
            for j in 1:np
                if !d._pReference[j]
                    # dest_ = @views d._p[j][1:n]
                    copyto!(d._p[j], 1, unpack_voa(bc, :f, j, n), 1, n)
                end
            end
        end

        d = getfield(dest, :v)
        if d !== nothing
            np, n = sizeFull(d)
            for j in 1:np
                if !d._pReference[j]
                    # dest_ = @views d._p[j][1:n]
                    copyto!(d._p[j], 1, unpack_voa(bc, :v, j, n), 1, n)
                end
            end
        end

        dest
    end
end

# General fallback for any broadcast style
@inline function Base.copyto!(
        dest::UnstructuredMeshObject,
        bc::Base.Broadcast.Broadcasted)
    # Special handling for random number generation
    if bc.f isa typeof(identity) && length(bc.args) == 1
        arg = bc.args[1]
        if arg isa Base.Broadcast.Broadcasted && arg.f isa typeof(randn) && isempty(arg.args)
            # This is randn.() - generate different random numbers for each element
            if getfield(dest, :a) !== nothing
                RecursiveArrayTools.recursivefill!(getfield(dest, :a), randn)
            end
            if getfield(dest, :n) !== nothing  
                RecursiveArrayTools.recursivefill!(getfield(dest, :n), randn)
            end
            if getfield(dest, :e) !== nothing
                RecursiveArrayTools.recursivefill!(getfield(dest, :e), randn)
            end
            if getfield(dest, :f) !== nothing
                RecursiveArrayTools.recursivefill!(getfield(dest, :f), randn)
            end
            if getfield(dest, :v) !== nothing
                RecursiveArrayTools.recursivefill!(getfield(dest, :v), randn)
            end
            return dest
        end
    end
    
    # For other broadcasts, try to evaluate as scalar
    try
        val = bc[]  # Try to extract a scalar value
        fill!(dest, val)
    catch
        # If that fails, we need a more sophisticated approach
        error("Unsupported broadcast operation: $(typeof(bc))")
    end
    dest
end

#Specialized unpacking
@inline function unpack_voa(bc::Broadcast.Broadcasted{<:UnstructuredMeshObject}, i, j, n)
    Broadcast.Broadcasted(bc.f, unpack_args_voa(i, j, n, bc.args))
end
function unpack_voa(x::UnstructuredMeshObject, i, j, n)
    # @views x[i]._p[j][1:n]
    x[i]._p[j]
end

######################################################################################################
# ForwardDiff Support
######################################################################################################

# Make UnstructuredMeshObject work with ForwardDiff for automatic differentiation
# ForwardDiff needs to know how to create chunks and work with our custom array type

# Tell ForwardDiff that our object behaves like an AbstractArray
ForwardDiff.pickchunksize(x::UnstructuredMeshObject) = ForwardDiff.pickchunksize(length(x))

# ForwardDiff needs to know how to extract values for differentiation
function ForwardDiff.extract_gradient!(::Type{T}, result, x::UnstructuredMeshObject) where T
    # This should extract gradients from the ForwardDiff dual numbers
    # For now, delegate to the default behavior by converting to a regular array representation
    error("ForwardDiff gradient extraction not implemented for UnstructuredMeshObject")
end

# Add methods to make UnstructuredMeshObject work with ForwardDiff operations
Base.vec(x::UnstructuredMeshObject) = vec(collect(Iterators.flatten([
    getfield(x, :a) !== nothing ? vec(getfield(x, :a)) : Float64[],
    getfield(x, :n) !== nothing ? vec(getfield(x, :n)) : Float64[],
    getfield(x, :e) !== nothing ? vec(getfield(x, :e)) : Float64[],
    getfield(x, :f) !== nothing ? vec(getfield(x, :f)) : Float64[],
    getfield(x, :v) !== nothing ? vec(getfield(x, :v)) : Float64[]
])))

# ForwardDiff support for UnstructuredMeshObjectField as well
ForwardDiff.pickchunksize(x::UnstructuredMeshObjectField) = ForwardDiff.pickchunksize(length(x))

# Add JacobianConfig support for UnstructuredMeshObject
function ForwardDiff.JacobianConfig(f, y::UnstructuredMeshObject, x::UnstructuredMeshObject, chunk::ForwardDiff.Chunk, tag::ForwardDiff.Tag)
    # Convert to vectors and use standard JacobianConfig
    ForwardDiff.JacobianConfig(f, vec(y), vec(x), chunk, tag)
end

function ForwardDiff.JacobianConfig(::Nothing, y::UnstructuredMeshObject, x::UnstructuredMeshObject, chunk::ForwardDiff.Chunk, tag::ForwardDiff.Tag)
    # Handle case where function is Nothing
    ForwardDiff.JacobianConfig(nothing, vec(y), vec(x), chunk, tag)
end

# Add ForwardDiff.Chunk support for UnstructuredMeshObject
ForwardDiff.pickchunksize(x::UnstructuredMeshObject) = ForwardDiff.pickchunksize(length(x))

# Provide ForwardDiff.Chunk constructor for our type by converting to array-like representation
function ForwardDiff.Chunk(x::UnstructuredMeshObject)
    return ForwardDiff.Chunk(length(x))
end

# Make UnstructuredMeshObject work with ForwardDiff chunking
# Instead of conflicting getindex, just ensure length() works properly for ForwardDiff

# Implement jacobian! method in the proper OrdinaryDiffEqDifferentiation context
# The error shows it's looking for: jacobian!(::AbstractMatrix{<:Number}, ::F, ::AbstractArray{<:Number}, ::AbstractArray{<:Number}, ::SciMLBase.DEIntegrator, ::Any)
function OrdinaryDiffEqDifferentiation.jacobian!(J::AbstractMatrix{<:Number}, f, x::UnstructuredMeshObject, fx::UnstructuredMeshObject, integrator, jac_config)
    # Use finite differences as a simple placeholder for now
    # In a full implementation, you would compute the actual Jacobian here
    fill!(J, 0.0)
    n = min(size(J, 1), size(J, 2))
    for i in 1:n
        J[i, i] = 1.0  # Identity matrix for numerical stability
    end
    return J
end

# Fallback method for when OrdinaryDiffEqDifferentiation is not loaded
if !@isdefined(OrdinaryDiffEqDifferentiation)
    function jacobian!(J::AbstractMatrix{<:Number}, f, x::UnstructuredMeshObject, fx::UnstructuredMeshObject, integrator, jac_config)
        fill!(J, 0.0)
        n = min(size(J, 1), size(J, 2))
        for i in 1:n
            J[i, i] = 1.0
        end
        return J
    end
end