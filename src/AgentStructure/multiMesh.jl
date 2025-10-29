######################################################################################################
# MultiMesh
######################################################################################################

struct MultiMesh{D, M}

    _meshes::M

end

function MultiMesh(;kwargs...)

    D = nothing
    for (n,p) in pairs(kwargs)
        if startswith("$n", "_")
            error("Mesh names in MultiMesh cannot start with an underscore ($n) to prevent conflicts with internal fields.")
        elseif !(typeof(p) <: AbstractMesh)
            error("All values in MultiMesh must be of type AbstractMesh. Mesh $n is of type $(typeof(p)).")
        end
        if D === nothing
            D = CellBasedModels.spatialDims(p)
        else
            @assert D == CellBasedModels.spatialDims(p) "All meshes in MultiMesh must have the same spatial dimensions. Mesh $n has spatial dimensions $(CellBasedModels.spatialDims(p)) while previous meshes have spatial dimensions $D"
        end
    end

    MultiMesh{D, typeof(kwargs)}(kwargs)

end

function Base.show(io::IO, x::MultiMesh{D}) where {D}
    println(io, "MultiMeshMesh with dimensions $(D): \n")
    for (p, n) in pairs(x._meshes)
        println(io, "Mesh: $p")
        show(io, n)
        println(io, "\n")
    end
end

function Base.show(io::IO, ::Type{MultiMesh{D, P}}) where {D, P}
    println(io, "MultiMeshMesh Type with dimensions $(D)")
    for (p, n) in zip(P.parameters[1], P.parameters[2].parameters)
        println(io, "Mesh: $p => Type: $(n)")
    end
end

spatialDims(::MultiMesh{D, M}) where {D, M} = D
Base.length(mm::MultiMesh{D, M}) where {D, M} = length(mm._meshes)

Base.getindex(mm::MultiMesh{D, M}, key) where {D, M} = mm._meshes[key]
Base.getproperty(mm::MultiMesh{D, M}, key) where {D, M} = key == :_meshes ? getfield(mm, key) : getfield(getfield(mm, :_meshes), key)

######################################################################################################
# MultiMeshObject
######################################################################################################
struct MultiMeshObject{P, D, M}

    _meshes::M

end
Adapt.@adapt_structure MultiMeshObject

function MultiMeshObject(mm::MultiMesh{D, M}; kwargs...) where {D, M}

    mm_keys = keys(mm._meshes)
    for n in keys(mm._meshes)
        if ! n in mm_keys
            error("Mesh name $n not found in MultiMesh.")
        end
    end

    meshes = []
    for (n, mesh) in pairs(mm._meshes)
        m = typeof(mesh)
        mobj = CellBasedModels.eval(Meta.parse("$(m)Object"))
        push!(meshes, mobj(mm._meshes[n]; kwargs...))
    end
    meshes = NamedTuple{mm_keys}(meshes...)

    P = platform()

    MultiMeshObject{P, D, typeof(meshes)}(meshes)

end

function MultiMeshObject(_meshes)

    P = platform(_meshes[1])
    D = spatialDims(_meshes[1])
    M = typeof(_meshes)
    MultiMeshObject{P, D, M}(_meshes)

end

function Base.show(io::IO, x::MultiMeshObject{P, D}) where {P, D}
    println(io, "MultiMeshObject with dimensions $(D): \n")
    for (p, n) in pairs(x._meshes)
        println(io, "Mesh Object: $p")
        show(io, n)
        println(io, "\n")
    end
end

function Base.show(io::IO, ::Type{MultiMeshObject{P, D, M}}) where {P, D, M}
    println(io, "MultiMeshObject Type with dimensions $(D)")
    for (p, n) in zip(M.parameters[1], M.parameters[2].parameters)
        println(io, "Mesh Object: $p => Type: $(n)")
    end
end

Base.length(mm::MultiMeshObject{P, D, M}) where {P, D, M} = length(mm._meshes)
Base.size(mm::MultiMeshObject{P, D, M}) where {P, D, M} = (length(mm._meshes),)

Base.getindex(mm::MultiMeshObject{P, D, M}, key) where {P, D, M} = mm._meshes[key]

@generated function Base.getproperty(field::MultiMeshObject{P, D, M}, s::Symbol) where {P, D, M}
    general = [
        :(if s === $(QuoteNode(name)); return getfield(field, $(QuoteNode(name))); end)
        for name in fieldnames(MultiMeshObject)
    ]
    cases = [
        :(s === $(QuoteNode(name)) && return @views getfield(getfield(field, :_p), $(QuoteNode(name)))[1:length(field)])
        for name in M.parameters[1]
    ]

    quote
        $(general...)
        $(cases...)
        error("Unknown property: $s for of the MultiMeshObject.")
    end
end

## Copy
function Base.copy(mm::MultiMeshObject)

    MultiMeshObject(
        NamedTuple{keys(mm._meshes)}(
            [copy(mm._meshes[n]) for n in keys(mm._meshes)]
        )
    )

end

## Zero
function Base.zero(mm::MultiMeshObject)

    MultiMeshObject(
        NamedTuple{keys(mm._meshes)}(
            [zero(mm._meshes[n]) for n in keys(mm._meshes)]
        )
    )

end

## Copyto!
@eval @inline function Base.copyto!(
    dest::MultiMeshObject,
    bc::MultiMeshObject)
    @inbounds for i in keys(dest._meshes)
        if dest[i] !== nothing && bc[i] !== nothing
            copyto!(dest[i], bc[i])
        end
    end
    dest
end

## Broadcasting
struct MultiMeshObjectStyle{N} <: Broadcast.AbstractArrayStyle{N} end

# Allow constructing the style from Val or Int
MultiMeshObjectStyle(::Val{N}) where {N} = MultiMeshObjectStyle{N}()
MultiMeshObjectStyle(N::Int) = MultiMeshObjectStyle{N}()

# Your MultiMeshObject acts like a 1D array
Base.BroadcastStyle(::Type{<:MultiMeshObject}) = MultiMeshObjectStyle{1}()

# Combine styles safely
Base.Broadcast.result_style(::MultiMeshObjectStyle{M}) where {M} =
    MultiMeshObjectStyle{M}()
Base.Broadcast.result_style(::MultiMeshObjectStyle{M}, ::MultiMeshObjectStyle{N}) where {M,N} =
    MultiMeshObjectStyle{max(M,N)}()
Base.Broadcast.result_style(::MultiMeshObjectStyle{M}, ::Base.Broadcast.AbstractArrayStyle{N}) where {M,N} =
    MultiMeshObjectStyle{max(M,N)}()
Base.Broadcast.result_style(::Base.Broadcast.AbstractArrayStyle{M}, ::MultiMeshObjectStyle{N}) where {M,N} =
    MultiMeshObjectStyle{max(M,N)}()

Broadcast.broadcastable(x::MultiMeshObject) = x

for type in [
        Broadcast.Broadcasted{<:MultiMeshObject},
        Broadcast.Broadcasted{<:MultiMeshObjectStyle},
    ]

    @eval @inline function Base.copyto!(
            dest::MultiMeshObject{P, D, M},
            bc::$type) where {P, D, M}
        bc = Broadcast.flatten(bc)
        @inbounds for i in 1:length(dest)
            m = dest[i]
            for j in 1:length(m)
                d = m[j]
                if d !== nothing
                    np, n = size(d)
                    for k in 1:np
                        if !d._pReference[k]
                            dest_ = @views dest[i][j]._p[k][1:n]
                            copyto!(dest_, unpack_voa(bc, i, j, k, n))
                        end
                    end
                end
            end
        end
        dest
    end
end

#Specialized unpacking
@inline function unpack_voa(bc::Broadcast.Broadcasted{<:MultiMeshObject}, i, j, k, n)
    Broadcast.Broadcasted(bc.f, unpack_args_voa(i, j, k, n, bc.args))
end
function unpack_voa(x::MultiMeshObject, i, j, k, n)
    @views x[i][j][k][1:n]
end