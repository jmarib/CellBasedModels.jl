######################################################################################################
# AGENT STRUCTURE
######################################################################################################
struct StructuredMesh{D, S, P} <: AbstractMesh

    properties::NamedTuple                   # Dictionary to hold agent properties

end

function StructuredMesh(
    dims::Int;
    properties::NamedTuple
)

    if dims < 1 || dims > 3
        error("dims must be between 1 and 3. Found $dims")
    end

    properties = parameterConvert(properties)

    return StructuredMesh{dims, Nothing, typeof(properties)}(
        properties
    )
end

function Base.show(io::IO, x::StructuredMesh{D}) where {D}
    println(io, "StructuredMesh with dimensions $(D): \n")
    CellBasedModels.show(io, x)
end

function show(io::IO, x::StructuredMesh{D}) where {D}
    for p in propertynames(x)
        props = getfield(x, p)
        if props !== nothing
            println(io, replace(uppercase(string(p)), "PROPERTIES"=>"PROPERTIES "))
            println(io, @sprintf("\t%-15s %-15s %-15s %-20s %-20s %-s", "Name", "DataType", "Dimensions", "Default_Value", "ModifiedIn", "Description"))
            println(io, "\t" * repeat("-", 85))
            for (name, par) in pairs(props)
                println(io, @sprintf("\t%-15s %-15s %-15s %-20s %-20s %-s", 
                    name, 
                    dtype(par), 
                    par.dimensions === nothing ? "" : string(par.dimensions),
                    par.defaultValue === nothing ? "" : string(par.defaultValue), 
                    length(par._modifiedIn) === 0 ? "" : string(tuple(par._modifiedIn)),
                    par.description))
            end
            println(io)
        end
    end
end

function Base.show(io::IO, ::Type{StructuredMesh{D, S, P}}) where {D, S, P}
    println(io, "StructuredMesh{dims=", D, ", scopePosition=", S, ",")
    for (i, (n, t)) in enumerate(zip(P.parameters[1], P.parameters[2].parameters))
        i > 1 && print(io, ", ")
        print(io, string(n), "::", t.parameters[1])
    end
    println(io, ")")
    println(io, "}")
end

spatialDims(::StructuredMesh{D}) where {D} = D
specialization(::StructuredMesh{D, S}) where {D, S} = S

######################################################################################################
# StructuredMeshObject
######################################################################################################
struct StructuredMeshObject{
        P, D, S, PR, PRef, SB, NG
    } <: AbstractMeshObject
    p::PR
    _pReference::PRef
    _simulationBox::SB
    _gridSpacing::NG
end
Adapt.@adapt_structure StructuredMeshObject

mesh2Object(::Type{<:StructuredMesh}) = StructuredMeshObject
object2mesh(::Type{<:StructuredMeshObject}) = StructuredMesh

function StructuredMeshObject(
        mesh::StructuredMesh{D, S};
        simulationBox,
        gridSpacing,
    ) where {D, S}

    if size(simulationBox) != (D, 2)
        error("simulationBox must be of size ($(D), 2). Found size $(size(simulationBox))")
    end
    if length(gridSpacing) == 0
        gridSpacing = fill(gridSpacing, D)
    elseif length(gridSpacing) != D
        error("gridSpacing must be of length $(D). Found length $(length(gridSpacing))")
    end

    simulationBox = SizedMatrix{D, 2, standardDataType(AbstractFloat)}(simulationBox)
    gridSpacing = SizedVector{D,standardDataType(AbstractFloat)}(gridSpacing)

    gridSize = Int.(ceil.((simulationBox[:, 2] .- simulationBox[:, 1]) ./ gridSpacing))
    fields = NamedTuple{keys(mesh.properties)}([zeros(dtype(j, isbits=true), gridSize...) for (i,j) in pairs(mesh.properties)])

    _pReference = SizedVector{length(fields), Bool}([true for _ in 1:length(fields)])

    P = platform()

    return StructuredMeshObject{
            P, D, S, typeof(fields), typeof(_pReference), typeof(simulationBox), typeof(gridSpacing)
        }(
            fields, _pReference, simulationBox, gridSpacing
        )
end

function StructuredMeshObject(
        fields, pReference, simulationBox, gridSpacing
    )

    P = platform(fields[1])
    D = length(gridSpacing)
    S = Nothing
    PR = typeof(fields)
    PRef = typeof(pReference)
    SB = typeof(simulationBox)
    NG = typeof(gridSpacing)

    return StructuredMeshObject{
            P, D, S, PR, PRef, SB, NG
        }(
            fields, pReference, simulationBox, gridSpacing
        )
end

function Base.show(io::IO, x::StructuredMeshObject{P, D}) where {P, D}
    println(io, "\nStructuredMeshObject platform=$P dimensions=$D shape=$(size(x.p[1]))\n")
    println(io, "simulationBox=$(x._simulationBox) spacing=$(x._gridSpacing)\n")
    CellBasedModels.show(io, x)
    println(io, "\t* -> indicates passed by reference\n")
end

function show(io::IO, x::StructuredMeshObject{P, D, S, PR, PRef, SB, NG}, full=false) where {P, D, S, PR, PRef, SB, NG}
    println(io, @sprintf("\t%-20s %-15s", "Name (Public)", "DataType"))
    println(io, "\t" * repeat("-", 85))
    for ((name, par), c) in zip(pairs(x.p), x._pReference)
        println(io, @sprintf("\t%-20s %-15s", 
            c ? string("*", name) : string(name),
            typeof(par)))
    end
end

# function Base.show(io::IO, x::Type{StructuredMeshObject{
#             P, D, S, A, N, E, F, V,
#         }}) where {
#             P, D, S, A, N, E, F, V,
#         }
#     println(io, "StructuredMesh{platform=", P, ", dimension=", D, ", scopePosition=", S,)
#     CellBasedModels.show(io, x)
#     println(io, "}")
# end

# function show(io::IO, ::Type{StructuredMeshObject{
#             P, D, S, A, N, E, F, V,
#         }}) where {
#             P, D, S, A, N, E, F, V,
#         }
#     for (props, propsnames) in zip((A, N, E, F, V), ("PropertiesAgent", "PropertiesNode", "PropertiesEdge", "PropertiesFace", "PropertiesVolume"))
#         if props !== Nothing
#             print(io, "\t", string(propsnames), "Meta", "=(")
#             # println(propsmeta)
#             CellBasedModels.show(io, props)
#             println(io, ")")
#             # print(io, "\t", string(props), "=(")
#             # for (i, (n, t)) in enumerate(zip(props.parameters[1], props.parameters[2].parameters))
#             #     i > 1 && print(io, ", ")
#             #     print(io, string(n), "::", t)
#             # end
#             # println(io, ")")
#         end
#     end
# end

Base.length(::StructuredMeshObject{P, D, S, PR}) where {P, D, S, PR} = length(PR.parameters[1])

Base.size(mesh::StructuredMeshObject) = (length(mesh), size(mesh.p[1])...)

Base.ndims(::StructuredMeshObject{P, D}) where {P, D} = 1 + D
Base.ndims(::Type{<:StructuredMeshObject{P, D}}) where {P, D} = 1 + D

function Base.eltype(::Type{<:StructuredMeshObject})
    return CellBasedModels.concreteDataType(AbstractFloat)
end

function Base.iterate(mesh::StructuredMeshObject, state = 1)
    state >= length(mesh) ? nothing : (state, state + 1)
end

platform(mesh::StructuredMeshObject{P}) where {P} = P
spatialDims(::StructuredMeshObject{P, D}) where {P, D} = D
specialization(::StructuredMeshObject{P, D, S}) where {P, D, S} = S

#Getindex
Base.getindex(community::StructuredMeshObject, i::Integer) = community.p[i]
Base.getindex(community::StructuredMeshObject, s::Symbol) = community.p[s]

## Copy
function Base.copy(field::StructuredMeshObject{P, D, S}) where {P, D, S}

    StructuredMeshObject{P, D, S, typeof(field.p), typeof(field._pReference), typeof(field._simulationBox), typeof(field._gridSpacing)}(
        NamedTuple{keys(field.p)}([r ? field.p[i] : copy(field.p[i]) for (i,r) in zip(keys(field.p), field._pReference)]),
        field._pReference,
        field._simulationBox,
        field._gridSpacing,
    )

end

## Zero
function Base.zero(field::StructuredMeshObject{P, D, S}) where {P, D, S}

    StructuredMeshObject{P, D, S, typeof(field.p), typeof(field._pReference), typeof(field._simulationBox), typeof(field._gridSpacing)}(
        NamedTuple{keys(field.p)}([r ? field.p[i] : zero(field.p[i]) for (i,r) in zip(keys(field.p), field._pReference)]),
        field._pReference,
        field._simulationBox,
        field._gridSpacing,
    )

end

## Copyto!
@eval @inline function Base.copyto!(
    dest::StructuredMeshObject,
    bc::StructuredMeshObject)
    for (i,r) in zip(1:length(dest), dest._pReference)
        if !r
            copyto!(dest[i], bc[i])
        end
    end
    dest
end

## Broadcasting
struct StructuredMeshObjectStyle{N} <: Broadcast.AbstractArrayStyle{N} end

# Allow constructing the style from Val or Int
StructuredMeshObjectStyle(::Val{N}) where {N} = StructuredMeshObjectStyle{N}()
StructuredMeshObjectStyle(N::Int) = StructuredMeshObjectStyle{N}()

# Your StructuredMeshObject acts like a 2D array
Base.BroadcastStyle(::Type{<:StructuredMeshObject{P, D}}) where {P, D} = StructuredMeshObjectStyle{D}()

# Combine styles safely
Base.Broadcast.result_style(::StructuredMeshObjectStyle{M}) where {M} =
    StructuredMeshObjectStyle{M}()
Base.Broadcast.result_style(::StructuredMeshObjectStyle{M}, ::StructuredMeshObjectStyle{N}) where {M,N} =
    StructuredMeshObjectStyle{max(M,N)}()
Base.Broadcast.result_style(::StructuredMeshObjectStyle{M}, ::Base.Broadcast.AbstractArrayStyle{N}) where {M,N} =
    StructuredMeshObjectStyle{max(M,N)}()
Base.Broadcast.result_style(::Base.Broadcast.AbstractArrayStyle{M}, ::StructuredMeshObjectStyle{N}) where {M,N} =
    StructuredMeshObjectStyle{max(M,N)}()

Broadcast.broadcastable(x::StructuredMeshObject) = x

for type in [
        Broadcast.Broadcasted{<:StructuredMeshObject},
        Broadcast.Broadcasted{<:StructuredMeshObjectStyle},
    ]

    @eval @inline function Base.copyto!(
            dest::StructuredMeshObject,
            bc::$type)
        bc = Broadcast.flatten(bc)
        @inbounds for i in 1:length(dest)
            if !dest._pReference[i]
                copyto!(dest[i], unpack_voa(bc, i))
            end
        end
        dest
    end
end

#Specialized unpacking
@inline function unpack_voa(bc::Broadcast.Broadcasted{<:StructuredMeshObject}, i)
    Broadcast.Broadcasted(bc.f, unpack_args_voa(i, bc.args))
end
function unpack_voa(x::StructuredMeshObject, i)
    x[i]
end