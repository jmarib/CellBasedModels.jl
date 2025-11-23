abstract type AgentPointModel end
const AgentPoint{D, PN} = UnstructuredMesh{D, AgentPointModel, Nothing, PN, Nothing, Nothing, Nothing}
const AgentPointObject{P, D, DT, NN, PN} = UnstructuredMeshObject{P, D, AgentPointModel, DT, NN, Nothing, PN, Nothing, Nothing, Nothing}

function AgentPoint(
    dims::Int;
    properties::Union{NamedTuple, Nothing}=nothing,
)

    UnstructuredMesh(
        dims,
        propertiesAgent=nothing,
        propertiesNode=properties,
        propertiesEdge=nothing,
        propertiesFace=nothing,
        propertiesVolume=nothing,
        specialization=AgentPointModel,
    )

end

# Specialized show methods for AgentPoint types
function Base.show(io::IO, x::AgentPoint{D, PN}) where {D, PN}
    println(io, "AgentPoint{dims=$D}:")
    if D > 0
        spatial_props = []
        D >= 1 && push!(spatial_props, "x")
        D >= 2 && push!(spatial_props, "y") 
        D >= 3 && push!(spatial_props, "z")
        println(io, "  Spatial dimensions: $(join(spatial_props, ", "))")
    else
        println(io, "  Non-spatial agent model")
    end
    
    if PN <: NamedTuple
        node_keys = PN.parameters[1]
        if length(node_keys) > 0
            non_spatial = filter(k -> !(k in (:x, :y, :z)), node_keys)
            if length(non_spatial) > 0
                println(io, "  Agent properties: $(join(non_spatial, ", "))")
            else
                println(io, "  Agent properties: (spatial only)")
            end
        else
            println(io, "  Agent properties: none")
        end
    else
        println(io, "  Agent properties: $(PN)")
    end
end

# Specialized show methods for AgentPoint types (Type display)
function Base.show(io::IO, ::Type{AgentPoint{D, PN}}) where {D, PN}
    print(io, "AgentPoint{dims=$D")
    if PN <: NamedTuple
        node_keys = PN.parameters[1]
        if length(node_keys) > 0
            non_spatial = filter(k -> !(k in (:x, :y, :z)), node_keys)
            if length(non_spatial) > 0
                print(io, ", properties=($(join(non_spatial, ", ")))")
            else
                print(io, ", properties=(spatial only)")
            end
        else
            print(io, ", properties=(none)")
        end
    else
        print(io, ", properties=$PN")
    end
    print(io, "}")
end

function AgentPointObject(
        mesh::AgentPoint;
        N::Integer=0,
        NCache::Union{Nothing, Integer}=nothing,
    )

    UnstructuredMeshObject(
        mesh;
        nodeN=N,
        nodeNCache=NCache,
    )

end

function Base.show(io::IO, x::AgentPointObject{D, PN}) where {D, PN}
    println(io, "AgentPointObject{dims=$D}:")
    
    # Show spatial dimensions
    if D > 0
        spatial_props = []
        D >= 1 && push!(spatial_props, "x")
        D >= 2 && push!(spatial_props, "y")
        D >= 3 && push!(spatial_props, "z")
        println(io, "  Spatial dimensions: $(join(spatial_props, ", "))")
    else
        println(io, "  Non-spatial agent model")
    end
    
    # Show agent count and cache info
    n_agents = lengthProperties(x.n)
    n_cache = lengthCache(x.n)
    println(io, "  Active agents: $n_agents")
    println(io, "  Cache capacity: $n_cache")
    
    # Show agent properties
    if PN <: NamedTuple
        node_keys = PN.parameters[1]
        if length(node_keys) > 0
            non_spatial = filter(k -> !(k in (:x, :y, :z)), node_keys)
            if length(non_spatial) > 0
                println(io, "  Agent properties: $(join(non_spatial, ", "))")
            else
                println(io, "  Agent properties: (spatial only)")
            end
        else
            println(io, "  Agent properties: none")
        end
    else
        println(io, "  Agent properties: $(PN)")
    end
    
    # Show memory usage info
    n_added = x.n._NAdded[]
    n_removed = x.n._NRemoved[]
    if n_added > 0 || n_removed > 0
        println(io, "  Pending: +$n_added agents, -$n_removed agents")
    end
    
    # Show platform information
    platform_type = platform(x)
    println(io, "  Platform: $platform_type")
end

function Base.show(io::IO, ::Type{AgentPointObject{D, PN}}) where {D, PN}
    print(io, "AgentPointObject{dims=$D")
    if PN <: NamedTuple
        node_keys = PN.parameters[1]
        if length(node_keys) > 0
            non_spatial = filter(k -> !(k in (:x, :y, :z)), node_keys)
            if length(non_spatial) > 0
                print(io, ", properties=($(join(non_spatial, ", ")))")
            else
                print(io, ", properties=(spatial only)")
            end
        else
            print(io, ", properties=(none)")
        end
    else
        print(io, ", properties=$PN")
    end
    print(io, "}")
end

##########################################################################
# Functions for working with Agents
##########################################################################

iterateAgents(mesh::AgentPointObject) = 1:mesh.n._N[]
iterateNeighbors(mesh::AgentPointObject, agentIndex::Integer) = 1:mesh.n._N[]

@generated function addAgent!(
    meshObject::AgentPointObject{P, D, DT, NN, PN},
    fields::NamedTuple{F, T}
) where {P, D, DT, NN, PN, F, T}

    # Compile-time check: ensure that the field names of `fields` (F)
    # match the expected node property names encoded in PN (NamedTuple type)
    expected_keys = PN.parameters[3].parameters[1]
    f = nothing
    if F != expected_keys
        # Build a helpful error message showing differences
        text = "addAgent: Model expected keys $(expected_keys), got $(F)."
        f = :(error($text))
    end

    quote
        $(f)
        threadId = Threads.threadid()
        pos = meshObject.n._NAddedThread[threadId] + 1
        if pos > length(meshObject.n._AddedAgents[threadId])
            push!(meshObject.n._AddedAgents[threadId], fields)
        else
            meshObject.n._AddedAgents[threadId][pos] = fields
        end
        meshObject.n._NAddedThread[threadId] += 1
    end

end

function removeAgent!(
    meshObject::AgentPointObject,
    agentIndex::Integer,
)

    threadId = Threads.threadid()
    meshObject.n._RemovedAgents[agentIndex] = true
    meshObject.n._NRemovedThread[threadId] += 1

end