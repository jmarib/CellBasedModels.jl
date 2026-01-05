abstract type AgentPointModel end
const AgentPoint{D, PN} = UnstructuredMesh{D, AgentPointModel, Nothing, PN, Nothing, Nothing, Nothing}
const AgentPointObject{P, D, DT, NN, PN} = UnstructuredMeshObject{P, D, AgentPointModel, DT, NN, Nothing, PN, Nothing, Nothing, Nothing}

function AgentPoint(
    dims::Int;
    properties::Union{NamedTuple, Nothing}=nothing,
)

    UnstructuredMesh(
        dims,
        n=properties,
        specialization=AgentPointModel,
    )

end

function AgentPointObject(
        mesh::AgentPoint;
        n::Union{Integer,Tuple{Integer, Integer}}=0,
    )

    UnstructuredMeshObject(
        mesh;
        n=n,
    )

end

##########################################################################
# Functions for working with Agents
##########################################################################

iterateAgents(mesh::AgentPointObject) = 1:mesh.n._N[]
iterateOverNeighbors(mesh::AgentPointObject, agentIndex::Integer) = iterateOverNeighbors(mesh, :n, agentIndex)

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
    meshObject.n._FlagsSurvived[agentIndex] = false
    meshObject.n._NRemovedThread[threadId] += 1

end