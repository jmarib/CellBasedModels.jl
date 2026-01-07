abstract type AgentPointModel end
const AgentPoint{D, P} = UnstructuredMesh{D, AgentPointModel, P}
const AgentPointObject{P, D, DT, NN, PAR} = UnstructuredMeshObject{P, D, AgentPointModel, DT, NN, PAR}

function AgentPoint(
    dims::Int,
    properties::Union{NamedTuple, Nothing}=nothing,
)

    UnstructuredMesh(
        dims,
        n=Node(properties),
        specialization=AgentPointModel,
    )

end

function createObject(
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

iterateOverNeighbors(mesh::AgentPointObject, agentIndex::Integer) = iterateOverNeighbors(mesh, :n, agentIndex)

@generated function addAgent!(
    meshObject::AgentPointObject{P, D, DT, NN, PN},
    fields::NamedTuple{F, T}
) where {P, D, DT, NN, PN, F, T}

    # Compile-time check: ensure that the field names of `fields` (F)
    # match the expected node property names encoded in PN (NamedTuple type)
    expected_keys = PN.parameters[1].parameters[3].parameters[1]
    f = nothing
    if sort(F) != sort(expected_keys)
        # Build a helpful error message showing differences
        text = "addAgent: AgentPoint model expected keys $(expected_keys), got $(F)."
        f = :(error($text))
    end

    updates = [:(meshObject.n._$(k)[nPos] = fields.$k) for k in F]

    quote
        $(f)
        println("Adding AgentPoint at position ", nPos)
        nPos = Atomics.@atomic meshObject.n._NNew[] += 1     
        nPos += 1
        meshObject.n._FlagsSurvived[nPos] = true
        $(updates...)
    end

end

function removeAgent!(
    meshObject::AgentPointObject,
    agentIndex::Integer,
)

    meshObject.n._FlagsSurvived[agentIndex] = false

end