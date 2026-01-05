abstract type AgentGlobalModel end
const AgentGlobal{PN} = UnstructuredMesh{0, AgentGlobalModel, PN}
const AgentGlobalObject{P, DT, NN, PN} = UnstructuredMeshObject{P, 0, AgentGlobalModel, DT, NN, PN}

function AgentGlobal(
    properties::Union{NamedTuple, Nothing}=nothing,
)

    UnstructuredMesh(
        0,
        g=Node(properties),
        specialization=AgentGlobalModel
    )

end

function createObject(
        mesh::AgentGlobal;
    )

    UnstructuredMeshObject(
        mesh;
        g=(1,1)
    )

end