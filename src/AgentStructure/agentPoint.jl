const AgentPoint{D, PN} = UnstructuredMesh{D, AgentPoint, Nothing, PN, Nothing, Nothing, Nothing}
const AgentPointObject{D, PN} = UnstructuredMeshObject{D, AgentPoint, Nothing, PN, Nothing, Nothing, Nothing}

function AgentPoint(
    dims::Int;
    properties::Union{NamedTuple, Nothing}=nothing,
)

    UnstructuredMeshProperties{dims, Nothing, typeof(propertiesAgent), Nothing, Nothing, Nothing, Nothing, Nothing}(
        dims,
        propertiesAgent=nothing,
        propertiesNode=properties,
        propertiesEdge=nothing,
        propertiesFace=nothing,
        propertiesVolume=nothing,
        scopePosition=:propertiesAgent
    )

end

const CommunityPoint{D, P, S, AM, AP, ANP} = UnstructuredMeshObject{
            D, P, S,
            AM, AP, ANP,
            Nothing, Nothing, Nothing,
            Nothing, Nothing, Nothing,
            Nothing, Nothing, Nothing,
            Nothing, Nothing, Nothing,
        }

function UnstructuredMeshObject(
        mesh::AgentPoint;
        N::Integer=0,
        NCache::Union{Nothing, Integer}=nothing,
    )

    UnstructuredMeshObject(
        mesh;
        agentN=N,
        agentNCache=NCache,
    )

end