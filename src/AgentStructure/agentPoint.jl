const AgentPoint{D, S, PA} = UnstructuredMeshProperties{D, S, PA, Nothing, Nothing, Nothing, Nothing}

function AgentPoint(
    dims::Int;
    propertiesAgent::Union{NamedTuple, Nothing}=nothing,
)

    UnstructuredMeshProperties{dims, typeof(propertiesAgent), Nothing, Nothing, Nothing, Nothing, Nothing}(
        dims,
        propertiesAgent=propertiesAgent,
        propertiesNode=nothing,
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