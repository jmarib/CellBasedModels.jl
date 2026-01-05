abstract type AgentAgentPolylineModel end
const AgentAgentPolyline{D, PN} = UnstructuredMesh{D, AgentAgentPolylineModel, Nothing, PN, Nothing, Nothing, Nothing}
const AgentAgentPolylineObject{P, D, DT, NN, PN} = UnstructuredMeshObject{P, D, AgentAgentPolylineModel, DT, NN, Nothing, PN, Nothing, Nothing, Nothing}

function AgentAgentPolyline(
    dims::Int;
    propertiesNode::Union{NamedTuple, Nothing}=nothing,
    propertiesEdge::Union{NamedTuple, Nothing}=nothing,
    propertiesCell::Union{NamedTuple, Nothing}=nothing,
)

    UnstructuredMesh(
        dims,
        n=propertiesNode,
        e=propertiesEdge,
        c=propertiesCell,
        specialization=AgentAgentPolylineModel,
    )

end

function AgentAgentPolylineObject(
        mesh::AgentAgentPolyline;
        n::Union{Integer,Tuple{Integer, Integer}}=0,
        e::Union{Integer,Tuple{Integer, Integer}}=0,
        c::Union{Integer,Tuple{Integer, Integer}}=0,
    )

    UnstructuredMeshObject(
        mesh;
        (n=n, e=e, c=c),
    )

end

##########################################################################
# Functions for working with AgentPolyline
##########################################################################

function iterateNodes(mesh::AgentAgentPolylineObject)
    1:mesh.n._N[]
end

function iterateEdges(mesh::AgentAgentPolylineObject)
    1:mesh.e._N[]
end

function iterateCells(mesh::AgentAgentPolylineObject)
    1:mesh.c._N[]
end

function divideEdge()

end

function addEdge()

end

function removeEdge()

end

function divideCellThroughNode()

end

function divideCellThroughEdge()

end

function removeCell()

end