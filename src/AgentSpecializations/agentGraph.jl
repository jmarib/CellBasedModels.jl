abstract type AgentGraphModel end
const AgentGraph{D, PN} = UnstructuredMesh{D, AgentGraphModel, Nothing, PN, Nothing, Nothing, Nothing}
const AgentGraphObject{P, D, DT, NN, PN} = UnstructuredMeshObject{P, D, AgentGraphModel, DT, NN, Nothing, PN, Nothing, Nothing, Nothing}

function AgentGraph(
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
        specialization=AgentGraphModel,
    )

end

function AgentGraphObject(
        mesh::AgentGraph;
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
# Functions for working with Graph
##########################################################################

function iterateNodes(mesh::AgentGraphObject)
    1:mesh.n._N[]
end

function iterateEdges(mesh::AgentGraphObject)
    1:mesh.e._N[]
end

function iterateCells(mesh::AgentGraphObject)
    1:mesh.c._N[]
end

function divideEdge()

end

function addEdge()

end

function addEdgeBetween()

end

function removeEdge()

end

function divideCellThroughNode()

end

function divideCellThroughEdge()

end

function removeCell()

end