abstract type AgentVertex2DModel end
const AgentVertex2D{PN} = UnstructuredMesh{2, AgentVertex2DModel, Nothing, PN, Nothing, Nothing, Nothing}
const AgentVertex2DObject{P, DT, NN, PN} = UnstructuredMeshObject{P, 2, AgentVertex2DModel, DT, NN, Nothing, PN, Nothing, Nothing, Nothing}

function AgentVertex2D(
    propertiesNode::Union{NamedTuple, Nothing}=nothing,
    propertiesEdge::Union{NamedTuple, Nothing}=nothing,
    propertiesCell::Union{NamedTuple, Nothing}=nothing,
)

    UnstructuredMesh(
        2,
        n=propertiesNode,
        e=propertiesEdge,
        c=propertiesCell,
        specialization=AgentVertex2DModel,
    )

end

function AgentVertex2DObject(
        mesh::AgentVertex2D;
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
# Functions for working with Vertex2D
##########################################################################

function iterateNodes(mesh::AgentVertex2DObject)
    1:mesh.n._N[]
end

function iterateEdges(mesh::AgentVertex2DObject)
    1:mesh.e._N[]
end

function iterateCells(mesh::AgentVertex2DObject)
    1:mesh.c._N[]
end

function divideCell()

end

function splitEdge()

end

function separateEdges()

end

function mergeCells()

end

function removeCell()

end

function removeEdge()

end

