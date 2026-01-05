abstract type AgentVertex3DModel end
const AgentVertex3D{PN} = UnstructuredMesh{3, AgentVertex3DModel, Nothing, PN, Nothing, Nothing, Nothing}
const AgentVertex3DObject{P, DT, NN, PN} = UnstructuredMeshObject{P, 3, AgentVertex3DModel, DT, NN, Nothing, PN, Nothing, Nothing, Nothing}

function AgentVertex3D(
    propertiesNode::Union{NamedTuple, Nothing}=nothing,
    propertiesFace::Union{NamedTuple, Nothing}=nothing,
    propertiesCell::Union{NamedTuple, Nothing}=nothing,
)

    UnstructuredMesh(
        3,
        n=propertiesNode,
        f=propertiesFace,
        c=propertiesCell,
        specialization=AgentVertex3DModel,
    )

end

function AgentVertex3DObject(
        mesh::AgentVertex3D;
        n::Union{Integer,Tuple{Integer, Integer}}=0,
        f::Union{Integer,Tuple{Integer, Integer}}=0,
        c::Union{Integer,Tuple{Integer, Integer}}=0,
    )

    UnstructuredMeshObject(
        mesh;
        (n=n, f=f, c=c),
    )

end

##########################################################################
# Functions for working with Vertex3D
##########################################################################

function iterateNodes(mesh::AgentVertex3DObject)
    1:mesh.n._N[]
end

function iterateFaces(mesh::AgentVertex3DObject)
    1:mesh.f._N[]
end

function iterateCells(mesh::AgentVertex3DObject)
    1:mesh.c._N[]
end

function divideCell()

end

function splitFace()

end

function separateFaces()

end

function mergeCells()

end

function removeCell()

end