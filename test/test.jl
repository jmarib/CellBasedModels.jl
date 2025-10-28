using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector
import CellBasedModels

    props = (
            a = Parameter(Float64, description="param a", dimensions=:M, defaultValue=1.0),
            b = Parameter(Int, description="param b", dimensions=:count, defaultValue=0),
        )

    all_scopes = (:propertiesAgent, :propertiesNode, :propertiesEdge, :propertiesFace, :propertiesVolume)

    mesh = UnstructuredMesh(
        3,
        propertiesAgent=props,
    )
    field = UnstructuredMeshObject(mesh; agentN=3, agentNCache=5)

    println(field.a._pReference)
    field.a._pReference .= [true, true, true, true, false]
    fieldCopy = copy(field.a)
