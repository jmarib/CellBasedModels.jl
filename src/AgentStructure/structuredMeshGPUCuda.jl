import CellBasedModels: StructuredMeshObject
import CellBasedModels: toCPU, toGPU, CPU, CPUSinglethread, CPUMultithreading, platform
import StaticArrays: SizedVector, SizedMatrix

toCPU(mesh::StructuredMeshObject{P}) where {P<:CPU} = mesh
toGPU(mesh::StructuredMeshObject{P}) where {P<:GPU} = mesh

function toCPU(mesh::StructuredMeshObject{P, D, S}) where {P<:GPUCuda, D, S}

    PNew = platform()

    p = Adapt.adapt(Base.Array, mesh.p)
    PR = typeof(p)

    _pReference = SizedVector{length(mesh), Bool}([mesh._pReference...])
    PRef = typeof(_pReference)

    simulationBox = SizedMatrix{size(mesh._simulationBox)..., CellBasedModels.standardDataType(AbstractFloat)}(Array(mesh._simulationBox))
    SB = typeof(simulationBox)

    gridSpacing = SizedVector{D, CellBasedModels.standardDataType(AbstractFloat)}(Array(mesh._gridSpacing))
    NG = typeof(gridSpacing)

    StructuredMeshObject{PNew, D, S, PR, PRef, SB, NG}(
        p,
        _pReference,
        simulationBox,
        gridSpacing
    )
end

function toGPU(mesh::StructuredMeshObject{P, D, S}) where {P<:CPU, D, S}

    PNew = GPUCuda

    p = Adapt.adapt(CUDA.CuArray, mesh.p)
    PR = typeof(p)

    _pReference = tuple(mesh._pReference...)
    PRef = typeof(_pReference)

    simulationBox = Adapt.adapt(CUDA.CuArray, mesh._simulationBox)
    SB = typeof(simulationBox)

    gridSpacing = Adapt.adapt(CUDA.CuArray, mesh._gridSpacing)
    NG = typeof(gridSpacing)

    StructuredMeshObject{PNew, D, S, PR, PRef, SB, NG}(
        p,
        _pReference,
        simulationBox,
        gridSpacing
    )

end