import CellBasedModels: MultiMeshObject
import CellBasedModels: toCPU, toGPU, CPU, CPUSinglethread, CPUMultithreading

toCPU(mm::MultiMeshObject{P}) where {P<:CPU} = mm
toGPU(mm::MultiMeshObject{P}) where {P<:GPU} = mm

function toCPU(mm::MultiMeshObject{P}) where {P<:GPUCuda}
    MultiMeshObject(
        NamedTuple{keys(mm._meshes)}(
            [toCPU(mm._meshes[n]) for n in keys(mm._meshes)]
        )
    )
end

function toGPU(mm::MultiMeshObject{P}) where {P<:CPU}
    MultiMeshObject(
        NamedTuple{keys(mm._meshes)}(
            [toGPU(mm._meshes[n]) for n in keys(mm._meshes)]
        )
    )
end