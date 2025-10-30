import CellBasedModels: AbstractPlatform, AbstractMeshObject, GPU, platform, toGPU

abstract type GPUCuda <: GPU end
abstract type GPUCuDevice <: GPU end

toGPU(x::AbstractMeshObject) = @error "GPU found but no conversion for type $(typeof(x))"

function platform(::CUDA.CuArray)
    return GPUCuda
end

function platform(::CUDA.CuDeviceArray)
    return GPUCuDevice
end
