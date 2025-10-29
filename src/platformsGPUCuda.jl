import CellBasedModels: AbstractPlatform, GPU, platform

abstract type GPUCuda <: GPU end
abstract type GPUCuDevice <: GPU end

function platform(::CUDA.CuArray)
    return GPUCuda
end

function platform(::CUDA.CuDeviceArray)
    return GPUCuDevice
end
