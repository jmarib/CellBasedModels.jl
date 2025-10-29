module CellBasedModelsGPUCudaExt

    using CellBasedModels
    using CUDA
    using Adapt
    import CUDA: CuArray
    import StaticArrays: SizedVector, SizedArray

    include("../src/platformsGPUCuda.jl")
    include("../src/AgentStructure/unstructuredMeshGPUCuda.jl")

end