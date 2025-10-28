module CellBasedModelsGPUExt

    using CellBasedModels
    using CUDA
    using Adapt
    import CUDA: CuArray
    import StaticArrays: SizedVector, SizedArray
    import CellBasedModels: toCPU, toGPU

    include("../src/AgentStructure/unstructuredMeshGPU.jl")

end