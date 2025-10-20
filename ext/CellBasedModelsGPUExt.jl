module CellBasedModelsGPUExt

    using CellBasedModels
    using CUDA
    using Adapt
    import CellBasedModels: CommunityPointMeta, toCPU, toGPU
    import CUDA: CuArray
    import StaticArrays: SizedVector

    include("../src/AgentStructure/agentPointGPU.jl")

end