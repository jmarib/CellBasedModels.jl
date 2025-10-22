module CellBasedModelsGPUExt

    using CellBasedModels
    using CUDA
    using Adapt
    import CUDA: CuArray
    import StaticArrays: SizedVector

    include("../src/AgentStructure/agentPointGPU.jl")

end