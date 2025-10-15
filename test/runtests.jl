# using CUDA
using CellBasedModels
using Test

verbose = true
benchmark = true

@testset verbose=true "CellBasedModels.jl" begin
    using CUDA
    # include("testIndexing.jl")
    # include("testParameter.jl")
    include("testRecursiveCachedArrays.jl")
    # include("testTypeIntegrators.jl")
    # include("testAgentGlobal.jl")
    # include("testABM.jl")
end

if benchmark

    using BenchmarkTools

    include("benchmarkRecursiveCachedArrays.jl")

end