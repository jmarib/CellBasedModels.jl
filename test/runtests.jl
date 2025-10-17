# using CUDA
using CellBasedModels
using Test
using CUDA   

verbose = true
benchmark = false #just for internal optimizations

@testset verbose=true "CellBasedModels.jl" begin
    using CUDA
    # include("testIndexing.jl")
    # include("testParameter.jl")
    # include("testRecursiveCachedArrays.jl")
    # include("testTypeIntegrators.jl")
    # include("testAgentGlobal.jl")
    # include("testABM.jl")
    include("testAgentPoint.jl")
end

if benchmark

    using BenchmarkTools
    using Printf

    include("benchmarkRecursiveCachedArrays.jl")

end