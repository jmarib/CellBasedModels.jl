# using CUDA
using CellBasedModels
using Test
using CUDA   
using BenchmarkTools
using Printf

verbose = true
benchmark = true #just for internal optimizations

@testset verbose=true "CellBasedModels.jl" begin
    using CUDA
    # include("testIndexing.jl")
    # include("testParameter.jl")

    include("testUnstructuredMesh.jl")
    # include("testStructuredMesh.jl")
    # include("testMultiMesh.jl")
    # include("testAddFunctions.jl")

    # include("testAgentPoint.jl")
end

if benchmark

    include("test.jl")

    N = 100000
    n = 10000

    # include("benchmarkCommunityIndices.jl")
    # include("benchmarkRecursiveCachedArrays.jl")
    # include("benchmarkCommunityPoint.jl")
    # include("benchmarkPointGPUIterator.jl")

end