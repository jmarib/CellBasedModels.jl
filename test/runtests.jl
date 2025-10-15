# using CUDA
using CellBasedModels
using Test
using DataFrames
import MacroTools: prettify
using OrderedCollections
using Distributions
using Random

verbose = true
benchmark = true

@testset verbose=true "CellBasedModels.jl" begin
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