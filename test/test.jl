using BenchmarkTools
import InteractiveUtils: @code_warntype
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector
using Profile
using Profile.Allocs
using PProf
import SparseConnectivityTracer, ADTypes

function g!(x, b)

    c = 0
    for i in eachindex(x)
        if i % 2 == 0
            c += 1
            x[i] = 1
        end
    end

end

function f!(x)

    # a = ntuple(_ -> 1, 16)
    b = CuStaticSharedArray(Int32, 100)

    g!(x, b)

    return

end

CUDA.@cuda threads=1 f!(CUDA.zeros(100))

occ = CUDA.occupancy(f!, blocksize)
println(occ.blocks)  # active blocks per SM
println(occ.threads) # active threads per SM
println(occ.occupancy)