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

    props = (
            a = Parameter(Float64, description="param a", dimensions=:M, defaultValue=1.0),
            b = Parameter(Int, description="param b", dimensions=:count, defaultValue=0),
        )

show(typeof(UnstructuredMesh(2; n=Node(props))))