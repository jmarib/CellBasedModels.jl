using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector
import Base: Broadcast

struct MyStruct{A, B}
    x::Float64
end

const sMyStruct{A} = MyStruct{A, 1} where {A}

println(sMyStruct{1}(1))
println(typeof(sMyStruct{1}(1)))
println(typeof(MyStruct{1,1}(1)))

