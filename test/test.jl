using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector

props = (
        a = Parameter(Float64, description="param a", dimensions=:M, defaultValue=1.0),
        b = Parameter(Int, description="param b", dimensions=:count, defaultValue=0),
    )

mesh = UnstructuredMesh(
    3;
    propertiesAgent  = props,
)

@addRule mesh function f!(du, u, p)
    du.a.x[i] += 1
    x = du.a
    x = 0
    y = u.a
    for i in 1:10
        y.y.j[i] += i
        y = 0
    end
end

function f(x::Vector{Float64})
    return x .^ 2
end

function f(x::Matrix{Float64})
    return x .^ 2
end

function g(x)
    f(x)
    return
end

function g2(x::Vector{Float64})
    f(x)
    return
end

A = Any[rand(1)]  # store as Any to hide type
@btime for i in 1:10000 g((A[1])) end   # dynamic dispatch!
@btime for i in 1:10000 g2((A[1])) end  # still specialized
