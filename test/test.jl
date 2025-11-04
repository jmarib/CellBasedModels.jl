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

@addRule(model=mesh, scope=mechanics,
function f!(du, u, p, t)
    for i in 1:length(du.a)
        du.a.a[i] = 0
    end
end
)

@addODE(model=mesh, scope=biochemistry,
function g!(du, u, p, t)
    a = du.a
    n = du.n
    for i in 1:length(a)
        a.a[i] += 1
        n.x[i] += 1
    end
end
)

println(mesh)

meshObject = UnstructuredMeshObject(
        mesh,
        agentN=10,
        nodeN=5,
    )

problem = CBProblem(
    mesh,
    meshObject,
    (0.0, 1.0),
)
integrator = init(
    problem,
    dt=0.1
)

# for i in 1:10
#     step!(integrator)
#     # println(integrator.u.n.x)
# end
