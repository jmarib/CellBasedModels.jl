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

# @addRule(model=mesh, scope=mechanics,
# function f!(du, u, p, t)

#     for i in 1:length(u.a)
#         du.a.a[i] = rand()
#         du.a.b[i] += 1
#     end

# end
# )

@addODE(model=mesh, scope=biochemistry, broadcasting=true,
function g!(du, u, p, t)

    du.n.x .= 1

end
)

println(mesh)

N = 10000
function evolve!(integrator, steps)
    for i in 1:steps
        step!(integrator)
    end
end

meshObject = UnstructuredMeshObject(
        mesh,
        # agentN=10,
        nodeN=N,
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

@btime evolve!(integrator, 10)

f!(du,u,p,t) = begin
    du .= 1
end

x = zeros(N)

problem = ODEProblem(f!, x, (0.0, 1.0))
integrator = init(
    problem,
    Euler(),
    dt=0.1
)

@btime evolve!(integrator, 10)
