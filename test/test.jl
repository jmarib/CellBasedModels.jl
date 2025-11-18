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
        b = Parameter(Float64, description="param b", dimensions=:count, defaultValue=0.0),
    )

mesh = UnstructuredMesh(
    3;
    propertiesNode  = props,
)

@addSDE(model=mesh, scope=biochemistry, broadcasting=true,
function f!(du, u, p, t)

    du.n.a .= 0
    du.n.b .= 0

end, 
function g!(du, u, p, t)

    du.n.a .= 1.0
    du.n.b .= 1.0

end
)

meshObject = UnstructuredMeshObject(
        mesh,
        nodeN = 5
    )

# detector = SparseConnectivityTracer.TracerSparsityDetector()
# dmeshObject = copy(meshObject)
# jac_sparsity = ADTypes.jacobian_sparsity(
#     (du, u) -> g!(du, u, tuple(), 0.0), dmeshObject, meshObject, detector)

# println("Jacobian sparsity pattern:")
# display(jac_sparsity)

problem = CBProblem(
    mesh,
    meshObject,
    (0.0, 1.0),
)
integrator = init(
    problem,
    dt=0.1,
    biochemistry=ImplicitEM()
)

for i in 1:10
    step!(integrator)
    println("Step $i: ", integrator.u.n.a)
    println("Step $i: ", integrator.u.n.b)
end

# @btime evolve!(integrator, T)

# # Profile.clear()
# # @profile evolve!(integrator, T)
# # # pprof(; webport=8080, webhost="localhost")
# # Profile.print(format=:flat, noisefloor=2, sortedby=:count)

# f!(du,u,p,t) = begin
#     du .= 1
# end

# x = zeros(N)

# problem = ODEProblem(f!, x, (0.0, 1.0))
# integrator2 = init(
#     problem,
#     Euler(),
#     dt=0.1,
#     save_everystep=false,
# )

# @btime evolve!(integrator2, T)

# # Profile.clear()
# # @profile evolve!(integrator2, T)
# # # pprof(; webport=8080, webhost="localhost")
# # Profile.print(format=:flat, noisefloor=2, sortedby=:count)

# # function f!(integrator)
# #     DifferentialEquations.DiffEqBase.@.. integrator.integrators.biochemistry.u = integrator.integrators.biochemistry.u + 2.0
# # end
# # @btime f!(integrator)
# # function f2!(integrator)
# #     DifferentialEquations.DiffEqBase.@.. integrator2.u = integrator2.u + 2.0
# # end
# # @btime f2!(integrator2)
