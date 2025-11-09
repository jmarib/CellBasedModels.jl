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

mesh = StructuredMesh(
    3;
    propertiesCell  = props,
)

@addODE(model=mesh, scope=biochemistry, broadcasting=true,
function g!(du, u, p, t)

    du.c.a .= u.c.a

end
)

meshObject = StructuredMeshObject(
            mesh,
            simulationBox=[0 1;0 1;0 1], gridSpacing=[0.1, 0.1, 0.1]
        )

detector = SparseConnectivityTracer.TracerSparsityDetector()
dmeshObject = copy(meshObject)
jac_sparsity = ADTypes.jacobian_sparsity(
    (du, u) -> g!(du, u, tuple(), 0.0), dmeshObject, meshObject, detector)

println("Jacobian sparsity pattern:")
display(jac_sparsity)

# problem = CBProblem(
#     mesh,
#     meshObject,
#     (0.0, 1.0),
# )
# integrator = init(
#     problem,
#     dt=0.1,
#     biochemistry=Euler()
# )

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
