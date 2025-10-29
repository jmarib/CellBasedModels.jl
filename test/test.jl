using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector

import Base: Broadcast

props = (a=Float64, b=Float64)
mesh = UnstructuredMesh(0; propertiesAgent=props, scopePosition=:propertiesAgent)

function fODE!(du, u, p, t)
    @. du.a.a = 1
    return
end
obj = UnstructuredMeshObject(mesh, 
    agentN=2, agentNCache=4, 
    nodeN=2, nodeNCache=4, 
    edgeN=2, edgeNCache=4, 
    faceN=2, faceNCache=4, 
    volumeN=2, volumeNCache=4
)
obj.a._pReference .= (false, false)

prob = ODEProblem(fODE!, obj, (0.0, 1.0))
integrator = init(prob, Euler(), dt=0.1, save_everystep=false)
for i in 1:10
    step!(integrator)
    println(integrator.u.a.a)
end

