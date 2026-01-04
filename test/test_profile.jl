using BenchmarkTools
import InteractiveUtils: @code_warntype
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector
using Profile
using Profile.Allocs
using PProf

            # 3D
            box = [0.0 1.0; 0.0 1.0; 0.0 1.0]
            cellSize = [0.1, 0.1, 0.1]
            mesh = UnstructuredMesh(3; n  = Node((nnCL=Int,nnFull=Int)))
            @addRule model=mesh scope=integrator function get_neighbors3D(uNew, u, p, t)
                x1 = y1 = z1 = 0.0
                x2 = y2 = z2 = 0.0
                for i in iterateOver(u.n)
                    u.n.nnFull[i] = 0
                    u.n.nnCL[i] = 0
                    x1 = u.n.x[i]
                    y1 = u.n.y[i]
                    z1 = u.n.z[i]
                    # Full neighbors
                    for j in iterateOver(u.n)
                        i == j ? continue : nothing
                        x2 = u.n.x[j]
                        y2 = u.n.y[j]
                        z2 = u.n.z[j]
                        dist = sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
                        if dist <= 0.1
                            uNew.n.nnFull[i] += 1
                        end
                    end
                    # Cell-linked neighbors
                    for j in iterateOverNeighbors(u, :n, x1, y1, z1)
                        i == j ? continue : nothing
                        x2 = u.n.x[j]
                        y2 = u.n.y[j]
                        z2 = u.n.z[j]
                        dist = sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
                        # println("Index: ", (i,j), (x1,y1,z1), (x2,y2,z2), dist)
                        # println("i: $i, j: $j, dist: $dist")
                        if dist <= 0.1
                            uNew.n.nnCL[i] += 1
                        end
                    end
                end
                CellBasedModels.update!(uNew)
            end
            obj = UnstructuredMeshObject(mesh, n = 10000, neighbors=NeighborsCellLinked(box=box, cellSize=cellSize))

            obj.n.x .= rand(10000)
            obj.n.y .= rand(10000)
            obj.n.z .= rand(10000)

            problem = CBProblem(mesh, obj, (0.0, 1.0))
            integrator = init(problem, dt=0.1)
            @btime step!(integrator)