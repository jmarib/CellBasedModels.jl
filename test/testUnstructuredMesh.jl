using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector
using KernelAbstractions

@testset verbose = verbose "ABM - UnstructuredMesh" begin

    #######################################################################
    # HELPERS
    #######################################################################
    props = (
            a = Parameter(Float64, description="param a", dimensions=:M, defaultValue=1.0),
            b = Parameter(Int, description="param b", dimensions=:count, defaultValue=0),
        )

    all_scopes = (Node, Edge, Face, Volume, Agent)
    all_scopes_index = (:n, :e, :f, :v, :a)

    # #######################################################################
    # # TEST 1: UnstructuredMesh
    # #######################################################################
    # @testset "UnstructuredMesh - full coverage" begin
    #     for dims in 0:3
    #         for scope in all_scopes
    #             mesh = nothing
    #             if scope === Node
    #                 mesh = UnstructuredMesh(dims; n  = Node(props))
    #             elseif scope === Edge
    #                 mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
    #             elseif scope === Face
    #                 mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
    #             elseif scope === Volume
    #                 mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
    #             elseif scope === Agent
    #                 mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
    #             end
    #             @test mesh isa UnstructuredMesh
    #             @test CellBasedModels.spatialDims(mesh) == dims
    #             @test CellBasedModels.specialization(mesh) === Nothing
    #         end
    #     end

    #     # Invalid dimension
    #     @test_throws ErrorException UnstructuredMesh(-1)
    #     @test_throws ErrorException UnstructuredMesh(4)

    #     # Duplicate protected parameter name (e.g., x)
    #     bad_props = (x = Parameter(Float64, description="duplicate", dimensions=:L, defaultValue=0.0),)
    #     @test_throws ErrorException UnstructuredMesh(1; n=Node(bad_props))

    #     # Show output test
    #     io = IOBuffer()
    #     show(io, UnstructuredMesh(2; n=Node(props)))
    #     output = String(take!(io))
    #     # show(UnstructuredMesh(2; n=Node(props)))
    #     @test occursin("UnstructuredMesh with dimensions 2", output)

    #     # Show output test
    #     io = IOBuffer()
    #     show(io, typeof(UnstructuredMesh(2; n=Node(props))))
    #     output = String(take!(io))
    #     # show(typeof(UnstructuredMesh(2; n=Node(props))))
    #     @test occursin("UnstructuredMesh{dims=2", output)
    # end

    # #######################################################################
    # # TEST 2: UnstructuredMeshField
    # #######################################################################
    # @testset "UnstructuredMeshField - construction and logic" begin

    #     # Valid case
    #     meshProperty = Node(props)
    #     field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
    #     @test field isa UnstructuredMeshField
    #     @test field._N[] == 3
    #     @test field._NCache[] == 5
    #     @test length(field._FlagsSurvived) == 5
    #     @test field._FlagOverflow[] == false
    #     @test field._idMax[] == 3
    #     @test field._NP == length(props)
    #     @test field._pReference isa SizedVector

    #     # Get properties
    #     @test field.a == zeros(Float64, 3)
    #     @test field._p.a == zeros(Float64, 5)

    #     # With id=false
    #     field2 = UnstructuredMeshField(meshProperty; N=2, NCache=5, id=false)
    #     @test field2._id === nothing
    #     @test field2._idMax === nothing

    #     # Show output formatting
    #     io = IOBuffer()
    #     show(io, field)
    #     txt = String(take!(io))
    #     @test occursin("_id", txt)
    #     @test occursin("_AddedAgents", txt)

    #     # Copy
    #     field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
    #     field._pReference .= [true, false]
    #     fieldCopy = copy(field)
    #     field.a .= 7.0
    #     field.b .= 2
    #     @test fieldCopy.a == fill(7.0, 3)
    #     @test fieldCopy.b == zeros(Int, 3)

    #     # zero
    #     field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
    #     field._pReference .= [true, false]
    #     field.a .= 7.0
    #     field.b .= 2        
    #     fieldZero = zero(field)
    #     @test fieldZero.a == fill(7.0, 3)
    #     @test fieldZero.b == zeros(Int, 3)

    #     # toGPU/toCPU
    #     if CUDA.has_cuda()
    #         field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
    #         field._pReference .= [true, false]
    #         field.a .= 7.0
    #         field.b .= 2

    #         field_gpu = toGPU(field)

    #         @test typeof(field_gpu._id) <: CUDA.CuArray
    #         @test typeof(field_gpu.a) <: CUDA.CuArray

    #         field_cpu = toCPU(field_gpu)

    #         @test typeof(field_cpu) == typeof(field)
    #     end

    #     # Copyto!
    #     field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
    #     field._pReference .= [false, true]
    #     field.a .= 7.0
    #     field.b .= 2
    #     field2 = UnstructuredMeshField(meshProperty; N=3, NCache=5)
    #     copyto!(field2, field)
    #     @test field2.a == fill(7.0, 3)
    #     @test field2.b == fill(0, 3)
    #     if CUDA.has_cuda()
    #         field_gpu = toGPU(field)
    #         field2_gpu = toGPU(field2)
    #         copyto!(field2_gpu, field_gpu)
    #         @test Array(field2_gpu.a) == fill(7.0, 3)
    #         @test Array(field2_gpu.b) == fill(0, 3)
    #     end

    #     # Broadcast
    #     field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
    #     field._pReference .= [false, true]
    #     field.a .= 7.0
    #     field.b .= 2
    #     field2 = UnstructuredMeshField(meshProperty; N=3, NCache=5)
    #     field2._pReference .= [false, true]
    #     @. field2 = field * 0.1 + 3.0
    #     @test field2.a == fill(3.7, 3)
    #     @test field2.b == fill(0, 3)
    #     if CUDA.has_cuda()
    #         field_gpu = toGPU(field)
    #         field2_gpu = toGPU(field2)
    #         @. field2_gpu = field_gpu * 0.1 + 3.0
    #         @test Array(field2_gpu.a) == fill(3.7, 3)
    #         @test Array(field2_gpu.b) == fill(0, 3)
    #     end

    #     # Broadcast @..
    #     field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
    #     field._pReference .= [false, true]
    #     field.a .= 7.0
    #     field.b .= 2
    #     field2 = UnstructuredMeshField(meshProperty; N=3, NCache=5)
    #     field2._pReference .= [false, true]
    #     DifferentialEquations.DiffEqBase.@.. field2 = field * 0.1 + 3.0
    #     @test field2.a == fill(3.7, 3)
    #     @test field2.b == fill(0, 3)
    #     if CUDA.has_cuda()
    #         field_gpu = toGPU(field)
    #         field2_gpu = toGPU(field2)
    #         DifferentialEquations.DiffEqBase.@.. field2_gpu = field_gpu * 0.1 + 3.0
    #         @test Array(field2_gpu.a) == fill(3.7, 3)
    #         @test Array(field2_gpu.b) == fill(0, 3)
    #     end

    #     # Kernel works
    #     if CUDA.has_cuda()

    #         function test_kernel_field!(x)
    #             x.a[1]
    #             nothing
    #         end

    #         field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
    #         field._pReference .= [true, false]
    #         field_gpu = toGPU(field)

    #         @test_nowarn CUDA.@cuda test_kernel_field!(field_gpu)

    #     end

    # end

    # #######################################################################
    # # TEST 3: UnstructuredMeshObject
    # #######################################################################
    # @testset "UnstructuredMeshObject - construction" begin
    #     for dims in 0:3
    #         for (scope, scope_index) in zip(all_scopes, all_scopes_index)
    #             mesh = nothing
    #             if scope === Node
    #                 mesh = UnstructuredMesh(dims; n  = Node(props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4))
    #                 @test obj isa UnstructuredMeshObject
    #                 @test obj.n isa UnstructuredMeshField
    #             elseif scope === Edge
    #                 mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
    #                 @test obj isa UnstructuredMeshObject
    #                 @test obj.n isa UnstructuredMeshField
    #                 @test obj.e isa UnstructuredMeshField
    #             elseif scope === Face
    #                 mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
    #                 @test obj isa UnstructuredMeshObject
    #                 @test obj.n isa UnstructuredMeshField
    #                 @test obj.f isa UnstructuredMeshField
    #             elseif scope === Volume
    #                 mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
    #                 @test obj isa UnstructuredMeshObject
    #                 @test obj.n isa UnstructuredMeshField
    #                 @test obj.v isa UnstructuredMeshField
    #             elseif scope === Agent
    #                 mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
    #                 @test obj isa UnstructuredMeshObject
    #                 @test obj.n isa UnstructuredMeshField
    #                 @test obj.a isa UnstructuredMeshField
    #             end

    #             # Invalid cache values
    #             @test_throws ErrorException UnstructuredMeshObject(mesh, agentN=4, agentNCache=2)
    #             @test_throws ErrorException UnstructuredMeshObject(mesh, agentN=-1, agentNCache=0)
    #         end
    #     end
    # end

    # @testset "UnstructuredMeshObject - copy" begin
    #     for dims in 0:3
    #         for (scope, scope_index) in zip(all_scopes, all_scopes_index)
    #             mesh = nothing
    #             obj = nothing
    #             if scope === Node
    #                 mesh = UnstructuredMesh(dims; n  = Node(props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4))
    #             elseif scope === Edge
    #                 mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
    #             elseif scope === Face
    #                 mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
    #             elseif scope === Volume
    #                 mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
    #             elseif scope === Agent
    #                 mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
    #             end
    #             if scope == Node
    #                 obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., true, false]
    #             else
    #                 obj[scope_index]._pReference .= [true, false]
    #             end
    #             objCopy = copy(obj)
    #             obj[scope_index].a .= 7.0
    #             obj[scope_index].b .= 2
    #             @test objCopy[scope_index].a == fill(7.0, 2)
    #             @test objCopy[scope_index]._p.a == [7.0, 7.0, 0.0, 0.0]
    #             @test objCopy[scope_index].b == zeros(Int, 2)
    #             @test objCopy[scope_index]._p.b == [0, 0, 0, 0]
    #         end
    #     end
    # end

    # @testset "UnstructuredMeshObject - zero" begin
    #     for dims in 0:3
    #         for (scope, scope_index) in zip(all_scopes, all_scopes_index)
    #             mesh = nothing
    #             obj = nothing
    #             if scope === Node
    #                 mesh = UnstructuredMesh(dims; n  = Node(props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4))
    #             elseif scope === Edge
    #                 mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
    #             elseif scope === Face
    #                 mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
    #             elseif scope === Volume
    #                 mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
    #             elseif scope === Agent
    #                 mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
    #             end
    #             if scope == Node
    #                 obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., true, false]
    #             else
    #                 obj[scope_index]._pReference .= [true, false]
    #             end
    #             obj[scope_index].a .= 7.0
    #             obj[scope_index].b .= 2
    #             objZero = zero(obj)
    #             @test objZero[scope_index].a == fill(7.0, 2)
    #             @test objZero[scope_index]._p.a == [7.0, 7.0, 0.0, 0.0]
    #             @test objZero[scope_index].b == zeros(Int, 2)
    #             @test objZero[scope_index]._p.b == [0, 0, 0, 0]
    #         end
    #     end
    # end

    # @testset "UnstructuredMeshObject - toGPU/toCPU" begin
    #     for dims in 0:3
    #         for (scope, scope_index) in zip(all_scopes, all_scopes_index)
    #             # toGPU/toCPU
    #             if CUDA.has_cuda()
    #                 mesh = nothing
    #                 obj = nothing
    #                 if scope === Node
    #                     mesh = UnstructuredMesh(dims; n  = Node(props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4))
    #                 elseif scope === Edge
    #                     mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
    #                 elseif scope === Face
    #                     mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
    #                 elseif scope === Volume
    #                     mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
    #                 elseif scope === Agent
    #                     mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
    #                 end
    #                 if scope == Node
    #                     obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., true, false]
    #                 else
    #                     obj[scope_index]._pReference .= [true, false]
    #                 end
    #                 obj[scope_index].a .= 7.0
    #                 obj[scope_index].b .= 2
    #                 obj_gpu = toGPU(obj)
    #                 @test typeof(obj_gpu[scope_index]._id) <: CUDA.CuArray
    #                 @test typeof(obj_gpu[scope_index].a) <: CUDA.CuArray
    #                 obj_cpu = toCPU(obj_gpu)
    #                 @test typeof(obj_cpu) == typeof(obj)
    #             end
    #         end
    #     end
    # end

    # @testset "UnstructuredMeshObject - copyto!" begin
    #     for dims in 0:3
    #         for (scope, scope_index) in zip(all_scopes, all_scopes_index)
    #                 mesh = nothing
    #                 obj = nothing
    #                 if scope === Node
    #                     mesh = UnstructuredMesh(dims; n  = Node(props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4))
    #                     obj2 = UnstructuredMeshObject(mesh, n = (2,4))
    #                 elseif scope === Edge
    #                     mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
    #                     obj2 = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
    #                 elseif scope === Face
    #                     mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
    #                     obj2 = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
    #                 elseif scope === Volume
    #                     mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
    #                     obj2 = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
    #                 elseif scope === Agent
    #                     mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
    #                     obj2 = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
    #                 end
    #             if scope == Node
    #                 obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., true, false]
    #             else
    #                 obj[scope_index]._pReference .= [true, false]
    #             end
    #             obj[scope_index].a .= 7.0
    #             obj[scope_index].b .= 2
    #             if scope == Node
    #                 obj2[scope_index]._pReference .= [[true, true, true][1:1:dims]..., true, false]
    #             else
    #                 obj2[scope_index]._pReference .= [true, false]
    #             end
    #             copyto!(obj2, obj)
    #             @test obj2[scope_index].a == fill(0.0, 2)
    #             @test obj2[scope_index]._p.a == [0.0, 0.0, 0.0, 0.0]
    #             @test obj2[scope_index].b == fill(2, 2)
    #             @test obj2[scope_index]._p.b == [2, 2, 0, 0]
    #             if CUDA.has_cuda()
    #                 obj_gpu = toGPU(obj)
    #                 obj2_gpu = toGPU(obj2)
    #                 copyto!(obj2_gpu, obj_gpu)
    #                 @test Array(obj2_gpu[scope_index].a) == fill(0.0, 2)
    #                 @test Array(obj2_gpu[scope_index]._p.a) == [0.0, 0.0, 0.0, 0.0]
    #                 @test Array(obj2_gpu[scope_index].b) == fill(2, 2)
    #                 @test Array(obj2_gpu[scope_index]._p.b) == [2, 2, 0, 0]
    #             end
    #         end
    #     end
    # end

    # @testset "UnstructuredMeshObject - broadcasting" begin
    #     for dims in 0:3
    #         for (scope, scope_index) in zip(all_scopes, all_scopes_index)
    #             mesh = nothing
    #             obj = nothing
    #             if scope === Node
    #                 mesh = UnstructuredMesh(dims; n  = Node(props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4))
    #                 obj2 = UnstructuredMeshObject(mesh, n = (2,4))
    #             elseif scope === Edge
    #                 mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
    #                 obj2 = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
    #             elseif scope === Face
    #                 mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
    #                 obj2 = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
    #             elseif scope === Volume
    #                 mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
    #                 obj2 = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
    #             elseif scope === Agent
    #                 mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
    #                 obj2 = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
    #             end
    #             if scope == Node
    #                 obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., false, true]
    #             else
    #                 obj[scope_index]._pReference .= [false, true]
    #             end
    #             obj[scope_index].a .= 7.0
    #             obj[scope_index].b .= 2
    #             if scope == Node
    #                 obj2[scope_index]._pReference .= [[true, true, true][1:1:dims]..., false, true]
    #             else
    #                 obj2[scope_index]._pReference .= [false, true]
    #             end
    #             @. obj2 = obj * 0.1 + 3.0
    #             @test obj2[scope_index].a == fill(3.7, 2)
    #             @test obj2[scope_index]._p.a == [3.7, 3.7, 0.0, 0.0]
    #             @test obj2[scope_index].b == fill(0, 2)
    #             @test obj2[scope_index]._p.b == [0, 0, 0, 0]
    #             if CUDA.has_cuda()
    #                 obj_gpu = toGPU(obj)
    #                 obj2_gpu = toGPU(obj2)
    #                 obj2_gpu[scope_index].a .= 0.0
    #                 @. obj2_gpu = obj_gpu * 0.1 + 3.0
    #                 @test Array(obj2_gpu[scope_index].a) == fill(3.7, 2)
    #                 @test Array(obj2_gpu[scope_index]._p.a) == [3.7, 3.7, 0.0, 0.0]
    #                 @test Array(obj2_gpu[scope_index].b) == fill(0, 2)
    #                 @test Array(obj2_gpu[scope_index]._p.b) == [0, 0, 0, 0]
    #             end

    #             # Broadcast @..
    #             mesh = nothing
    #             obj = nothing
    #             if scope === Node
    #                 mesh = UnstructuredMesh(dims; n  = Node(props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4))
    #                 obj2 = UnstructuredMeshObject(mesh, n = (2,4))
    #             elseif scope === Edge
    #                 mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
    #                 obj2 = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
    #             elseif scope === Face
    #                 mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
    #                 obj2 = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
    #             elseif scope === Volume
    #                 mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
    #                 obj2 = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
    #             elseif scope === Agent
    #                 mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
    #                 obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
    #                 obj2 = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
    #             end
    #             if scope == Node
    #                 obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., false, true]
    #             else
    #                 obj[scope_index]._pReference .= [false, true]
    #             end
    #             obj[scope_index].a .= 7.0
    #             obj[scope_index].b .= 2
    #             if scope == Node
    #                 obj2[scope_index]._pReference .= [[true, true, true][1:1:dims]..., false, true]
    #             else
    #                 obj2[scope_index]._pReference .= [false, true]
    #             end
    #             DifferentialEquations.DiffEqBase.@.. obj2 = obj * 0.1 + 3.0
    #             @test obj2[scope_index].a == fill(3.7, 2)
    #             @test obj2[scope_index]._p.a == [3.7, 3.7, 0.0, 0.0]
    #             @test obj2[scope_index].b == fill(0, 2)
    #             @test obj2[scope_index]._p.b == [0, 0, 0, 0]
    #             if CUDA.has_cuda()
    #                 obj_gpu = toGPU(obj)
    #                 obj2_gpu = toGPU(obj2)
    #                 obj2_gpu[scope_index].a .= 0.0
    #                 DifferentialEquations.DiffEqBase.@.. obj2_gpu = obj_gpu * 0.1 + 3.0
    #                 @test Array(obj2_gpu[scope_index].a) == fill(3.7, 2)
    #                 @test Array(obj2_gpu[scope_index]._p.a) == [3.7, 3.7, 0.0, 0.0]
    #                 @test Array(obj2_gpu[scope_index].b) == fill(0, 2)
    #                 @test Array(obj2_gpu[scope_index]._p.b) == [0, 0, 0, 0]
    #             end
    #         end
    #     end
    # end

    # @testset "UnstructuredMeshObject - GPU kernels" begin
    #     for dims in 0:3
    #         for (scope, scope_index) in zip(all_scopes, all_scopes_index)            
    #             # Kernel works
    #             if CUDA.has_cuda()
    #                 function test_kernel_object!(x)
    #                     x.n
    #                     nothing
    #                 end

    #                 mesh = nothing
    #                 obj = nothing
    #                 if scope === Node
    #                     mesh = UnstructuredMesh(dims; n  = Node(props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4))
    #                 elseif scope === Edge
    #                     mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
    #                 elseif scope === Face
    #                     mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
    #                 elseif scope === Volume
    #                     mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
    #                 elseif scope === Agent
    #                     mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
    #                 end
    #                 obj_gpu = toGPU(obj)

    #                 @test_nowarn CUDA.@cuda test_kernel_object!(obj_gpu)
    #             end
    #         end
    #     end
    # end

    # @testset "UnstructuredMeshObject - DifferentialEquations compatibility" begin
    #     for dims in 0:3
    #         for (scope, scope_index) in zip(all_scopes, all_scopes_index)
    #             for integratorAlg in (Euler(), RK4())#, Tsit5())

    #                 # DifferentialEquations.jl compatibility
    #                 function fODE!(du, u, p, t)
    #                     @. du[scope_index].a = 1
    #                     return
    #                 end
    #                 mesh = nothing
    #                 obj = nothing
    #                 if scope === Node
    #                     mesh = UnstructuredMesh(dims; n  = Node(props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4))
    #                 elseif scope === Edge
    #                     mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
    #                 elseif scope === Face
    #                     mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
    #                 elseif scope === Volume
    #                     mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
    #                 elseif scope === Agent
    #                     mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
    #                     obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
    #                 end
    #                 if scope == Node
    #                     obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., false, true]
    #                 else
    #                     obj[scope_index]._pReference .= [false, true]
    #                 end
    #                 prob = ODEProblem(fODE!, obj, (0.0, 1.0))
    #                 integrator = init(prob, integratorAlg, dt=0.1, save_everystep=false)
    #                 for i in 1:10
    #                     step!(integrator, 0.1, true)
    #                 end
    #                 @test all(integrator.u[scope_index].a .≈ 1.0)
    #                 if CUDA.has_cuda()
    #                     mesh = nothing
    #                     obj = nothing
    #                     if scope === Node
    #                         mesh = UnstructuredMesh(dims; n  = Node(props))
    #                         obj = UnstructuredMeshObject(mesh, n = (2,4))
    #                     elseif scope === Edge
    #                         mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
    #                         obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
    #                     elseif scope === Face
    #                         mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
    #                         obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
    #                     elseif scope === Volume
    #                         mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
    #                         obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
    #                     elseif scope === Agent
    #                         mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
    #                         obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
    #                     end
    #                     if scope == Node
    #                         obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., false, true]
    #                     else
    #                         obj[scope_index]._pReference .= [false, true]
    #                     end
    #                     obj_gpu = toGPU(obj)
    #                     prob = ODEProblem(fODE!, obj_gpu, (0.0, 1.0))
    #                     integrator = init(prob, integratorAlg, dt=0.1, save_everystep=false)
    #                     for i in 1:10
    #                         step!(integrator, 0.1, true)
    #                     end
    #                     @test all(Array(integrator.u[scope_index].a) .≈ 1.0)
    #                 end
    #             end
    #         end
    #     end
    # end

    @testset "UnstructuredMeshObject - neighbors" begin

        @testset "UnstructuredMeshObject - NeighborsFull" begin
            for dims in 1:3  # Skip dims=0 as neighbor algorithms require spatial coordinates
                for (scope, scope_index) in zip(all_scopes, all_scopes_index)
                    mesh = nothing
                    obj = nothing
                    if scope === Node
                        mesh = UnstructuredMesh(dims; n  = Node(props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), neighbors=NeighborsFull())
                    elseif scope === Edge
                        mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4), neighbors=NeighborsFull())
                    elseif scope === Face
                        mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4), neighbors=NeighborsFull())
                    elseif scope === Volume
                        mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4), neighbors=NeighborsFull())
                    elseif scope === Agent
                        mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4), neighbors=NeighborsFull())
                    end

                    l = []
                    for i in iterateOverNeighbors(obj, scope_index, 1)
                        push!(l, i)
                    end

                    @test length(l) == 2
                
                end
            end
        end

        @testset "UnstructuredMeshObject - NeighborsCellLinked" begin
            # Cell-linked neighbors only make sense for spatial dimensions >= 1
            # 1D
            box = [0.0 1.0]
            cellSize = [0.1]
            mesh = UnstructuredMesh(1; n  = Node((nnCL=Int,nnFull=Int)))
            @addRule model=mesh scope=integrator function get_neighbors1D(uNew, u, p, t)
                @kernel function get_neighbors1D_kernel!(uNew, u, p, t)
                    x1 = 0.0
                    x2 = 0.0
                    i = @index(Global)

                    if i < length(u.n.x)
                        u.n.nnFull[i] = 0
                        u.n.nnCL[i] = 0
                        x1 = u.n.x[i]
                        # Full neighbors
                        for j in iterateOver(u.n)
                            i == j ? continue : nothing
                            x2 = u.n.x[j]
                            dist = sqrt((x2 - x1)^2)
                            if dist <= 0.1
                                uNew.n.nnFull[i] += 1
                            end
                        end
                        # Cell-linked neighbors
                        for j in iterateOverNeighbors(u, :n, x1)
                            i == j ? continue : nothing
                            x2 = u.n.x[j]
                            dist = sqrt((x2 - x1)^2)
                            if dist <= 0.1
                                uNew.n.nnCL[i] += 1
                            end
                        end
                    end
                end

                dev = get_backend(uNew)
                nthreads = 256
                if dev == CPU()
                    nthreads = Threads.nthreads()
                end
                get_neighbors1D_kernel!(dev, nthreads)(uNew, u, p, t, ndrange=length(u.n.x))
                KernelAbstractions.synchronize(dev)
            end
            obj = UnstructuredMeshObject(mesh, n = 10000, neighbors=NeighborsCellLinked(box=box, cellSize=cellSize))

            obj.n.x .= rand(10000)

            problem = CBProblem(mesh, obj, (0.0, 1.0))
            integrator = init(problem, dt=0.1)
            step!(integrator)

            @test all(integrator.u.n.nnCL .== integrator.u.n.nnFull)

            if CUDA.has_cuda()
                obj_gpu = toGPU(obj)
                problem_gpu = CBProblem(mesh, obj_gpu, (0.0, 1.0))
                integrator_gpu = init(problem_gpu, dt=0.1)
                step!(integrator_gpu)
                # obj = toCPU(integrator_gpu.u)

                @test all(Array(integrator_gpu.u.n.nnCL) .== Array(integrator_gpu.u.n.nnFull))
            end

            # 2D
            box = [0.0 1.0; 0.0 1.0]
            cellSize = [0.1, 0.1]
            mesh = UnstructuredMesh(2; n  = Node((nnCL=Int,nnFull=Int)))
            @addRule model=mesh scope=integrator function get_neighbors2D(uNew, u, p, t)
                @kernel function get_neighbors2D_kernel!(uNew, u, p, t)
                    x1 = y1 = 0.0
                    x2 = y2 = 0.0
                    i = @index(Global)

                    if i < length(u.n.x)
                        u.n.nnFull[i] = 0
                        u.n.nnCL[i] = 0
                        x1 = u.n.x[i]
                        y1 = u.n.y[i]
                        # Full neighbors
                        for j in iterateOver(u.n)
                            i == j ? continue : nothing
                            x2 = u.n.x[j]
                            y2 = u.n.y[j]
                            dist = sqrt((x2 - x1)^2 + (y2 - y1)^2)
                            if dist <= 0.1
                                uNew.n.nnFull[i] += 1
                            end
                        end
                        # Cell-linked neighbors
                        for j in iterateOverNeighbors(u, :n, x1, y1)
                            i == j ? continue : nothing
                            x2 = u.n.x[j]
                            y2 = u.n.y[j]
                            dist = sqrt((x2 - x1)^2 + (y2 - y1)^2)
                            if dist <= 0.1
                                uNew.n.nnCL[i] += 1
                            end
                        end
                    end
                end

                dev = get_backend(uNew)
                nthreads = 256
                if dev == CPU()
                    nthreads = Threads.nthreads()
                end
                get_neighbors2D_kernel!(dev, nthreads)(uNew, u, p, t, ndrange=length(u.n.x))
                KernelAbstractions.synchronize(dev)
            end
            obj = UnstructuredMeshObject(mesh, n = 10000, neighbors=NeighborsCellLinked(box=box, cellSize=cellSize))

            obj.n.x .= rand(10000)
            obj.n.y .= rand(10000)

            problem = CBProblem(mesh, obj, (0.0, 1.0))
            integrator = init(problem, dt=0.1)
            step!(integrator)

            @test all(integrator.u.n.nnCL .== integrator.u.n.nnFull)

            if CUDA.has_cuda()
                obj_gpu = toGPU(obj)
                problem_gpu = CBProblem(mesh, obj_gpu, (0.0, 1.0))
                integrator_gpu = init(problem_gpu, dt=0.1)
                step!(integrator_gpu)
                # obj = toCPU(integrator_gpu.u)

                @test all(Array(integrator_gpu.u.n.nnCL) .== Array(integrator_gpu.u.n.nnFull))
            end

            # 3D
            box = [0.0 1.0; 0.0 1.0; 0.0 1.0]
            cellSize = [0.1, 0.1, 0.1]
            mesh = UnstructuredMesh(3; n  = Node((nnCL=Int,nnFull=Int)))
            @addRule model=mesh scope=integrator function get_neighbors3D(uNew, u, p, t)
                @kernel function get_neighbors3D_kernel!(uNew, u, p, t)
                    x1 = y1 = z1 = 0.0
                    x2 = y2 = z2 = 0.0
                    i = @index(Global)

                    if i < length(u.n.x)
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
                            if dist <= 0.1
                                uNew.n.nnCL[i] += 1
                            end
                        end
                    end
                end

                dev = get_backend(uNew)
                nthreads = 256
                if dev == CPU()
                    nthreads = Threads.nthreads()
                end
                get_neighbors3D_kernel!(dev, nthreads)(uNew, u, p, t, ndrange=length(u.n.x))
                KernelAbstractions.synchronize(dev)
            end
            obj = UnstructuredMeshObject(mesh, n = 10000, neighbors=NeighborsCellLinked(box=box, cellSize=cellSize))

            obj.n.x .= rand(10000)
            obj.n.y .= rand(10000)
            obj.n.z .= rand(10000)

            problem = CBProblem(mesh, obj, (0.0, 1.0))
            integrator = init(problem, dt=0.1)
            step!(integrator)

            @test all(integrator.u.n.nnCL .== integrator.u.n.nnFull)

            if CUDA.has_cuda()
                obj_gpu = toGPU(obj)
                problem_gpu = CBProblem(mesh, obj_gpu, (0.0, 1.0))
                integrator_gpu = init(problem_gpu, dt=0.1)
                step!(integrator_gpu)

                @test all(Array(integrator_gpu.u.n.nnCL) .== Array(integrator_gpu.u.n.nnFull))
            end

        end

        # @testset "UnstructuredMeshObject - CellLinked Algorithm Correctness" begin
        #     # Test cell assignment
        #     @testset "Cell Assignment" begin
        #         # 2D case
        #         props2d = (a=Float64, b=Float64)
        #         mesh = UnstructuredMesh(2; n=Node(props2d))
        #         box = [0.0 10.0; 0.0 10.0]
        #         neighbors = NeighborsCellLinked(;box=box, cellSize=(2.0, 2.0))
        #         obj = UnstructuredMeshObject(mesh, n=(4, 4), neighbors=neighbors)
                
        #         # Set up test positions
        #         obj.n.x[1] = 0.5; obj.n.y[1] = 0.5   # Should be in cell 1
        #         obj.n.x[2] = 2.5; obj.n.y[2] = 0.5   # Should be in cell 2
        #         obj.n.x[3] = 0.5; obj.n.y[3] = 2.5   # Should be in cell 6
        #         obj.n.x[4] = 2.5; obj.n.y[4] = 2.5   # Should be in cell 7
                
        #         CellBasedModels.update!(obj)
                
        #         # Test cell assignments
        #         @test CellBasedModels.assignCell(obj._neighbors, 0.5, 0.5) == 1
        #         @test CellBasedModels.assignCell(obj._neighbors, 2.5, 0.5) == 2  
        #         @test CellBasedModels.assignCell(obj._neighbors, 0.5, 2.5) == 8  # 1 * 7 + 0 + 1 
        #         @test CellBasedModels.assignCell(obj._neighbors, 2.5, 2.5) == 9  # 1 * 7 + 1 + 1
        #     end

        #     # Test basic neighbor finding
        #     @testset "Basic Neighbor Finding" begin
        #         mesh = UnstructuredMesh(2; n=Node())
        #         box = [0.0 10.0; 0.0 10.0]
        #         neighbors = NeighborsCellLinked(;box=box, cellSize=(3.0, 3.0))
        #         obj = UnstructuredMeshObject(mesh, n=(3, 3), neighbors=neighbors)
                
        #         # Place particles in same cell
        #         obj.n.x[1] = 1.0; obj.n.y[1] = 1.0   # Cell 1
        #         obj.n.x[2] = 1.5; obj.n.y[2] = 1.5   # Cell 1  
        #         obj.n.x[3] = 8.0; obj.n.y[3] = 8.0   # Different cell
                
        #         CellBasedModels.update!(obj)
                
        #         # Particle 1 should find particle 2 (same cell) but not particle 3
        #         neighbors_1 = collect(iterateOverNeighbors(obj, :n, obj.n.x[1], obj.n.y[1]))
        #         @test 1 in neighbors_1
        #         @test 2 in neighbors_1
        #         @test !(3 in neighbors_1)
        #         @test length(neighbors_1) == 2
        #     end
        # end

        # @testset "UnstructuredMeshObject - Periodic Boundaries" begin
        #     # Test periodic boundary conditions
        #     @testset "Periodic Boundary Cell Assignment" begin
        #         mesh = UnstructuredMesh(2; n=Node())
        #         box = [0.0 10.0; 0.0 10.0]
        #         neighbors = NeighborsCellLinked(;box=box, cellSize=(2.0, 2.0), periodic=(true, true))
        #         obj = UnstructuredMeshObject(mesh, n=(4, 4), neighbors=neighbors)
                
        #         # Test particles at boundaries
        #         obj.n.x[1] = 0.5; obj.n.y[1] = 0.5     # Bottom-left  
        #         obj.n.x[2] = 9.5; obj.n.y[2] = 0.5     # Bottom-right
        #         obj.n.x[3] = 0.5; obj.n.y[3] = 9.5     # Top-left
        #         obj.n.x[4] = 9.5; obj.n.y[4] = 9.5     # Top-right
                
        #         CellBasedModels.update!(obj)
                
        #         # Verify grid size is correct for periodic (no padding)
        #         @test obj._neighbors.grid == (5, 5)  # 10/2 = 5, no padding for periodic
                
        #         # Test cell assignments
        #         @test CellBasedModels.assignCell(obj._neighbors, 0.5, 0.5) == 1   # Bottom-left corner
        #         @test CellBasedModels.assignCell(obj._neighbors, 9.5, 0.5) == 5   # Bottom-right corner  
        #         @test CellBasedModels.assignCell(obj._neighbors, 0.5, 9.5) == 21  # Top-left corner
        #         @test CellBasedModels.assignCell(obj._neighbors, 9.5, 9.5) == 25  # Top-right corner
        #     end

        #     @testset "Periodic Boundary Neighbor Finding" begin
        #         mesh = UnstructuredMesh(2; n=Node())
        #         box = [0.0 10.0; 0.0 10.0]
        #         neighbors = NeighborsCellLinked(;box=box, cellSize=(2.0, 2.0), periodic=(true, true))
        #         obj = UnstructuredMeshObject(mesh, n=(4, 4), neighbors=neighbors)
                
        #         # Place particles at corners to test wraparound
        #         obj.n.x[1] = 0.5; obj.n.y[1] = 0.5     # Bottom-left
        #         obj.n.x[2] = 9.5; obj.n.y[2] = 0.5     # Bottom-right  
        #         obj.n.x[3] = 0.5; obj.n.y[3] = 9.5     # Top-left
        #         obj.n.x[4] = 9.5; obj.n.y[4] = 9.5     # Top-right
                
        #         CellBasedModels.update!(obj)
                
        #         # Particle 1 (bottom-left) should find all others due to periodic wrapping
        #         neighbors_1 = collect(iterateOverNeighbors(obj, :n, obj.n.x[1], obj.n.y[1]))
        #         @test length(neighbors_1) == 4  # Should find all 4 particles
        #         @test 1 in neighbors_1  # Itself
        #         @test 2 in neighbors_1  # Right neighbor (x-wrap)
        #         @test 3 in neighbors_1  # Top neighbor (y-wrap)  
        #         @test 4 in neighbors_1  # Diagonal neighbor (x,y-wrap)
        #     end

        #     @testset "Mixed Periodic Boundaries" begin
        #         # Test periodic in x but not y
        #         mesh = UnstructuredMesh(2; n=Node())
        #         box = [0.0 10.0; 0.0 10.0]
        #         neighbors = NeighborsCellLinked(;box=box, cellSize=(2.0, 2.0), periodic=(true, false))
        #         obj = UnstructuredMeshObject(mesh, n=(4, 4), neighbors=neighbors)
                
        #         # Place particles at corners
        #         obj.n.x[1] = 0.5; obj.n.y[1] = 0.5     # Bottom-left
        #         obj.n.x[2] = 9.5; obj.n.y[2] = 0.5     # Bottom-right (x-wraps to left)
        #         obj.n.x[3] = 0.5; obj.n.y[3] = 9.5     # Top-left (y-doesn't wrap)
        #         obj.n.x[4] = 9.5; obj.n.y[4] = 9.5     # Top-right
                
        #         CellBasedModels.update!(obj)
                
        #         # Grid should have padding in y but not x
        #         @test obj._neighbors.grid[1] == 5   # x: no padding (10/2 = 5)  
        #         @test obj._neighbors.grid[2] == 7   # y: padding (10/2 + 2 = 7)
                
        #         # Particle 1 should find particle 2 (x-wrap) but not particles 3,4 (too far in y)
        #         neighbors_1 = collect(iterateOverNeighbors(obj, :n, obj.n.x[1], obj.n.y[1]))
        #         @test 1 in neighbors_1  # Itself
        #         @test 2 in neighbors_1  # x-wrapped neighbor
        #         # Should not find 3 or 4 due to large y-distance and no y-wrapping
        #         @test length(neighbors_1) >= 2  # At least itself and x-wrapped neighbor
        #     end

        #     @testset "Non-Periodic Boundaries" begin
        #         # Test no periodic boundaries (original behavior)
        #         mesh = UnstructuredMesh(2; n=Node())
        #         box = [0.0 10.0; 0.0 10.0] 
        #         neighbors = NeighborsCellLinked(;box=box, cellSize=(2.0, 2.0), periodic=(false, false))
        #         obj = UnstructuredMeshObject(mesh, n=(4, 4), neighbors=neighbors)
                
        #         # Place particles at corners
        #         obj.n.x[1] = 0.5; obj.n.y[1] = 0.5     # Bottom-left
        #         obj.n.x[2] = 9.5; obj.n.y[2] = 0.5     # Bottom-right
        #         obj.n.x[3] = 0.5; obj.n.y[3] = 9.5     # Top-left  
        #         obj.n.x[4] = 9.5; obj.n.y[4] = 9.5     # Top-right
                
        #         CellBasedModels.update!(obj)
                
        #         # Grid should have padding in both dimensions
        #         @test obj._neighbors.grid == (7, 7)  # 10/2 + 2 = 7 for both dimensions
                
        #         # Particle 1 should only find itself (no wrapping, corners are far apart)
        #         neighbors_1 = collect(iterateOverNeighbors(obj, :n, obj.n.x[1], obj.n.y[1]))
        #         @test 1 in neighbors_1  # Always finds itself
        #         @test length(neighbors_1) == 1  # Should only find itself
        #     end
        # end

        # @testset "UnstructuredMeshObject - CellLinked Edge Cases" begin
        #     @testset "Empty Cells" begin
        #         # Test behavior with many empty cells
        #         mesh = UnstructuredMesh(2; n=Node())
        #         box = [0.0 10.0; 0.0 10.0]
        #         neighbors = NeighborsCellLinked(;box=box, cellSize=(1.0, 1.0))  # Many small cells
        #         obj = UnstructuredMeshObject(mesh, n=(2, 2), neighbors=neighbors)
                
        #         # Place only 2 particles far apart
        #         obj.n.x[1] = 0.5; obj.n.y[1] = 0.5  # Cell 1
        #         obj.n.x[2] = 9.5; obj.n.y[2] = 9.5  # Cell 100 (far away)
                
        #         CellBasedModels.update!(obj)
                
        #         # Each particle should only find itself
        #         neighbors_1 = collect(iterateOverNeighbors(obj, :n, obj.n.x[1], obj.n.y[1]))
        #         neighbors_2 = collect(iterateOverNeighbors(obj, :n, obj.n.x[2], obj.n.y[2]))
                
        #         @test length(neighbors_1) == 1
        #         @test length(neighbors_2) == 1
        #         @test neighbors_1[1] == 1
        #         @test neighbors_2[1] == 2
        #     end

        #     @testset "Single Cell" begin
        #         # Test when all particles are in the same cell
        #         mesh = UnstructuredMesh(2; n=Node())
        #         box = [0.0 10.0; 0.0 10.0]
        #         neighbors = NeighborsCellLinked(;box=box, cellSize=(20.0, 20.0))  # Very large cells
        #         obj = UnstructuredMeshObject(mesh, n=(3, 3), neighbors=neighbors)
                
        #         # All particles in same cell
        #         obj.n.x[1] = 2.0; obj.n.y[1] = 2.0
        #         obj.n.x[2] = 4.0; obj.n.y[2] = 4.0  
        #         obj.n.x[3] = 6.0; obj.n.y[3] = 6.0
                
        #         CellBasedModels.update!(obj)
                
        #         # Each particle should find all particles
        #         for i in 1:3
        #             neighbors_i = collect(iterateOverNeighbors(obj, :n, obj.n.x[i], obj.n.y[i]))
        #             @test length(neighbors_i) == 3
        #             @test 1 in neighbors_i
        #             @test 2 in neighbors_i  
        #             @test 3 in neighbors_i
        #         end
        #     end
        # end
    end
end