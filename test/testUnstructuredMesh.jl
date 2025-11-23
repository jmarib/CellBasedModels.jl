using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector

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

    #######################################################################
    # TEST 1: UnstructuredMesh
    #######################################################################
    @testset "UnstructuredMesh - full coverage" begin
        for dims in 0:3
            for scope in all_scopes
                mesh = nothing
                if scope === Node
                    mesh = UnstructuredMesh(dims; n  = Node(props))
                elseif scope === Edge
                    mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
                elseif scope === Face
                    mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
                elseif scope === Volume
                    mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
                elseif scope === Agent
                    mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
                end
                @test mesh isa UnstructuredMesh
                @test CellBasedModels.spatialDims(mesh) == dims
                @test CellBasedModels.specialization(mesh) === Nothing
            end
        end

        # Invalid dimension
        @test_throws ErrorException UnstructuredMesh(-1)
        @test_throws ErrorException UnstructuredMesh(4)

        # Duplicate protected parameter name (e.g., x)
        bad_props = (x = Parameter(Float64, description="duplicate", dimensions=:L, defaultValue=0.0),)
        @test_throws ErrorException UnstructuredMesh(1; n=Node(bad_props))

        # Show output test
        io = IOBuffer()
        show(io, UnstructuredMesh(2; n=Node(props)))
        output = String(take!(io))
        # show(UnstructuredMesh(2; n=Node(props)))
        @test occursin("UnstructuredMesh with dimensions 2", output)

        # Show output test
        io = IOBuffer()
        show(io, typeof(UnstructuredMesh(2; n=Node(props))))
        output = String(take!(io))
        # show(typeof(UnstructuredMesh(2; n=Node(props))))
        @test occursin("UnstructuredMesh{dims=2", output)
    end

    #######################################################################
    # TEST 2: UnstructuredMeshField
    #######################################################################
    @testset "UnstructuredMeshField - construction and logic" begin

        # Valid case
        meshProperty = Node(props)
        field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
        @test field isa UnstructuredMeshField
        @test field._N[] == 3
        @test field._NCache[] == 5
        @test length(field._FlagsRemoved) == 5
        @test field._FlagOverflow[] == false
        @test field._idMax[] == 3
        @test field._NP == length(props)
        @test field._pReference isa SizedVector

        # Get properties
        @test field.a == zeros(Float64, 3)
        @test field._p.a == zeros(Float64, 5)

        # With id=false
        field2 = UnstructuredMeshField(meshProperty; N=2, NCache=5, id=false)
        @test field2._id === nothing
        @test field2._idMax === nothing

        # Show output formatting
        io = IOBuffer()
        show(io, field)
        txt = String(take!(io))
        @test occursin("_id", txt)
        @test occursin("_AddedAgents", txt)

        # Copy
        field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
        field._pReference .= [true, false]
        fieldCopy = copy(field)
        field.a .= 7.0
        field.b .= 2
        @test fieldCopy.a == fill(7.0, 3)
        @test fieldCopy.b == zeros(Int, 3)

        # zero
        field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
        field._pReference .= [true, false]
        field.a .= 7.0
        field.b .= 2        
        fieldZero = zero(field)
        @test fieldZero.a == fill(7.0, 3)
        @test fieldZero.b == zeros(Int, 3)

        # toGPU/toCPU
        if CUDA.has_cuda()
            field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
            field._pReference .= [true, false]
            field.a .= 7.0
            field.b .= 2

            field_gpu = toGPU(field)

            @test typeof(field_gpu._id) <: CUDA.CuArray
            @test typeof(field_gpu.a) <: CUDA.CuArray

            field_cpu = toCPU(field_gpu)

            @test typeof(field_cpu) == typeof(field)
        end

        # Copyto!
        field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
        field._pReference .= [false, true]
        field.a .= 7.0
        field.b .= 2
        field2 = UnstructuredMeshField(meshProperty; N=3, NCache=5)
        copyto!(field2, field)
        @test field2.a == fill(7.0, 3)
        @test field2.b == fill(0, 3)
        if CUDA.has_cuda()
            field_gpu = toGPU(field)
            field2_gpu = toGPU(field2)
            copyto!(field2_gpu, field_gpu)
            @test Array(field2_gpu.a) == fill(7.0, 3)
            @test Array(field2_gpu.b) == fill(0, 3)
        end

        # Broadcast
        field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
        field._pReference .= [false, true]
        field.a .= 7.0
        field.b .= 2
        field2 = UnstructuredMeshField(meshProperty; N=3, NCache=5)
        field2._pReference .= [false, true]
        @. field2 = field * 0.1 + 3.0
        @test field2.a == fill(3.7, 3)
        @test field2.b == fill(0, 3)
        if CUDA.has_cuda()
            field_gpu = toGPU(field)
            field2_gpu = toGPU(field2)
            @. field2_gpu = field_gpu * 0.1 + 3.0
            @test Array(field2_gpu.a) == fill(3.7, 3)
            @test Array(field2_gpu.b) == fill(0, 3)
        end

        # Broadcast @..
        field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
        field._pReference .= [false, true]
        field.a .= 7.0
        field.b .= 2
        field2 = UnstructuredMeshField(meshProperty; N=3, NCache=5)
        field2._pReference .= [false, true]
        DifferentialEquations.DiffEqBase.@.. field2 = field * 0.1 + 3.0
        @test field2.a == fill(3.7, 3)
        @test field2.b == fill(0, 3)
        if CUDA.has_cuda()
            field_gpu = toGPU(field)
            field2_gpu = toGPU(field2)
            DifferentialEquations.DiffEqBase.@.. field2_gpu = field_gpu * 0.1 + 3.0
            @test Array(field2_gpu.a) == fill(3.7, 3)
            @test Array(field2_gpu.b) == fill(0, 3)
        end

        # Kernel works
        if CUDA.has_cuda()

            function test_kernel_field!(x)
                x.a[1]
                nothing
            end

            field = UnstructuredMeshField(meshProperty; N=3, NCache=5)
            field._pReference .= [true, false]
            field_gpu = toGPU(field)

            @test_nowarn CUDA.@cuda test_kernel_field!(field_gpu)

        end

    end

    #######################################################################
    # TEST 3: UnstructuredMeshObject
    #######################################################################
    @testset "UnstructuredMeshObject - construction" begin
        for dims in 0:3
            for (scope, scope_index) in zip(all_scopes, all_scopes_index)
                mesh = nothing
                if scope === Node
                    mesh = UnstructuredMesh(dims; n  = Node(props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4))
                    @test obj isa UnstructuredMeshObject
                    @test obj.n isa UnstructuredMeshField
                elseif scope === Edge
                    mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
                    @test obj isa UnstructuredMeshObject
                    @test obj.n isa UnstructuredMeshField
                    @test obj.e isa UnstructuredMeshField
                elseif scope === Face
                    mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
                    @test obj isa UnstructuredMeshObject
                    @test obj.n isa UnstructuredMeshField
                    @test obj.f isa UnstructuredMeshField
                elseif scope === Volume
                    mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
                    @test obj isa UnstructuredMeshObject
                    @test obj.n isa UnstructuredMeshField
                    @test obj.v isa UnstructuredMeshField
                elseif scope === Agent
                    mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
                    @test obj isa UnstructuredMeshObject
                    @test obj.n isa UnstructuredMeshField
                    @test obj.a isa UnstructuredMeshField
                end

                # Invalid cache values
                @test_throws ErrorException UnstructuredMeshObject(mesh, agentN=4, agentNCache=2)
                @test_throws ErrorException UnstructuredMeshObject(mesh, agentN=-1, agentNCache=0)
            end
        end
    end

    @testset "UnstructuredMeshObject - copy" begin
        for dims in 0:3
            for (scope, scope_index) in zip(all_scopes, all_scopes_index)
                mesh = nothing
                obj = nothing
                if scope === Node
                    mesh = UnstructuredMesh(dims; n  = Node(props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4))
                elseif scope === Edge
                    mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
                elseif scope === Face
                    mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
                elseif scope === Volume
                    mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
                elseif scope === Agent
                    mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
                end
                if scope == Node
                    obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., true, false]
                else
                    obj[scope_index]._pReference .= [true, false]
                end
                objCopy = copy(obj)
                obj[scope_index].a .= 7.0
                obj[scope_index].b .= 2
                @test objCopy[scope_index].a == fill(7.0, 2)
                @test objCopy[scope_index]._p.a == [7.0, 7.0, 0.0, 0.0]
                @test objCopy[scope_index].b == zeros(Int, 2)
                @test objCopy[scope_index]._p.b == [0, 0, 0, 0]
            end
        end
    end

    @testset "UnstructuredMeshObject - zero" begin
        for dims in 0:3
            for (scope, scope_index) in zip(all_scopes, all_scopes_index)
                mesh = nothing
                obj = nothing
                if scope === Node
                    mesh = UnstructuredMesh(dims; n  = Node(props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4))
                elseif scope === Edge
                    mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
                elseif scope === Face
                    mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
                elseif scope === Volume
                    mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
                elseif scope === Agent
                    mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
                end
                if scope == Node
                    obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., true, false]
                else
                    obj[scope_index]._pReference .= [true, false]
                end
                obj[scope_index].a .= 7.0
                obj[scope_index].b .= 2
                objZero = zero(obj)
                @test objZero[scope_index].a == fill(7.0, 2)
                @test objZero[scope_index]._p.a == [7.0, 7.0, 0.0, 0.0]
                @test objZero[scope_index].b == zeros(Int, 2)
                @test objZero[scope_index]._p.b == [0, 0, 0, 0]
            end
        end
    end

    @testset "UnstructuredMeshObject - toGPU/toCPU" begin
        for dims in 0:3
            for (scope, scope_index) in zip(all_scopes, all_scopes_index)
                # toGPU/toCPU
                if CUDA.has_cuda()
                    mesh = nothing
                    obj = nothing
                    if scope === Node
                        mesh = UnstructuredMesh(dims; n  = Node(props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4))
                    elseif scope === Edge
                        mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
                    elseif scope === Face
                        mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
                    elseif scope === Volume
                        mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
                    elseif scope === Agent
                        mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
                    end
                    if scope == Node
                        obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., true, false]
                    else
                        obj[scope_index]._pReference .= [true, false]
                    end
                    obj[scope_index].a .= 7.0
                    obj[scope_index].b .= 2
                    obj_gpu = toGPU(obj)
                    @test typeof(obj_gpu[scope_index]._id) <: CUDA.CuArray
                    @test typeof(obj_gpu[scope_index].a) <: CUDA.CuArray
                    obj_cpu = toCPU(obj_gpu)
                    @test typeof(obj_cpu) == typeof(obj)
                end
            end
        end
    end

    @testset "UnstructuredMeshObject - copyto!" begin
        for dims in 0:3
            for (scope, scope_index) in zip(all_scopes, all_scopes_index)
                    mesh = nothing
                    obj = nothing
                    if scope === Node
                        mesh = UnstructuredMesh(dims; n  = Node(props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4))
                        obj2 = UnstructuredMeshObject(mesh, n = (2,4))
                    elseif scope === Edge
                        mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
                        obj2 = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
                    elseif scope === Face
                        mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
                        obj2 = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
                    elseif scope === Volume
                        mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
                        obj2 = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
                    elseif scope === Agent
                        mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
                        obj2 = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
                    end
                if scope == Node
                    obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., true, false]
                else
                    obj[scope_index]._pReference .= [true, false]
                end
                obj[scope_index].a .= 7.0
                obj[scope_index].b .= 2
                if scope == Node
                    obj2[scope_index]._pReference .= [[true, true, true][1:1:dims]..., true, false]
                else
                    obj2[scope_index]._pReference .= [true, false]
                end
                copyto!(obj2, obj)
                @test obj2[scope_index].a == fill(0.0, 2)
                @test obj2[scope_index]._p.a == [0.0, 0.0, 0.0, 0.0]
                @test obj2[scope_index].b == fill(2, 2)
                @test obj2[scope_index]._p.b == [2, 2, 0, 0]
                if CUDA.has_cuda()
                    obj_gpu = toGPU(obj)
                    obj2_gpu = toGPU(obj2)
                    copyto!(obj2_gpu, obj_gpu)
                    @test Array(obj2_gpu[scope_index].a) == fill(0.0, 2)
                    @test Array(obj2_gpu[scope_index]._p.a) == [0.0, 0.0, 0.0, 0.0]
                    @test Array(obj2_gpu[scope_index].b) == fill(2, 2)
                    @test Array(obj2_gpu[scope_index]._p.b) == [2, 2, 0, 0]
                end
            end
        end
    end

    @testset "UnstructuredMeshObject - broadcasting" begin
        for dims in 0:3
            for (scope, scope_index) in zip(all_scopes, all_scopes_index)
                mesh = nothing
                obj = nothing
                if scope === Node
                    mesh = UnstructuredMesh(dims; n  = Node(props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4))
                    obj2 = UnstructuredMeshObject(mesh, n = (2,4))
                elseif scope === Edge
                    mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
                    obj2 = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
                elseif scope === Face
                    mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
                    obj2 = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
                elseif scope === Volume
                    mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
                    obj2 = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
                elseif scope === Agent
                    mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
                    obj2 = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
                end
                if scope == Node
                    obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., false, true]
                else
                    obj[scope_index]._pReference .= [false, true]
                end
                obj[scope_index].a .= 7.0
                obj[scope_index].b .= 2
                if scope == Node
                    obj2[scope_index]._pReference .= [[true, true, true][1:1:dims]..., false, true]
                else
                    obj2[scope_index]._pReference .= [false, true]
                end
                @. obj2 = obj * 0.1 + 3.0
                @test obj2[scope_index].a == fill(3.7, 2)
                @test obj2[scope_index]._p.a == [3.7, 3.7, 0.0, 0.0]
                @test obj2[scope_index].b == fill(0, 2)
                @test obj2[scope_index]._p.b == [0, 0, 0, 0]
                if CUDA.has_cuda()
                    obj_gpu = toGPU(obj)
                    obj2_gpu = toGPU(obj2)
                    obj2_gpu[scope_index].a .= 0.0
                    @. obj2_gpu = obj_gpu * 0.1 + 3.0
                    @test Array(obj2_gpu[scope_index].a) == fill(3.7, 2)
                    @test Array(obj2_gpu[scope_index]._p.a) == [3.7, 3.7, 0.0, 0.0]
                    @test Array(obj2_gpu[scope_index].b) == fill(0, 2)
                    @test Array(obj2_gpu[scope_index]._p.b) == [0, 0, 0, 0]
                end

                # Broadcast @..
                mesh = nothing
                obj = nothing
                if scope === Node
                    mesh = UnstructuredMesh(dims; n  = Node(props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4))
                    obj2 = UnstructuredMeshObject(mesh, n = (2,4))
                elseif scope === Edge
                    mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
                    obj2 = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
                elseif scope === Face
                    mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
                    obj2 = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
                elseif scope === Volume
                    mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
                    obj2 = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
                elseif scope === Agent
                    mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
                    obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
                    obj2 = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
                end
                if scope == Node
                    obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., false, true]
                else
                    obj[scope_index]._pReference .= [false, true]
                end
                obj[scope_index].a .= 7.0
                obj[scope_index].b .= 2
                if scope == Node
                    obj2[scope_index]._pReference .= [[true, true, true][1:1:dims]..., false, true]
                else
                    obj2[scope_index]._pReference .= [false, true]
                end
                DifferentialEquations.DiffEqBase.@.. obj2 = obj * 0.1 + 3.0
                @test obj2[scope_index].a == fill(3.7, 2)
                @test obj2[scope_index]._p.a == [3.7, 3.7, 0.0, 0.0]
                @test obj2[scope_index].b == fill(0, 2)
                @test obj2[scope_index]._p.b == [0, 0, 0, 0]
                if CUDA.has_cuda()
                    obj_gpu = toGPU(obj)
                    obj2_gpu = toGPU(obj2)
                    obj2_gpu[scope_index].a .= 0.0
                    DifferentialEquations.DiffEqBase.@.. obj2_gpu = obj_gpu * 0.1 + 3.0
                    @test Array(obj2_gpu[scope_index].a) == fill(3.7, 2)
                    @test Array(obj2_gpu[scope_index]._p.a) == [3.7, 3.7, 0.0, 0.0]
                    @test Array(obj2_gpu[scope_index].b) == fill(0, 2)
                    @test Array(obj2_gpu[scope_index]._p.b) == [0, 0, 0, 0]
                end
            end
        end
    end

    @testset "UnstructuredMeshObject - GPU kernels" begin
        for dims in 0:3
            for (scope, scope_index) in zip(all_scopes, all_scopes_index)            
                # Kernel works
                if CUDA.has_cuda()
                    function test_kernel_object!(x)
                        x.n
                        nothing
                    end

                    mesh = nothing
                    obj = nothing
                    if scope === Node
                        mesh = UnstructuredMesh(dims; n  = Node(props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4))
                    elseif scope === Edge
                        mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
                    elseif scope === Face
                        mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
                    elseif scope === Volume
                        mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
                    elseif scope === Agent
                        mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
                    end
                    obj_gpu = toGPU(obj)

                    @test_nowarn CUDA.@cuda test_kernel_object!(obj_gpu)
                end
            end
        end
    end

    @testset "UnstructuredMeshObject - DifferentialEquations compatibility" begin
        for dims in 0:3
            for (scope, scope_index) in zip(all_scopes, all_scopes_index)
                for integratorAlg in (Euler(), RK4())#, Tsit5())

                    # DifferentialEquations.jl compatibility
                    function fODE!(du, u, p, t)
                        @. du[scope_index].a = 1
                        return
                    end
                    mesh = nothing
                    obj = nothing
                    if scope === Node
                        mesh = UnstructuredMesh(dims; n  = Node(props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4))
                    elseif scope === Edge
                        mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
                    elseif scope === Face
                        mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
                    elseif scope === Volume
                        mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
                    elseif scope === Agent
                        mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
                        obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
                    end
                    if scope == Node
                        obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., false, true]
                    else
                        obj[scope_index]._pReference .= [false, true]
                    end
                    prob = ODEProblem(fODE!, obj, (0.0, 1.0))
                    integrator = init(prob, integratorAlg, dt=0.1, save_everystep=false)
                    for i in 1:10
                        step!(integrator, 0.1, true)
                    end
                    @test all(integrator.u[scope_index].a .≈ 1.0)
                    if CUDA.has_cuda()
                        mesh = nothing
                        obj = nothing
                        if scope === Node
                            mesh = UnstructuredMesh(dims; n  = Node(props))
                            obj = UnstructuredMeshObject(mesh, n = (2,4))
                        elseif scope === Edge
                            mesh = UnstructuredMesh(dims; n = Node(), e  = Edge((:n,:n), props))
                            obj = UnstructuredMeshObject(mesh, n = (2,4), e = (2,4))
                        elseif scope === Face
                            mesh = UnstructuredMesh(dims; n = Node(), f  = Face(:n, props))
                            obj = UnstructuredMeshObject(mesh, n = (2,4), f = (2,4))
                        elseif scope === Volume
                            mesh = UnstructuredMesh(dims; n = Node(), v  = Volume(:n, props))
                            obj = UnstructuredMeshObject(mesh, n = (2,4), v = (2,4))
                        elseif scope === Agent
                            mesh = UnstructuredMesh(dims; n = Node(), a  = Agent(props))
                            obj = UnstructuredMeshObject(mesh, n = (2,4), a = (2,4))
                        end
                        if scope == Node
                            obj[scope_index]._pReference .= [[true, true, true][1:1:dims]..., false, true]
                        else
                            obj[scope_index]._pReference .= [false, true]
                        end
                        obj_gpu = toGPU(obj)
                        prob = ODEProblem(fODE!, obj_gpu, (0.0, 1.0))
                        integrator = init(prob, integratorAlg, dt=0.1, save_everystep=false)
                        for i in 1:10
                            step!(integrator, 0.1, true)
                        end
                        @test all(Array(integrator.u[scope_index].a) .≈ 1.0)
                    end
                end
            end
        end
    end
end