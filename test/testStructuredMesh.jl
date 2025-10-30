using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector

@testset verbose = verbose "ABM - StructuredMesh" begin

    #######################################################################
    # HELPERS
    #######################################################################
    props = (
            a = Parameter(Float64, description="param a", dimensions=:M, defaultValue=1.0),
            b = Parameter(Int, description="param b", dimensions=:count, defaultValue=0),
        )

    #######################################################################
    # TEST 1: StructuredMesh
    #######################################################################
    @testset "StructuredMesh - construction" begin
        for dims in 1:3
            mesh = StructuredMesh(
                dims;
                properties  = props,
            )

            @test mesh isa StructuredMesh
            @test CellBasedModels.spatialDims(mesh) == dims
            @test CellBasedModels.specialization(mesh) === Nothing
        end

        # Invalid dimension
        @test_throws ErrorException StructuredMesh(-1, properties=props)
        @test_throws ErrorException StructuredMesh(4, properties=props)

        # Show output test
        io = IOBuffer()
        show(io, StructuredMesh(2; properties=props))
        output = String(take!(io))
        @test occursin("StructuredMesh with dimensions 2", output)
    end

    #######################################################################
    # TEST 3: StructuredMeshObject
    #######################################################################
    @testset "StructuredMeshObject - construction" begin
        for dims in 1:3
            mesh = StructuredMesh(
                dims;
                properties = props,
            )

            obj = StructuredMeshObject(mesh, 
                simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[0.5, 0.5, 0.5][1:dims]
            )
            @test obj isa StructuredMeshObject
            @test CellBasedModels.spatialDims(obj) == dims
            @test CellBasedModels.specialization(obj) === Nothing

            # Invalid cache values
            @test_throws ErrorException StructuredMeshObject(mesh, simulationBox=[0,1], gridSpacing=[0.5, 0.5, 0.5][1:dims])
            @test_throws ErrorException StructuredMeshObject(mesh, simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[1.0, 1.0, 1.0, 1.0])

        end
    end
    @testset "StructuredMeshObject - copy" begin
        for dims in 1:3
            mesh = StructuredMesh(
                dims;
                properties = props,
            )
            # Copy
            obj = StructuredMeshObject(mesh, 
                simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[0.5, 0.5, 0.5][1:dims]
            )
            obj._pReference .= [true, false]
            objCopy = copy(obj)
            obj.p.a .= 7.0
            obj.p.b .= 2
            @test all(objCopy.p.a .== 7.0)
            @test all(objCopy.p.b .== 0)

        end
    end
    @testset "StructuredMeshObject - zero" begin
        for dims in 1:3
            mesh = StructuredMesh(
                dims;
                properties = props,
            )
            # zero
            obj = StructuredMeshObject(mesh, 
                simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[0.5, 0.5, 0.5][1:dims]
            )
            obj._pReference .= [true, false]
            obj.p.a .= 7.0
            obj.p.b .= 2
            objZero = zero(obj)
            @test all(objZero.p.a .== 7.0)
            @test all(objZero.p.b .== 0)

        end
    end
    @testset "StructuredMeshObject - toGPU/toCPU" begin
        for dims in 1:3
            mesh = StructuredMesh(
                dims;
                properties = props,
            )
            # toGPU/toCPU
            if CUDA.has_cuda()
                obj = StructuredMeshObject(mesh, 
                    simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[0.5, 0.5, 0.5][1:dims]
                )
                obj._pReference .= [true, false]
                obj.p.a .= 7.0
                obj.p.b .= 2
                obj_gpu = toGPU(obj)
                @test typeof(obj_gpu.p.a) <: CUDA.CuArray
                obj_cpu = toCPU(obj_gpu)
                @test typeof(obj_cpu) == typeof(obj)
            end
        end
    end

    @testset "StructuredMeshObject - copyto!" begin
        for dims in 1:3
            mesh = StructuredMesh(
                dims;
                properties = props,
            )
            # copyto!
            obj = StructuredMeshObject(mesh, 
                simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[0.5, 0.5, 0.5][1:dims]
            )
            obj._pReference .= [true, false]
            obj.p.a .= 7.0
            obj.p.b .= 2
            obj2 = StructuredMeshObject(mesh, 
                simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[0.5, 0.5, 0.5][1:dims]
            )
            obj2._pReference .= [true, false]
            copyto!(obj2, obj)
            @test all(obj2.p.a .== 0.0)
            @test all(obj2.p.b .== 2)
            if CUDA.has_cuda()
                obj_gpu = toGPU(obj)
                obj2_gpu = toGPU(obj2)
                copyto!(obj2_gpu, obj_gpu)
                @test all(Array(obj2_gpu.p.a) .== 0)
                @test all(Array(obj2_gpu.p.b) .== 2)
            end

        end
    end

    @testset "StructuredMeshObject - broadcasting" begin
        for dims in 1:3
            mesh = StructuredMesh(
                dims;
                properties = props,
            )
            # Broadcast
            obj = StructuredMeshObject(mesh, 
                simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[0.5, 0.5, 0.5][1:dims]
            )
            obj._pReference .= [false, true]
            obj.p.a .= 7.0
            obj.p.b .= 2
            obj2 = StructuredMeshObject(mesh, 
                simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[0.5, 0.5, 0.5][1:dims]
            )
            obj2._pReference .= [false, true]
            @. obj2 = obj * 0.1 + 3.0
            @test all(obj2.p.a .== 3.7)
            @test all(obj2.p.b .== 0)
            if CUDA.has_cuda()
                obj_gpu = toGPU(obj)
                obj2_gpu = toGPU(obj2)
                obj2_gpu.p.a .= 0.0
                @. obj2_gpu = obj_gpu * 0.1 + 3.0
                @test all(Array(obj2_gpu.p.a) .== 3.7)
                @test all(Array(obj2_gpu.p.b) .== 0)
            end

            # Broadcast @..
            obj = StructuredMeshObject(mesh, 
                simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[0.5, 0.5, 0.5][1:dims]
            )
            obj._pReference .= [false, true]
            obj.p.a .= 7.0
            obj.p.b .= 2
            obj2 = StructuredMeshObject(mesh, 
                simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[0.5, 0.5, 0.5][1:dims]
            )
            obj2._pReference .= [false, true]
            DifferentialEquations.DiffEqBase.@.. obj2 = obj * 0.1 + 3.0
            @test all(obj2.p.a .== 3.7)
            @test all(obj2.p.b .== 0)
            if CUDA.has_cuda()
                obj_gpu = toGPU(obj)
                obj2_gpu = toGPU(obj2)
                obj2_gpu.p.a .= 0.0
                DifferentialEquations.DiffEqBase.@.. obj2_gpu = obj_gpu * 0.1 + 3.0
                @test all(Array(obj2_gpu.p.a) .== 3.7)
                @test all(Array(obj2_gpu.p.b) .== 0)
            end
        end
    end

    @testset "StructuredMeshObject - GPU kernels" begin
        for dims in 1:3
            mesh = StructuredMesh(
                dims;
                properties = props,
            )
            # Kernel works
            if CUDA.has_cuda()
                function test_kernel_object!(x)
                    x.p.a
                    nothing
                end

                obj = StructuredMeshObject(mesh, 
                    simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[0.5, 0.5, 0.5][1:dims]
                )

                obj_gpu = toGPU(obj)
                @test_nowarn CUDA.@cuda test_kernel_object!(obj_gpu)
            end
        end
    end

    @testset "StructuredMeshObject - DifferentialEquations compatibility" begin
        for dims in 1:3
            mesh = StructuredMesh(
                dims;
                properties = props,
            )
            # DifferentialEquations.jl compatibility
            function fODE!(du, u, p, t)
                @. du.p.a = 1
                return
            end
            obj = StructuredMeshObject(mesh, 
                simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[0.5, 0.5, 0.5][1:dims]
            )
            obj._pReference .= [false, true]
            prob = ODEProblem(fODE!, obj, (0.0, 1.0))
            integrator = init(prob, Euler(), dt=0.1, save_everystep=false)
            for i in 1:10
                step!(integrator)
            end
            @test all(integrator.u.p.a .≈ 1.0)
            if CUDA.has_cuda()
                obj = StructuredMeshObject(mesh, 
                    simulationBox=[0 1;0 1;0 1][1:dims,:], gridSpacing=[0.5, 0.5, 0.5][1:dims]
                )
                obj._pReference .= [false, true]
                obj_gpu = toGPU(obj)
                prob = ODEProblem(fODE!, obj_gpu, (0.0, 1.0))
                integrator = init(prob, Euler(), dt=0.1, save_everystep=false)
                for i in 1:10
                    step!(integrator)
                end
                @test all(Array(integrator.u.p.a) .≈ 1.0)
            end
        end
    end
end