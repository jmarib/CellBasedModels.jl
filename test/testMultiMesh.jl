using Test
using CUDA
using DifferentialEquations
using Adapt
using BenchmarkTools

@testset verbose = true "ABM - MultiMesh (CPU + GPU)" begin

    #######################################################################
    # HELPERS
    #######################################################################
    props = (
        a = Parameter(Float64, description="param a", dimensions=:M, defaultValue=1.0),
        b = Parameter(Int, description="param b", dimensions=:count, defaultValue=0),
    )

    function make_meshes(D)
        mA = UnstructuredMesh(D; propertiesAgent=props, scopePosition=:propertiesAgent)
        mB = UnstructuredMesh(D; propertiesNode=props, scopePosition=:propertiesNode)
        return (meshA=mA, meshB=mB)
    end

    #######################################################################
    # TEST 1: MultiMesh Construction
    #######################################################################
    @testset "MultiMesh - Construction and I/O" begin
        for D in 0:3
            meshes = make_meshes(D)
            mm = MultiMesh(; meshes...)

            @test mm isa MultiMesh
            @test CellBasedModels.spatialDims(mm) == D
            @test length(mm) == 2
            @test mm[:meshA] === meshes.meshA
            @test mm.meshB === meshes.meshB

            io = IOBuffer()
            show(io, mm)
            s = String(take!(io))
            @test occursin("MultiMeshMesh with dimensions $D", s)

            io = IOBuffer()
            show(io, typeof(mm))
            s = String(take!(io))
            @test occursin("MultiMeshMesh Type with dimensions $D", s)
        end

        # Invalid combinations
        m1 = UnstructuredMesh(1; propertiesAgent=props, scopePosition=:propertiesAgent)
        m2 = UnstructuredMesh(2; propertiesNode=props, scopePosition=:propertiesNode)
        @test_throws AssertionError MultiMesh(meshA=m1, meshB=m2)
        @test_throws ErrorException MultiMesh(_bad=m1)
    end


    #######################################################################
    # TEST 2: MultiMeshObject (CPU + GPU)
    #######################################################################
    @testset "MultiMeshObject - Construction & Platform consistency" begin
        for D in 0:3
            meshes = make_meshes(D)
            mm = MultiMesh(; meshes...)

            # CPU
            mmo = MultiMeshObject(mm; agentN=2, agentNCache=4)
            @test mmo isa MultiMeshObject
            @test length(mmo) == 2
            @test size(mmo) == (2,)
            @test mmo._meshes.meshA isa UnstructuredMeshObject

            io = IOBuffer()
            show(io, mmo)
            s = String(take!(io))
            @test occursin("MultiMeshObject with dimensions $D", s)

            io = IOBuffer()
            show(io, typeof(mmo))
            s = String(take!(io))
            @test occursin("MultiMeshObject Type with dimensions $D", s)

            # GPU
            if CUDA.has_cuda()
                gpu_mmo = toGPU(mmo)
                @test gpu_mmo isa MultiMeshObject
                @test typeof(gpu_mmo._meshes.meshA.a) <: CUDA.CuArray
                cpu_back = toCPU(gpu_mmo)
                @test cpu_back isa MultiMeshObject
            end
        end
    end


    #######################################################################
    # TEST 3: Copy / Zero / Copyto! (CPU + GPU)
    #######################################################################
    @testset "MultiMeshObject - Copying semantics" begin
        for D in 0:3
            meshes = make_meshes(D)
            mm = MultiMesh(; meshes...)
            mmo = MultiMeshObject(mm; agentN=3, agentNCache=5)

            cpy = copy(mmo)
            zro = zero(mmo)

            @test cpy isa MultiMeshObject
            @test zro isa MultiMeshObject
            @test typeof(cpy._meshes) == typeof(mmo._meshes)
            @test typeof(zro._meshes) == typeof(mmo._meshes)

            copyto!(cpy, mmo)
            @test cpy._meshes.meshA.a == mmo._meshes.meshA.a

            if CUDA.has_cuda()
                gpu_mmo = toGPU(mmo)
                gpu_cpy = toGPU(cpy)
                gpu_zro = toGPU(zro)
                copyto!(gpu_cpy, gpu_mmo)
                @test all(Array(gpu_cpy._meshes.meshA.a) .== Array(gpu_mmo._meshes.meshA.a))
                copyto!(gpu_zro, gpu_mmo)
                @test gpu_zro isa MultiMeshObject
            end
        end
    end


    #######################################################################
    # TEST 4: Broadcasting (CPU + GPU)
    #######################################################################
    @testset "MultiMeshObject - Broadcasting behavior" begin
        for D in 0:3
            meshes = make_meshes(D)
            mm = MultiMesh(; meshes...)
            mmo = MultiMeshObject(mm; agentN=2, agentNCache=4)
            mmo2 = copy(mmo)

            # Basic broadcast
            @. mmo2 = mmo * 0.5 + 1.0
            @test all(mmo2._meshes.meshA.a .≈ 1.5)

            # DifferentialEquations broadcast (@..)
            DifferentialEquations.DiffEqBase.@.. mmo2 = mmo * 2.0
            @test all(mmo2._meshes.meshA.a .≈ 2.0)

            if CUDA.has_cuda()
                mmo_gpu = toGPU(mmo)
                mmo2_gpu = toGPU(mmo2)
                @. mmo2_gpu = mmo_gpu * 0.5 + 1.0
                @test all(Array(mmo2_gpu._meshes.meshA.a) .≈ 1.5)
                DifferentialEquations.DiffEqBase.@.. mmo2_gpu = mmo_gpu * 2.0
                @test all(Array(mmo2_gpu._meshes.meshA.a) .≈ 2.0)
            end
        end
    end


    #######################################################################
    # TEST 5: Kernel compatibility (GPU)
    #######################################################################
    if CUDA.has_cuda()
        @testset "MultiMeshObject - CUDA kernel use" begin
            meshes = make_meshes(2)
            mm = MultiMesh(; meshes...)
            mmo = MultiMeshObject(mm; agentN=2, agentNCache=4)
            mmo_gpu = toGPU(mmo)

            function kernel_multimesh!(x)
                x.meshA.a[1] += 1.0
                nothing
            end

            @test_nowarn CUDA.@cuda threads=1 kernel_multimesh!(mmo_gpu)
        end
    end


    #######################################################################
    # TEST 6: ODE integration (CPU + GPU)
    #######################################################################
    @testset "MultiMeshObject - ODE integration" begin
        meshes = make_meshes(2)
        mm = MultiMesh(; meshes...)
        mmo = MultiMeshObject(mm; agentN=2, agentNCache=4)

        function fODE!(du, u, p, t)
            @. du.meshA.a = 1.0
        end

        prob = ODEProblem(fODE!, mmo, (0.0, 1.0))
        integrator = init(prob, Euler(), dt=0.1, save_everystep=false)
        for _ in 1:5
            step!(integrator)
        end
        @test all(integrator.u.meshA.a .≈ 1.0)

        if CUDA.has_cuda()
            gpu_mmo = toGPU(mmo)
            prob_gpu = ODEProblem(fODE!, gpu_mmo, (0.0, 1.0))
            integ_gpu = init(prob_gpu, Euler(), dt=0.1, save_everystep=false)
            for _ in 1:5
                step!(integ_gpu)
            end
            @test all(Array(integ_gpu.u.meshA.a) .≈ 1.0)
        end
    end


    #######################################################################
    # TEST 7: Performance (CPU + GPU)
    #######################################################################
    @testset "MultiMeshObject - Performance sanity" begin
        meshes = make_meshes(2)
        mm = MultiMesh(; meshes...)
        mmo = MultiMeshObject(mm; agentN=10, agentNCache=20)
        @test_nowarn @btime copy($mmo)
        @test_nowarn @btime zero($mmo)
        @test_nowarn @btime copyto!($mmo, $mmo)
        @test_nowarn @btime @. $mmo = $mmo * 0.1 + 1.0

        if CUDA.has_cuda()
            mmo_gpu = toGPU(mmo)
            @test_nowarn @btime copy($mmo_gpu)
            @test_nowarn @btime zero($mmo_gpu)
            @test_nowarn @btime copyto!($mmo_gpu, $mmo_gpu)
            @test_nowarn @btime @. $mmo_gpu = $mmo_gpu * 0.1 + 1.0
        end
    end
end
