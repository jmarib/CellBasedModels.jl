using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector

@testset verbose = true "ABM - UnstructuredMesh" begin

    #######################################################################
    # HELPERS
    #######################################################################
    props = (
            a = Parameter(Float64, description="param a", dimensions=:M, defaultValue=1.0),
            b = Parameter(Int, description="param b", dimensions=:count, defaultValue=0),
        )

    all_scopes = (:propertiesAgent, :propertiesNode, :propertiesEdge, :propertiesFace, :propertiesVolume)

    #######################################################################
    # TEST 1: UnstructuredMesh
    #######################################################################
    @testset "UnstructuredMesh - full coverage" begin
        for dims in 0:3
            for scope in all_scopes
                mesh = UnstructuredMesh(
                    dims;
                    scopePosition = scope,
                    propertiesAgent  = scope == :propertiesAgent  ? props : nothing,
                    propertiesNode   = scope == :propertiesNode   ? props : nothing,
                    propertiesEdge   = scope == :propertiesEdge   ? props : nothing,
                    propertiesFace   = scope == :propertiesFace   ? props : nothing,
                    propertiesVolume = scope == :propertiesVolume ? props : nothing,
                )

                @test mesh isa UnstructuredMesh
                @test CellBasedModels.spatialDims(mesh) == dims
                @test CellBasedModels.scopePosition(mesh) in 1:5
                @test getfield(mesh, scope) !== nothing
            end
        end

        # Invalid dimension
        @test_throws ErrorException UnstructuredMesh(-1)
        @test_throws ErrorException UnstructuredMesh(4)

        # Invalid scopePosition
        @test_throws ErrorException UnstructuredMesh(2; scopePosition=:invalid)

        # Duplicate protected parameter name (e.g., x)
        bad_props = (x = Parameter(Float64, description="duplicate", dimensions=:L, defaultValue=0.0),)
        @test_throws ErrorException UnstructuredMesh(1; propertiesAgent=bad_props, scopePosition=:propertiesAgent)

        # Show output test
        io = IOBuffer()
        show(io, UnstructuredMesh(2; propertiesAgent=props, scopePosition=:propertiesAgent))
        output = String(take!(io))
        @test occursin("UnstructuredMesh with dimensions 2", output)
    end


    #######################################################################
    # TEST 2: UnstructuredMeshObjectField
    #######################################################################
    @testset "UnstructuredMeshObjectField - construction and logic" begin

        # Valid case
        field = UnstructuredMeshObjectField(props; N=3, NCache=5)
        @test field isa UnstructuredMeshObjectField
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

        # No meshProperties → returns nothing
        @test UnstructuredMeshObjectField(nothing) === nothing

        # With id=false
        field2 = UnstructuredMeshObjectField(props; N=2, NCache=5, id=false)
        @test field2._id === nothing
        @test field2._idMax === nothing

        # Show output formatting
        io = IOBuffer()
        show(io, field)
        txt = String(take!(io))
        @test occursin("_id", txt)
        @test occursin("_AddedAgents", txt)

        # Copy
        field = UnstructuredMeshObjectField(props; N=3, NCache=5)
        field._pReference .= [true, false]
        fieldCopy = copy(field)
        field.a .= 7.0
        field.b .= 2
        @test fieldCopy.a == fill(7.0, 3)
        @test fieldCopy.b == zeros(Int, 3)

        # zero
        field = UnstructuredMeshObjectField(props; N=3, NCache=5)
        field._pReference .= [true, false]
        field.a .= 7.0
        field.b .= 2        
        fieldZero = zero(field)
        @test fieldZero.a == fill(7.0, 3)
        @test fieldZero.b == zeros(Int, 3)

        # toGPU/toCPU
        if CUDA.has_cuda()
            field = UnstructuredMeshObjectField(props; N=3, NCache=5)
            field._pReference .= [true, false]
            field.a .= 7.0
            field.b .= 2

            field_gpu = toGPU(field)

            @test typeof(field_gpu._id) <: CUDA.CuArray
            @test typeof(field_gpu.a) <: CUDA.CuArray

            field_cpu = toCPU(field_gpu)

            for i in fieldnames(UnstructuredMeshObjectField)
                @test typeof(getfield(field_cpu, i)) <: Threads.Atomic ? getfield(field_cpu, i)[] == getfield(field, i)[] : getfield(field_cpu, i) == getfield(field, i)
            end
        end

        # Copyto!
        field = UnstructuredMeshObjectField(props; N=3, NCache=5)
        field._pReference .= [false, true]
        field.a .= 7.0
        field.b .= 2
        field2 = UnstructuredMeshObjectField(props; N=3, NCache=5)
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
        field = UnstructuredMeshObjectField(props; N=3, NCache=5)
        field._pReference .= [false, true]
        field.a .= 7.0
        field.b .= 2
        field2 = UnstructuredMeshObjectField(props; N=3, NCache=5)
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
        field = UnstructuredMeshObjectField(props; N=3, NCache=5)
        field._pReference .= [false, true]
        field.a .= 7.0
        field.b .= 2
        field2 = UnstructuredMeshObjectField(props; N=3, NCache=5)
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

    end

    #######################################################################
    # TEST 3: UnstructuredMeshObject
    #######################################################################
    @testset "UnstructuredMeshObject - integrated construction" begin
        for dims in 0:3
            for scope in all_scopes
                mesh = UnstructuredMesh(
                    dims;
                    scopePosition = scope,
                    propertiesAgent  = scope == :propertiesAgent  ? props : nothing,
                    propertiesNode   = scope == :propertiesNode   ? props : nothing,
                    propertiesEdge   = scope == :propertiesEdge   ? props : nothing,
                    propertiesFace   = scope == :propertiesFace   ? props : nothing,
                    propertiesVolume = scope == :propertiesVolume ? props : nothing,
                )

                obj = UnstructuredMeshObject(mesh, 
                    agentN=2, agentNCache=4, 
                    nodeN=2, nodeNCache=4, 
                    edgeN=2, edgeNCache=4, 
                    faceN=2, faceNCache=4, 
                    volumeN=2, volumeNCache=4
                )
                @test obj isa UnstructuredMeshObject
                @test obj._scope == CellBasedModels.scopePosition(mesh)
                @test obj.a isa Union{Nothing, UnstructuredMeshObjectField}
                @test obj.n isa Union{Nothing, UnstructuredMeshObjectField}
                @test obj.e isa Union{Nothing, UnstructuredMeshObjectField}
                @test obj.f isa Union{Nothing, UnstructuredMeshObjectField}
                @test obj.v isa Union{Nothing, UnstructuredMeshObjectField}

                # Invalid cache values
                @test_throws ErrorException UnstructuredMeshObject(mesh, agentN=4, agentNCache=2)
                @test_throws ErrorException UnstructuredMeshObject(mesh, agentN=-1, agentNCache=0)

            end
        end
    end

    # @testset "Community" begin

    #     agent = AgentPoint(2)
    #     community = CommunityPoint(agent, 2)
    #     community = CommunityPoint(agent, 3, 5)
    #     @test community._m._N[] == 3
    #     @test community._m._NCache[] == 5
    #     @test community._m.id[1:3] == collect(1:3)
    #     @test community._m._idMax[] == 3
    #     @test community._m._flagsRemoved == zeros(Bool, 5)
    #     @test community._m._NRemoved[] == 0
    #     @test all(community._m._NRemovedThread .== zeros(Int,Threads.nthreads()))
    #     @test community._m._NAdded[] == 0
    #     @test all(community._m._NAddedThread .== zeros(Int,Threads.nthreads()))
    #     @test community._m._addedAgents == [NamedTuple{(:x,:y),Tuple{Float64,Float64}}[] for i in 1:Threads.nthreads()]
    #     @test community._m._flagOverflow[] == false
    #     @test all([length(p) == 5 for p in community._pa])

    #     agent = AgentPoint(
    #             2,
    #             propertiesAgent = (
    #                 velocity = AbstractFloat,
    #                 idea = Integer,
    #                 ok = Bool,
    #             )
    #         )

    #     community = CommunityPoint(agent, 3, 5)

    #     f!(x) = @. x = 5 * (x + 1.0)

    #     community = CellBasedModels.setCopyParameters(community, (:x, :velocity))
    #     f!(community)
    #     @test community.x == 5 .* ones(3)
    #     @test community.y == zeros(3)
    #     @test community.velocity == 5 .* ones(3)
    #     @test community.idea == zeros(Int, 3)
    #     @test community.ok == zeros(Bool, 3)

    #     community = CommunityPoint(agent, 3, 5)
    #     fODE!(du, u, p, t) = @. du.x = 1
    #     prob = ODEProblem(fODE!, community, (0.0, 1.0))
    #     integrator = init(prob, Euler(), dt=0.1, save_everystep=false)
    #     for i in 1:10
    #         step!(integrator)
    #     end
    #     @test all(integrator.u.x .≈ 1.)

    #     if CUDA.has_cuda()

    #         community = CommunityPoint(agent, 3, 5)
    #         community = CellBasedModels.setCopyParameters(community, (:x, :velocity))
    #         community_gpu = toGPU(community)

    #         f_gpu!(x) = @. x = 5 * (x + 1.0)

    #         f_gpu!(community_gpu)
    #         @test Array(community_gpu.x) == 5 .* ones(3)
    #         @test Array(community_gpu.y) == zeros(3)
    #         @test Array(community_gpu.velocity) == 5 .* ones(3)
    #         @test Array(community_gpu.idea) == zeros(Int, 3)
    #         @test Array(community_gpu.ok) == zeros(Bool, 3)

    #         f_gpu_kernel!(community)  = nothing

    #         community = CommunityPoint(agent, 3, 5)
    #         community = CellBasedModels.setCopyParameters(community, (:x,))
    #         community_gpu = toGPU(community)
    #         CUDA.@cuda f_gpu_kernel!(community_gpu)

    #         community = CommunityPoint(agent, 3, 5)
    #         community_gpu = toGPU(community)
    #         prob = ODEProblem(fODE!, community_gpu, (0.0, 1.0))
    #         integrator = init(prob, Euler(), dt=0.1, save_everystep=false)
    #         for i in 1:10
    #             step!(integrator)
    #         end
    #         @test all(integrator.u.x .≈ 1.)

    #     end

    #     # Kernel iterator
    #     kernel_iterate!(community, x) = @inbounds Threads.@threads for i in loopOverAgents(community)
    #         x[i] += 1
    #     end

    #     agent = AgentPoint(3)
    #     community = CommunityPoint(agent, 10, 20)
    #     x = zeros(Float64, 20)
    #     kernel_iterate!(community, x)
    #     @test [x[i] for i in 1:10] == ones(10)
    #     @test [x[i] for i in 11:20] == zeros(10)

    #     if CUDA.has_cuda()

    #         community = CommunityPoint(agent, 10, 20)
    #         community_gpu = toGPU(community)
    #         x = CUDA.zeros(Float64, 20)

    #         kernel_iterate_gpu!(community, x) = @inbounds for i in loopOverAgents(community)
    #             x[i] = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    #         end

    #         CUDA.@cuda threads=5 kernel_iterate_gpu!(community_gpu, x)
    #         @test Array(x)[1:5] == 1:5
    #         @test Array(x)[6:10] == 1:5
    #         @test Array(x)[11:end] == zeros(10)

    #     end

    #     # removeAgent!
    #     agent = AgentPoint(2)
    #     community = CommunityPoint(agent, 3, 5)

    #     removeAgent!(community, 2)

    #     @test community._m._flagsRemoved == [false, true, false, false, false]

    #     community = CommunityPoint(agent, 3, 5)
    #     removeAgent!(community, 5)

    #     @test community._m._flagsRemoved == [false, false, false, false, false]

    #     if CUDA.has_cuda()

    #         agent = AgentPoint(2)
    #         community = CommunityPoint(agent, 3, 5)
    #         community_gpu = toGPU(community)

    #         @test_throws ErrorException removeAgent!(community_gpu, 2)

    #         function removeAgent_kernel!(community, pos)
    #             CUDA.@cuprintln(typeof(community._m))
    #             removeAgent!(community, pos)
    #             return
    #         end

    #         CUDA.@cuda removeAgent_kernel!(community_gpu, 2)

    #         @test Array(community_gpu._m._flagsRemoved) == [false, true, false, false, false]

    #         community = CommunityPoint(agent, 3, 5)
    #         community_gpu = toGPU(community)
    #         CUDA.@cuda removeAgent_kernel!(community_gpu, 5)

    #         @test community._m._flagsRemoved == [false, false, false, false, false]

    #     end

    #     # addAgent!
    #     agent = AgentPoint(2)
    #     community = CommunityPoint(agent, 3, 5)

    #     addAgent!(community, (x=1.,y=2.))

    #     @test community._m.id[4] == 4
    #     @test community._pa.x[4] == 1.
    #     @test community._pa.y[4] == 2.
    #     @test community._m._NNew[] == 4
    #     @test community._m._idMax[] == 4

    #     if CUDA.has_cuda()

    #         agent = AgentPoint(2)
    #         community = CommunityPoint(agent, 3, 5)
    #         community_gpu = toGPU(community)

    #         @test_throws ErrorException addAgent!(community_gpu, (x=1.,y=2.))

    #         function addAgent_kernel!(community)
    #             addAgent!(community, (x=1.,y=2.))
    #             return
    #         end

    #         CUDA.@cuda addAgent_kernel!(community_gpu)

    #         @test Array(community_gpu._m.id)[4] == 4
    #         @test Array(community_gpu._pa.x)[4] == 1.
    #         @test Array(community_gpu._pa.y)[4] == 2.
    #         @test Array(community_gpu._m._NNew)[1] == 4
    #         @test Array(community_gpu._m._idMax)[1] == 4

    #     end

    # end

end