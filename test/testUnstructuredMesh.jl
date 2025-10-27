using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector

@testset verbose = true "ABM - UnstructuredMeshProperties" begin

    #######################################################################
    # HELPER FUNCTIONS
    #######################################################################
    props =
        (
            a = Parameter(Float64, description="param a", dimensions=:M, defaultValue=1.0),
            b = Parameter(Int, description="param b", dimensions=:count, defaultValue=0)
        )

    all_scopes = 
        (
            :propertiesAgent,
            :propertiesNode,
            :propertiesEdge,
            :propertiesFace,
            :propertiesVolume
        )

    #######################################################################
    # TESTSET 1: UnstructuredMeshProperties
    #######################################################################
    @testset "UnstructuredMeshProperties - all scopes and dimensions" begin
        for dims in 0:3
            for (scopePos, scope) in zip(1:5, all_scopes)
                mesh = UnstructuredMeshProperties(
                    dims;
                    scopePosition = scope,
                    propertiesAgent  = scope == :propertiesAgent  ? props : nothing,
                    propertiesNode   = scope == :propertiesNode   ? props : nothing,
                    propertiesEdge   = scope == :propertiesEdge   ? props : nothing,
                    propertiesFace   = scope == :propertiesFace   ? props : nothing,
                    propertiesVolume = scope == :propertiesVolume ? props : nothing,
                )
                @test mesh isa UnstructuredMeshProperties
                @test CellBasedModels.spatialDims(mesh) == dims
                @test CellBasedModels.scopePosition(mesh) === scopePos

                # Check that the chosen scope has properties
                pfield = getfield(mesh, scope)
                @test pfield !== nothing
                @test haskey(pfield, :a)
                @test CellBasedModels.dtype(pfield.a) == AbstractFloat
            end
        end

        # Invalid dimension
        @test_throws ErrorException UnstructuredMeshProperties(-1)
        @test_throws ErrorException UnstructuredMeshProperties(4)

        # Invalid scope
        @test_throws ErrorException UnstructuredMeshProperties(2; scopePosition=:invalid)

        # Duplicate protected parameter
        bad_props = (x = Parameter(Float64, description="pos x", dimensions=:L, defaultValue=0.0),)
        @test_throws ErrorException UnstructuredMeshProperties(1; propertiesAgent=bad_props, scopePosition=:propertiesAgent)

        # Printing check
        io = IOBuffer()
        show(io, UnstructuredMeshProperties(2; propertiesAgent=props, scopePosition=:propertiesAgent))
        out = String(take!(io))
        @test occursin("UnstructuredMeshProperties with dimensions 2", out)
    end


    #######################################################################
    # TESTSET 2: UnstructuredMeshObjectMeta
    #######################################################################
    @testset "UnstructuredMeshObjectMeta - full combinations" begin

        # Valid with ID
        meta = UnstructuredMeshObjectMeta(props; N=3, NCache=5, id=true)
        @test meta isa UnstructuredMeshObjectMeta
        @test typeof(meta._id) <: Vector{Int}
        @test meta._idMax[] == 3
        @test meta._N isa Threads.Atomic{Int}
        @test meta._FlagOverflow[] == false

        # Valid without ID
        meta2 = UnstructuredMeshObjectMeta(props; N=2, NCache=4, id=false)
        @test meta2._id === nothing

        # Test nothing case
        @test UnstructuredMeshObjectMeta(nothing) === nothing

        # Check multiple scopes
        for scope in all_scopes
            meta = UnstructuredMeshObjectMeta(props; N=2, NCache=3)
            @test meta isa UnstructuredMeshObjectMeta
            @test meta._N isa Threads.Atomic{Int}
            @test meta._NCache isa Threads.Atomic{Int}
            @test meta._FlagsRemoved isa Vector{Int}
            @test length(meta._FlagsRemoved) == 3
        end
    end


    #######################################################################
    # TESTSET 3: UnstructuredMeshObject - all scopes
    #######################################################################
    @testset "UnstructuredMeshObject - comprehensive coverage" begin
        for dims in 1:3
            for scope in all_scopes
                mesh = UnstructuredMeshProperties(
                    dims;
                    scopePosition = scope,
                    propertiesAgent  = scope == :propertiesAgent  ? props : nothing,
                    propertiesNode   = scope == :propertiesNode   ? props : nothing,
                    propertiesEdge   = scope == :propertiesEdge   ? props : nothing,
                    propertiesFace   = scope == :propertiesFace   ? props : nothing,
                    propertiesVolume = scope == :propertiesVolume ? props : nothing,
                )

                # Basic valid creation
                obj = UnstructuredMeshObject(mesh, 2, 4, 1, 2, 1, 2, 1, 2, 1, 2)
                @test obj isa UnstructuredMeshObject
                @test obj._am isa Union{Nothing, UnstructuredMeshObjectMeta}
                @test obj._nm isa Union{Nothing, UnstructuredMeshObjectMeta}
                @test obj._em isa Union{Nothing, UnstructuredMeshObjectMeta}
                @test obj._fm isa Union{Nothing, UnstructuredMeshObjectMeta}
                @test obj._vm isa Union{Nothing, UnstructuredMeshObjectMeta}

                # Verify sizes and structure
                for fld in fieldnames(typeof(obj))
                    val = getfield(obj, fld)
                    if fld in (:_ap, :_np, :_ep, :_fp, :_vp)
                        if val !== nothing
                            @test all(length(v) == 4 || length(v) == 2 for v in values(val))
                        end
                    end
                end

                # Cache consistency checks
                @test_throws ErrorException UnstructuredMeshObject(mesh, 4, 2)
                @test_throws ErrorException UnstructuredMeshObject(mesh, -1, 0)
            end
        end
    end

    # @testset "Community" begin

    #     agent = AgentPoint(2)
    #     community = CommunityPoint(agent, 2)
    #     community = CommunityPoint(agent, 3, 5)
    #     @test community._m._N[] == 3
    #     @test community._m._NCache[] == 5
    #     @test community._m._id[1:3] == collect(1:3)
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

    #     @test community._m._id[4] == 4
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

    #         @test Array(community_gpu._m._id)[4] == 4
    #         @test Array(community_gpu._pa.x)[4] == 1.
    #         @test Array(community_gpu._pa.y)[4] == 2.
    #         @test Array(community_gpu._m._NNew)[1] == 4
    #         @test Array(community_gpu._m._idMax)[1] == 4

    #     end

    # end

end