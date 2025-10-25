using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector
import CellBasedModels: CommunityPointMeta

@testset verbose = true "ABM - Point" begin

    @testset "Agent" begin

        for dims in 0:3
            agent = AgentPoint(dims)
            @test agent isa AgentPoint
            @test ndims(agent) === dims
            if dims >= 1
                @test haskey(agent.propertiesAgent, :x)
            else
                @test !haskey(agent.propertiesAgent, :x)
            end
            if dims >= 2
                @test haskey(agent.propertiesAgent, :y)
            else
                @test !haskey(agent.propertiesAgent, :y)
            end
            if dims >= 3
                @test haskey(agent.propertiesAgent, :z)
            else
                @test !haskey(agent.propertiesAgent, :z)
            end
        end

        agent = AgentPoint(
            2;
            propertiesAgent = (
                mass = Parameter(AbstractFloat, defaultValue=1.0, description="Mass of the agent"),
                velocity = AbstractFloat,
            )
        )
        @test haskey(agent.propertiesAgent, :mass)
        @test haskey(agent.propertiesAgent, :velocity)

    end

    @testset "Community" begin

        agent = AgentPoint(2)
        community = CommunityPoint(agent, 2)
        community = CommunityPoint(agent, 3, 5)
        @test community._m._N[] == 3
        @test community._m._NCache[] == 5
        @test community._m._NNew[] == 3
        @test community._m._idMax[] == 3
        @test community._m._id[1:3] == collect(1:3)
        @test community._m._removed == zeros(Int, 5)
        @test community._m._reorderedFlag[] == false
        @test community._m._overflowFlag[] == false
        @test all([length(p) == 5 for p in community._pa])

        agent = AgentPoint(
                2,
                propertiesAgent = (
                    velocity = AbstractFloat,
                    idea = Integer,
                    ok = Bool,
                )
            )

        community = CommunityPoint(agent, 3, 5)

        f!(x) = @. x = 5 * (x + 1.0)

        community = CellBasedModels.setCopyParameters(community, (:x, :velocity))
        f!(community)
        @test community.x == 5 .* ones(3)
        @test community.y == zeros(3)
        @test community.velocity == 5 .* ones(3)
        @test community.idea == zeros(Int, 3)
        @test community.ok == zeros(Bool, 3)

        community = CommunityPoint(agent, 3, 5)
        fODE!(du, u, p, t) = @. du.x = 1
        prob = ODEProblem(fODE!, community, (0.0, 1.0))
        integrator = init(prob, Euler(), dt=0.1, save_everystep=false)
        for i in 1:10
            step!(integrator)
        end
        @test all(integrator.u.x .≈ 1.)

        if CUDA.has_cuda()

            community = CommunityPoint(agent, 3, 5)
            community = CellBasedModels.setCopyParameters(community, (:x, :velocity))
            community_gpu = toGPU(community)

            f_gpu!(x) = @. x = 5 * (x + 1.0)

            f_gpu!(community_gpu)
            @test Array(community_gpu.x) == 5 .* ones(3)
            @test Array(community_gpu.y) == zeros(3)
            @test Array(community_gpu.velocity) == 5 .* ones(3)
            @test Array(community_gpu.idea) == zeros(Int, 3)
            @test Array(community_gpu.ok) == zeros(Bool, 3)

            f_gpu_kernel!(community)  = nothing

            community = CommunityPoint(agent, 3, 5)
            community = CellBasedModels.setCopyParameters(community, (:x,))
            community_gpu = toGPU(community)
            CUDA.@cuda f_gpu_kernel!(community_gpu)

            community = CommunityPoint(agent, 3, 5)
            community_gpu = toGPU(community)
            prob = ODEProblem(fODE!, community_gpu, (0.0, 1.0))
            integrator = init(prob, Euler(), dt=0.1, save_everystep=false)
            for i in 1:10
                step!(integrator)
            end
            @test all(integrator.u.x .≈ 1.)

        end

        # Kernel iterator
        kernel_iterate!(community, x) = @inbounds Threads.@threads for i in loopOverAgents(community)
            x[i] += 1
        end

        agent = AgentPoint(3)
        community = CommunityPoint(agent, 10, 20)
        x = zeros(Float64, 20)
        kernel_iterate!(community, x)
        @test [x[i] for i in 1:10] == ones(10)
        @test [x[i] for i in 11:20] == zeros(10)

        if CUDA.has_cuda()

            community = CommunityPoint(agent, 10, 20)
            community_gpu = toGPU(community)
            x = CUDA.zeros(Float64, 20)

            kernel_iterate_gpu!(community, x) = @inbounds for i in loopOverAgents(community)
                x[i] = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            end

            CUDA.@cuda threads=5 kernel_iterate_gpu!(community_gpu, x)
            @test Array(x)[1:5] == 1:5
            @test Array(x)[6:10] == 1:5
            @test Array(x)[11:end] == zeros(10)

        end

        # removeAgent!
        agent = AgentPoint(2)
        community = CommunityPoint(agent, 3, 5)

        removeAgent!(community, 2)

        @test community._m._reorderedFlag[] == true
        @test community._m._removed == [false, true, false, false, false]

        community = CommunityPoint(agent, 3, 5)
        removeAgent!(community, 5)

        @test community._m._reorderedFlag[] == false
        @test community._m._removed == [false, false, false, false, false]

        if CUDA.has_cuda()

            agent = AgentPoint(2)
            community = CommunityPoint(agent, 3, 5)
            community_gpu = toGPU(community)

            @test_throws ErrorException removeAgent!(community_gpu, 2)

            function removeAgent_kernel!(community, pos)
                CUDA.@cuprintln(typeof(community._m))
                removeAgent!(community, pos)
                return
            end

            CUDA.@cuda removeAgent_kernel!(community_gpu, 2)

            @test Array(community_gpu._m._reorderedFlag)[1] == true
            @test Array(community_gpu._m._removed) == [false, true, false, false, false]

            community = CommunityPoint(agent, 3, 5)
            community_gpu = toGPU(community)
            CUDA.@cuda removeAgent_kernel!(community_gpu, 5)

            @test community._m._reorderedFlag[] == false
            @test community._m._removed == [false, false, false, false, false]

        end

        # addAgent!
        agent = AgentPoint(2)
        community = CommunityPoint(agent, 3, 5)

        addAgent!(community, (x=1.,y=2.))

        @test community._m._id[4] == 4
        @test community._pa.x[4] == 1.
        @test community._pa.y[4] == 2.
        @test community._m._NNew[] == 4
        @test community._m._idMax[] == 4

        if CUDA.has_cuda()

            agent = AgentPoint(2)
            community = CommunityPoint(agent, 3, 5)
            community_gpu = toGPU(community)

            @test_throws ErrorException addAgent!(community_gpu, (x=1.,y=2.))

            function addAgent_kernel!(community)
                addAgent!(community, (x=1.,y=2.))
                return
            end

            CUDA.@cuda addAgent_kernel!(community_gpu)

            @test Array(community_gpu._m._id)[4] == 4
            @test Array(community_gpu._pa.x)[4] == 1.
            @test Array(community_gpu._pa.y)[4] == 2.
            @test Array(community_gpu._m._NNew)[1] == 4
            @test Array(community_gpu._m._idMax)[1] == 4

        end

    end

end