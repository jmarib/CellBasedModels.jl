using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt

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
            @test haskey(agent.propertiesAgent, :id)
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
        @test typeof(community) == CommunityPoint{2, (:x,:y,:id), Tuple{Vector{Float64},Vector{Float64},Vector{Int64}}, 2, 2}
        community = CommunityPoint(agent, 3, 5)
        @test typeof(community) == CommunityPoint{2, (:x,:y,:id), Tuple{Vector{Float64},Vector{Float64},Vector{Int64}}, 3, 5}
        @test community._N[1] == 3 &&
            community._NCache[1] == 5 &&
            community._NNew[1] == 3 &&
            community._idMax[1] == 3 &&
            community._NFlag[1] == false &&
            all([length(p) == 5 for p in community._propertiesAgent])

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

        f!(community)
        @test community.x == 5 .* ones(3)
        @test community.y == 5 .* ones(3)
        @test community.velocity == 5 .* ones(3)
        @test community.idea == zeros(Int, 3)
        @test community.ok == zeros(Bool, 3)

        community = CommunityPoint(agent, 3, 5)
        fODE!(du, u, p, t) = @. du.x = 5 * (u.x + 1.0)
        prob = ODEProblem(fODE!, community, (0.0, 1.0))
        integrator = init(prob, Euler(), dt=0.1)

        if CUDA.has_cuda()

            community = CommunityPoint(agent, 3, 5)
            community_gpu = Adapt.adapt(CuArray, community)

            f_gpu!(x) = @. x = 5 * (x + 1.0)

            f_gpu!(community_gpu)
            @test Array(community_gpu.x) == 5 .* ones(3)
            @test Array(community_gpu.y) == 5 .* ones(3)
            @test Array(community_gpu.velocity) == 5 .* ones(3)
            @test Array(community_gpu.idea) == zeros(Int, 3)
            @test Array(community_gpu.ok) == zeros(Bool, 3)

            community = CommunityPoint(agent, 3, 5)
            community_gpu = Adapt.adapt(CuArray, community)

            function f_gpu!(community) 
                index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
                stride = gridDim().x * blockDim().x
                x = community.x
                for i in index:stride:community._N[1]
                    @inbounds x[i] = 5 * (x[i] + 1.0)
                end
            end

            CUDA.@cuda threads=32 f_gpu!(community_gpu)
            @test Array(community_gpu.x) == 5 .* ones(3)

        end

    end

end