using BenchmarkTools
using CUDA
using DifferentialEquations

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
        @test typeof(community) == CommunityPoint{typeof(agent), 2, 2}
        community = CommunityPoint(agent, 2, 5)
        @test typeof(community) == CommunityPoint{typeof(agent), 2, 5}
        @test community._N[1] == 2 &&
            community._NCache[1] == 5 &&
            community._NNew[1] == 2 &&
            community._idMax[1] == 2 &&
            community._NFlag[1] == false

        f!(x) = @. x = 5 * cos(x) + 1.0

        f!(community)

    end

end