using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector
import CellBasedModels: CommunityPointMeta

@testset verbose = true "ABM - Point" begin

    @testset "AgentPoint" begin

        for dims in 0:3
            agent = AgentPoint(dims)
            @test agent isa AgentPoint
            @test CellBasedModels.spatialDims(agent) === dims
            if dims >= 1
                @test haskey(agent.n, :x)
            else
                @test !haskey(agent.n, :x)
            end
            if dims >= 2
                @test haskey(agent.n, :y)
            else
                @test !haskey(agent.n, :y)
            end
            if dims >= 3
                @test haskey(agent.n, :z)
            else
                @test !haskey(agent.n, :z)
            end
        end

        agent = AgentPoint(
            2;
            properties = (
                mass = Parameter(AbstractFloat, defaultValue=1.0, description="Mass of the agent"),
                velocity = AbstractFloat,
            )
        )
        @test haskey(agent.n, :mass)
        @test haskey(agent.n, :velocity)

    end

    @testset "AgentPointObject" begin

        agent = AgentPoint(3)

        @addRule model=agent scope=biochemistry function f!(uNew, u, p, t)

            for i in iterateAgents(u)
                addAgent!(uNew, 
                    (
                        x = 0.0,
                        y = 0.0,
                        z = 0.0,
                    )
                )
            end

        end

        agentObject = AgentPointObject(
            agent;
            N = 10,
            NCache = 20,
        )
        @test agentObject isa AgentPointObject

        problem = CBProblem(
            agent,
            agentObject,
            (0.0, 1.0),
        )

        integrator = init(
            problem,
            dt=0.1,
            biochemistry=Euler()
        )

        for i in 1:5
            step!(integrator)
        end

    end

end