using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector
import CellBasedModels: CommunityPointMeta

using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector
import CellBasedModels: CommunityPointMeta

begin

    begin

        for dims in 0:3
            agent = AgentPoint(dims)
             agent isa AgentPoint
             ndims(agent) === dims
            if dims >= 1
                 haskey(agent.propertiesAgent, :x)
            else
                 !haskey(agent.propertiesAgent, :x)
            end
            if dims >= 2
                 haskey(agent.propertiesAgent, :y)
            else
                 !haskey(agent.propertiesAgent, :y)
            end
            if dims >= 3
                 haskey(agent.propertiesAgent, :z)
            else
                 !haskey(agent.propertiesAgent, :z)
            end
        end

        agent = AgentPoint(
            2;
            propertiesAgent = (
                mass = Parameter(AbstractFloat, defaultValue=1.0, description="Mass of the agent"),
                velocity = AbstractFloat,
            )
        )
         haskey(agent.propertiesAgent, :mass)
         haskey(agent.propertiesAgent, :velocity)

    end

    begin

        agent = AgentPoint(2)
        community = CommunityPoint(agent, 2)
        community = CommunityPoint(agent, 3, 5)
         community._m._N[1] == 3
         community._m._NCache[1] == 5
         community._m._NNew[1] == 3
         community._m._idMax[1] == 3
         community._m._id[1:3] == collect(1:3)
         community._m._removed == zeros(Int, 5)
         community._m._overflowFlag[1] == false
         community.N == 3
         community.NCache == 5
         all([length(p) == 5 for p in community._pa])

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
         community.x == 5 .* ones(3)
         community.y == zeros(3)
         community.velocity == 5 .* ones(3)
         community.idea == zeros(Int, 3)
         community.ok == zeros(Bool, 3)

        community = CommunityPoint(agent, 3, 5)
        fODE!(du, u, p, t) = @. du.x = 1
        prob = ODEProblem(fODE!, community, (0.0, 1.0))
        integrator = init(prob, Euler(), dt=0.1, save_everystep=false)
        for i in 1:10
            step!(integrator)
        end
         all(integrator.u.x .≈ 1.)

        if CUDA.has_cuda()

            community = CommunityPoint(agent, 3, 5)
            community = CellBasedModels.setCopyParameters(community, (:x, :velocity))
            community_gpu = toGPU(community)

            f_gpu!(x) = @. x = 5 * (x + 1.0)

            f_gpu!(community_gpu)
             Array(community_gpu.x) == 5 .* ones(3)
             Array(community_gpu.y) == zeros(3)
             Array(community_gpu.velocity) == 5 .* ones(3)
             Array(community_gpu.idea) == zeros(Int, 3)
             Array(community_gpu.ok) == zeros(Bool, 3)

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
             all(integrator.u.x .≈ 1.)

        end

        # Kernel iterator
        kernel_iterate!(community, x) = @inbounds Threads.@threads for i in loopOverAgents(community)
            x[i] += 1
        end

        agent = AgentPoint(3)
        community = CommunityPoint(agent, 10, 20)
        x = zeros(Float64, 20)
        kernel_iterate!(community, x)
         [x[i] for i in 1:10] == ones(10)
         [x[i] for i in 11:20] == zeros(10)

        if CUDA.has_cuda()

            community = CommunityPoint(agent, 10, 20)
            community_gpu = toGPU(community)
            x = CUDA.zeros(Float64, 20)

            kernel_iterate_gpu!(community, x) = @inbounds for i in loopOverAgents(community)
                x[i] = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            end

            CUDA.@cuda threads=5 kernel_iterate_gpu!(community_gpu, x)
             Array(x)[1:5] == 1:5
             Array(x)[6:10] == 1:5
             Array(x)[11:end] == zeros(10)

        end

        # removeAgent!

        agent = AgentPoint(2)
        community = CommunityPoint(agent, 3, 5)

        removeAgent!(community, 2)

        @test community._m._reorderedFlag[1] == true
        @test community._m._removed == [false, true, false, false, false]

        

    end

end