using BenchmarkTools

N = 100000
n = 100

agent = AgentPoint(2)

community = CommunityPoint(agent, n, N)

f!(x) = x .= 5 .* x .+ 1

@btime f!(community)