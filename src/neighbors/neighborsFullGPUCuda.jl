function initNeighborsGPU(
        ::Int,
        ::NeighborsFull,
        meshParameters::NamedTuple
    )

    permTable = Dict()
    for (name, prop) in pairs(meshParameters)
        permTable[name] = CUDA.zeros(Int, lengthCache(prop))
    end

    permTableNamed = NamedTuple{tuple(keys(permTable)...)}(values(permTable))

    NeighborsFull{GPUCuda, typeof(meshParameters), typeof(permTableNamed)}(meshParameters, permTableNamed)

end
