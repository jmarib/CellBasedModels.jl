import CellBasedModels: fillPermTable!

function initNeighborsGPU(
        ::Int,
        ::NeighborsFull,
        meshParameters::NamedTuple
    )

    permTable = Dict()
    
    # Find the maximum cache size across all properties
    maxCacheSize = 0
    for (name, prop) in pairs(meshParameters)
        maxCacheSize = max(maxCacheSize, lengthCache(prop))
    end
    
    # Create one reusable GPU buffer per element type across all properties and fields
    typeBuffers = Dict()
    
    for (name, prop) in pairs(meshParameters)
        permTable[name] = CUDA.zeros(Int, lengthCache(prop))
        
        # Create or reuse GPU buffer for each field based on its element type
        for (fieldname, field) in pairs(prop._p)
            elemType = eltype(field)
            if !haskey(typeBuffers, elemType)
                typeBuffers[elemType] = CUDA.similar(field, maxCacheSize)
            end
        end
    end
    
    # Map each property's fields to their corresponding type buffer
    auxBuffers = Dict()
    for (name, prop) in pairs(meshParameters)
        auxBuffers[name] = NamedTuple{fieldnames(typeof(prop._p))}(
            [typeBuffers[eltype(prop._p[fname])] for fname in fieldnames(typeof(prop._p))]
        )
    end

    permTableNamed = NamedTuple{tuple(keys(permTable)...)}(values(permTable))
    auxBuffersNamed = NamedTuple{tuple(keys(auxBuffers)...)}(values(auxBuffers))

    NeighborsFull{GPUCuda, typeof(meshParameters), typeof(permTableNamed), typeof(auxBuffersNamed)}(meshParameters, permTableNamed, auxBuffersNamed)

end

function fillPermTable!(perm::CUDA.CuArray, flags::CUDA.CuArray, N::Int)
    # Parallelized version using prefix sum
    # Compute cumulative sum (parallel)
    @views CUDA.cumsum!(perm[1:N], flags[1:N])
    # Zero out removed elements (where original flag was false)
    @views perm[1:N] .*= flags[1:N]    
    # Reset flags in parallel
    @views flags[1:N] .= true
    
    # Return count of surviving elements
    return Array(perm[N:N])[1]
end
