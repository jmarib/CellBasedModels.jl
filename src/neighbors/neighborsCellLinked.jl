struct NeighborsCellLinked{D, P, UM, B, CS, G, C, CO, PT, CC} <: AbstractNeighbors 

    u::UM

    box::B
    cellSize::CS
    grid::G

    cell::C
    cellOffset::CO

    permTable::PT
    
    cellCounters::CC

end
Adapt.@adapt_structure NeighborsCellLinked

function NeighborsCellLinked(;box, cellSize)
    NeighborsCellLinked{Nothing, Nothing, Nothing, typeof(box), typeof(cellSize), Nothing, Nothing, Nothing, Nothing, Nothing}(
        nothing, 
        box, 
        cellSize,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing
    )
end

function NeighborsCellLinked(
    mesh,
    permTable,
    box,
    cellSize,
    grid,
    cell,
    cellOffset
)

    NeighborsCellLinked{
        length(cellSize),
        platform(),
        typeof(mesh),
        typeof(box),
        typeof(cellSize),
        typeof(grid),
        typeof(cell),
        typeof(cellOffset),
        typeof(permTable),
        Nothing
    }(
        mesh,
        box,
        cellSize,
        grid,
        cell,
        cellOffset,
        permTable,
        nothing
    )

end

function initNeighbors(
        dims, 
        neighbors::NeighborsCellLinked,
        meshParameters::NamedTuple
    )

    D = dims
    P = platform()

    if size(neighbors.box) != (D, 2)
        error("Box size mismatch. Expected size ($(D), 2), found size $(size(neighbors.box))")
    end
    box = neighbors.box

    cellSize = neighbors.cellSize
    if size(cellSize) == ()
        cellSize = fill(cellSize, D)
    elseif size(cellSize) != (D,)
        error("Cell size mismatch. Expected size ($(D),), found size $(size(cellSize))")        
    end

    grid = round.(Int, (box[:,2] - box[:,1]) ./ cellSize .+ 2)
    gridTuple = ntuple(i -> grid[i], D)  # Convert to NTuple for assignCell! compatibility
    gridSize = prod(grid)  # Total number of cells

    permTable = Dict()
    cell = Dict()
    cellOffset = Dict()
    cellCounters = Dict()  # Per-thread workspace arrays
    
    for (name, prop) in pairs(meshParameters)
        permTable[name] = zeros(Int, lengthCache(prop))
        cell[name] = zeros(Int, lengthCache(prop))
        cellOffset[name] = zeros(Int, gridSize+1)
        # Create per-thread workspace for cell counters (gridSize per thread)
        cellCounters[name] = [zeros(Int, gridSize) for _ in 1:Threads.nthreads()]
    end

    permTableNamed = NamedTuple{tuple(keys(permTable)...)}(values(permTable))
    cellNamed = NamedTuple{tuple(keys(cell)...)}(values(cell))
    cellOffsetNamed = NamedTuple{tuple(keys(cellOffset)...)}(values(cellOffset))
    cellCountersNamed = NamedTuple{tuple(keys(cellCounters)...)}(values(cellCounters))

    NeighborsCellLinked{
        D, P, 
        typeof(meshParameters), 
        typeof(box), typeof(cellSize), typeof(gridTuple), 
        typeof(cellNamed), typeof(cellOffsetNamed),
        typeof(permTableNamed), typeof(cellCountersNamed)
    }(meshParameters, box, cellSize, gridTuple, cellNamed, cellOffsetNamed, permTableNamed, cellCountersNamed)

end

function update!(mesh::UnstructuredMeshObject{D}) where {D}

    # Cell reassignment and permutation table filling
    for (name, prop) in pairs(mesh._p)
        N = lengthProperties(prop)
        
        # Step 1: Assign each particle to its cell
        assignCell!(mesh._neighbors.cell[name], N, prop, mesh._neighbors)
        
        # Step 2: Count particles in each cell (parallel, no allocations)
        countInCell!(mesh._neighbors.cellOffset[name], N, mesh._neighbors.cell[name], 
                    mesh._neighbors.cellCounters[name])
        
        # Step 3: Fill the permutation table (parallel, reuses cellCounters workspace)
        fillPermTable!(mesh._neighbors.permTable[name], mesh._neighbors.cellOffset[name], 
                      mesh._neighbors.cell[name], N, mesh._neighbors.cellCounters[name])
    end

    #Renaming TO DO
end

function assignCell(neighbors, x)
    clamp(floor(Int, (x - neighbors.box[1]) / neighbors.cellSize[1]), 0, neighbors.grid[1] - 1) + 1
end

function assignCell(neighbors, x, y)
    cellX = clamp(floor(Int, (x - neighbors.box[1,1]) / neighbors.cellSize[1]), 0, neighbors.grid[1] - 1)
    cellY = clamp(floor(Int, (y - neighbors.box[2,1]) / neighbors.cellSize[2]), 0, neighbors.grid[2] - 1)
    return cellY * neighbors.grid[1] + cellX + 1
end

function assignCell(neighbors, x, y, z)
    cellX = clamp(floor(Int, (x - neighbors.box[1,1]) / neighbors.cellSize[1]), 0, neighbors.grid[1] - 1)
    cellY = clamp(floor(Int, (y - neighbors.box[2,1]) / neighbors.cellSize[2]), 0, neighbors.grid[2] - 1)
    cellZ = clamp(floor(Int, (z - neighbors.box[3,1]) / neighbors.cellSize[3]), 0, neighbors.grid[3] - 1)
    return cellZ * neighbors.grid[1] * neighbors.grid[2] + cellY * neighbors.grid[1] + cellX + 1
end

function assignCell!(cellArray, N, prop, neighbors::NeighborsCellLinked{1})
    x = @views prop.x[1:N]
    cell = @views cellArray[1:N]
    cell .= clamp.(floor.(Int, (x .- neighbors.box[1]) ./ neighbors.cellSize[1]), 0, neighbors.grid[1] - 1) .+ 1
end

function assignCell!(cellArray, N, prop, neighbors::NeighborsCellLinked{2})
    x = @views prop.x[1:N]
    y = @views prop.y[1:N]
    cell = @views cellArray[1:N]
    cellX = clamp.(floor.(Int, (x .- neighbors.box[1,1]) ./ neighbors.cellSize[1]), 0, neighbors.grid[1] - 1)
    cellY = clamp.(floor.(Int, (y .- neighbors.box[2,1]) ./ neighbors.cellSize[2]), 0, neighbors.grid[2] - 1)
    cell .= cellY .* neighbors.grid[1] .+ cellX .+ 1
end

function assignCell!(cellArray, N, prop, neighbors::NeighborsCellLinked{3})
    x = @views prop.x[1:N]
    y = @views prop.y[1:N]
    z = @views prop.z[1:N]
    cell = @views cellArray[1:N]
    cellX = clamp.(floor.(Int, (x .- neighbors.box[1,1]) ./ neighbors.cellSize[1]), 0, neighbors.grid[1] - 1)
    cellY = clamp.(floor.(Int, (y .- neighbors.box[2,1]) ./ neighbors.cellSize[2]), 0, neighbors.grid[2] - 1)
    cellZ = clamp.(floor.(Int, (z .- neighbors.box[3,1]) ./ neighbors.cellSize[3]), 0, neighbors.grid[3] - 1)
    cell .= cellZ .* neighbors.grid[1] .* neighbors.grid[2] .+ cellY .* neighbors.grid[1] .+ cellX .+ 1
end

function countInCell!(cellOffset, N, cell, cellCounters)
    """
    Parallel version of cell counting using per-thread workspace arrays.
    No temporary allocations - uses pre-allocated cellCounters workspace.
    """
    
    nThreads = Threads.nthreads()
    nCells = length(cellOffset) - 1
    
    # Clear all thread-local counters (no allocation)
    Threads.@threads for tid in 1:nThreads
        fill!(cellCounters[tid], 0)
    end
    
    # Parallel counting phase - each thread counts its chunk
    Threads.@threads for i in 1:N
        tid = Threads.threadid()
        cellIdx = cell[i]
        @inbounds cellCounters[tid][cellIdx] += 1
    end
    
    # Reduction phase - sum all thread-local counts
    fill!(cellOffset, 0)
    for tid in 1:nThreads
        for cellIdx in 1:nCells
            @inbounds cellOffset[cellIdx] += cellCounters[tid][cellIdx]
        end
    end
    
    # Convert counts to cumulative offsets properly
    # We want cellOffset[i] to be the starting position for cell i (0-based)
    # So cellOffset[1] = 0, cellOffset[2] = count[1], cellOffset[3] = count[1] + count[2], etc.
    for i in nCells:-1:2
        cellOffset[i] = cellOffset[i-1]
    end
    cellOffset[1] = 0  # First cell starts at position 0
    cumsum!(cellOffset, cellOffset)  # Now cellOffset[i] = sum of counts for cells 1 to i-1
    
    return cellOffset
end

function fillPermTable!(permTable, cellOffset, cell, N, cellCounters)
    """
    Sequential version of permutation table filling to avoid race conditions.
    Creates a stable, deterministic ordering within each cell based on 
    the original particle indices (lower indices come first within each cell).
    
    permTable[sortedPos] = originalParticleIndex
    """
    
    nCells = length(cellOffset) - 1
    
    # cellOffset[i] now contains cumulative count (start position for cell i)
    # Initialize write heads to cell start positions (which are already correct in cellOffset)
    writeHeads = copy(cellOffset[1:nCells])  # Don't include the last element
    
    # Process particles in order to maintain stable sorting within cells
    for i in 1:N
        cellIdx = cell[i]
        
        # Get next available position in this cell and increment the write head
        sortedPos = writeHeads[cellIdx] + 1
        writeHeads[cellIdx] = sortedPos
        
        # Store original particle index at the sorted position
        if sortedPos <= length(permTable)
            permTable[sortedPos] = i
        else
            error("sortedPos ($sortedPos) exceeds permTable length ($(length(permTable))). Cell $cellIdx, writeHead was $(writeHeads[cellIdx]-1)")
        end
    end
    
    return permTable
end

function iterateNeighbors(mesh::UnstructuredMeshObject{P, 1, S, DT, NN}, symbol::Symbol, x) where {P, S, DT, NN<:NeighborsCellLinked}
    """
    Get all particles in the 3 neighboring cells in 1D.
    """
    
    cell = assignCell(mesh._neighbors, x)
    cells = [cell - 1, cell, cell + 1]

    allNeighbors = Int[]
    for cellIdx in cells
        if cellIdx >= 1 && cellIdx <= mesh._neighbors.grid[1]
            # Get the range of positions for this cell
            startPos = mesh._neighbors.cellOffset[symbol][cellIdx] + 1
            endPos = mesh._neighbors.cellOffset[symbol][cellIdx + 1]
            
            # Get the actual particle indices from the permutation table
            for pos in startPos:endPos
                if pos <= length(mesh._neighbors.permTable[symbol])
                    particleIdx = mesh._neighbors.permTable[symbol][pos]
                    push!(allNeighbors, particleIdx)
                end
            end
        end
    end

    return allNeighbors
end

function iterateNeighbors(mesh::UnstructuredMeshObject{P, 2, S, DT, NN}, symbol::Symbol, x, y) where {P, S, DT, NN<:NeighborsCellLinked}
    """
    Get all particles in the 9 neighboring cells in 2D.
    """

    cell = assignCell(mesh._neighbors, x, y)
    gridX, gridY = mesh._neighbors.grid
    
    # Generate all 9 neighboring cell indices correctly
    allNeighbors = Int[]
    
    # Convert linear cell index to 2D coordinates
    cellX = (cell - 1) % gridX
    cellY = div(cell - 1, gridX)
    
    # Check all 3x3 neighborhood
    for dy in -1:1, dx in -1:1
        neighborX = cellX + dx
        neighborY = cellY + dy
        
        # Check bounds
        if 0 <= neighborX < gridX && 0 <= neighborY < gridY
            neighborCell = neighborY * gridX + neighborX + 1
            
            # Get particles in this cell
            startPos = mesh._neighbors.cellOffset[symbol][neighborCell] + 1
            endPos = mesh._neighbors.cellOffset[symbol][neighborCell + 1]
            
            # Get the actual particle indices from the permutation table
            for pos in startPos:endPos
                if pos <= length(mesh._neighbors.permTable[symbol])
                    particleIdx = mesh._neighbors.permTable[symbol][pos]
                    push!(allNeighbors, particleIdx)
                end
            end
        end
    end

    return allNeighbors
end

function iterateNeighbors(mesh::UnstructuredMeshObject{P, 3, S, DT, NN}, symbol::Symbol, x, y, z) where {P, S, DT, NN<:NeighborsCellLinked}
    """
    Get all particles in the 27 neighboring cells in 3D.
    """

    cell = assignCell(mesh._neighbors, x, y, z)
    gridX, gridY, gridZ = mesh._neighbors.grid
    
    # Generate all 27 neighboring cell indices correctly
    allNeighbors = Int[]
    
    # Convert linear cell index to 3D coordinates
    cellX = (cell - 1) % gridX
    temp = div(cell - 1, gridX)
    cellY = temp % gridY
    cellZ = div(temp, gridY)
    
    # Check all 3x3x3 neighborhood
    for dz in -1:1, dy in -1:1, dx in -1:1
        neighborX = cellX + dx
        neighborY = cellY + dy
        neighborZ = cellZ + dz
        
        # Check bounds
        if 0 <= neighborX < gridX && 0 <= neighborY < gridY && 0 <= neighborZ < gridZ
            neighborCell = neighborZ * gridX * gridY + neighborY * gridX + neighborX + 1
            
            # Get particles in this cell
            startPos = mesh._neighbors.cellOffset[symbol][neighborCell] + 1
            endPos = mesh._neighbors.cellOffset[symbol][neighborCell + 1]
            
            # Get the actual particle indices from the permutation table
            for pos in startPos:endPos
                if pos <= length(mesh._neighbors.permTable[symbol])
                    particleIdx = mesh._neighbors.permTable[symbol][pos]
                    push!(allNeighbors, particleIdx)
                end
            end
        end
    end

    return allNeighbors
end