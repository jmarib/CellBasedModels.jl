struct NeighborsCellLinked{D, P, UM, B, CS, G, C, CO, PT, CC, PE} <: AbstractNeighbors 

    u::UM

    box::B
    cellSize::CS
    grid::G
    periodic::PE

    cell::C
    cellOffset::CO

    permTable::PT
    
    cellCounters::CC

end
Adapt.@adapt_structure NeighborsCellLinked

function NeighborsCellLinked(;box, cellSize, periodic=nothing)
    NeighborsCellLinked{Nothing, Nothing, Nothing, typeof(box), typeof(cellSize), Nothing, Nothing, Nothing, Nothing, Nothing, typeof(periodic)}(
        nothing, 
        box, 
        cellSize,
        nothing,
        periodic,
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
    periodic,
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
        Nothing,
        typeof(periodic)
    }(
        mesh,
        box,
        cellSize,
        grid,
        periodic,
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
    if cellSize isa Number
        # Scalar cellSize: replicate for all dimensions
        cellSize = fill(cellSize, D)
    elseif cellSize isa Tuple
        # Tuple cellSize: convert to vector and check length
        cellSize = collect(cellSize)
        if length(cellSize) != D
            error("Cell size mismatch. Expected length $(D), found length $(length(cellSize))")
        end
    elseif cellSize isa AbstractVector
        # Vector cellSize: check length
        if length(cellSize) != D
            error("Cell size mismatch. Expected length $(D), found length $(length(cellSize))")        
        end
    else
        error("Cell size must be a Number, Tuple, or Vector. Found type $(typeof(cellSize))")
    end

    # Handle periodic boundary parameter
    periodic = neighbors.periodic
    if periodic === nothing
        # Default: no periodic boundaries
        periodic = ntuple(i -> false, D)
    elseif periodic isa Bool
        # Single bool: apply to all dimensions
        periodic = ntuple(i -> periodic, D)
    elseif periodic isa Tuple
        # Tuple: check length
        if length(periodic) != D
            error("Periodic tuple length mismatch. Expected length $(D), found length $(length(periodic))")
        end
        periodic = ntuple(i -> periodic[i], D)
    elseif periodic isa AbstractVector
        # Vector: check length and convert to tuple
        if length(periodic) != D
            error("Periodic vector length mismatch. Expected length $(D), found length $(length(periodic))")
        end
        periodic = ntuple(i -> periodic[i], D)
    else
        error("Periodic parameter must be Nothing, Bool, Tuple, or Vector. Found type $(typeof(periodic))")
    end

    # Calculate grid size: no padding for periodic dimensions, padding for non-periodic
    grid = round.(Int, (box[:,2] - box[:,1]) ./ cellSize)
    for i in 1:D
        if !periodic[i]
            grid[i] += 2  # Add padding for non-periodic boundaries
        end
    end
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
        typeof(permTableNamed), typeof(cellCountersNamed), typeof(periodic)
    }(meshParameters, box, cellSize, gridTuple, periodic, cellNamed, cellOffsetNamed, permTableNamed, cellCountersNamed)

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
    # Calculate raw cell indices
    rawCellX = floor(Int, (x - neighbors.box[1,1]) / neighbors.cellSize[1])
    rawCellY = floor(Int, (y - neighbors.box[2,1]) / neighbors.cellSize[2])
    
    # Apply boundary conditions
    if neighbors.periodic[1]
        cellX = rawCellX % neighbors.grid[1]
    else
        cellX = clamp(rawCellX, 0, neighbors.grid[1] - 1)
    end
    
    if neighbors.periodic[2]
        cellY = rawCellY % neighbors.grid[2]
    else
        cellY = clamp(rawCellY, 0, neighbors.grid[2] - 1)
    end
    
    return cellY * neighbors.grid[1] + cellX + 1
end

function assignCell(neighbors, x, y, z)
    # Calculate raw cell indices
    rawCellX = floor(Int, (x - neighbors.box[1,1]) / neighbors.cellSize[1])
    rawCellY = floor(Int, (y - neighbors.box[2,1]) / neighbors.cellSize[2])
    rawCellZ = floor(Int, (z - neighbors.box[3,1]) / neighbors.cellSize[3])
    
    # Apply boundary conditions
    if neighbors.periodic[1]
        cellX = rawCellX % neighbors.grid[1]
    else
        cellX = clamp(rawCellX, 0, neighbors.grid[1] - 1)
    end
    
    if neighbors.periodic[2]
        cellY = rawCellY % neighbors.grid[2]
    else
        cellY = clamp(rawCellY, 0, neighbors.grid[2] - 1)
    end
    
    if neighbors.periodic[3]
        cellZ = rawCellZ % neighbors.grid[3]
    else
        cellZ = clamp(rawCellZ, 0, neighbors.grid[3] - 1)
    end
    
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
    
    # Calculate cell indices
    rawCellX = floor.(Int, (x .- neighbors.box[1,1]) ./ neighbors.cellSize[1])
    rawCellY = floor.(Int, (y .- neighbors.box[2,1]) ./ neighbors.cellSize[2])
    
    # Apply boundary conditions
    if neighbors.periodic[1]
        cellX = rawCellX .% neighbors.grid[1]
    else
        cellX = clamp.(rawCellX, 0, neighbors.grid[1] - 1)
    end
    
    if neighbors.periodic[2]
        cellY = rawCellY .% neighbors.grid[2]
    else
        cellY = clamp.(rawCellY, 0, neighbors.grid[2] - 1)
    end
    
    cell .= cellY .* neighbors.grid[1] .+ cellX .+ 1
end

function assignCell!(cellArray, N, prop, neighbors::NeighborsCellLinked{3})
    x = @views prop.x[1:N]
    y = @views prop.y[1:N]
    z = @views prop.z[1:N]
    cell = @views cellArray[1:N]
    
    # Calculate cell indices
    rawCellX = floor.(Int, (x .- neighbors.box[1,1]) ./ neighbors.cellSize[1])
    rawCellY = floor.(Int, (y .- neighbors.box[2,1]) ./ neighbors.cellSize[2])
    rawCellZ = floor.(Int, (z .- neighbors.box[3,1]) ./ neighbors.cellSize[3])
    
    # Apply boundary conditions
    if neighbors.periodic[1]
        cellX = rawCellX .% neighbors.grid[1]
    else
        cellX = clamp.(rawCellX, 0, neighbors.grid[1] - 1)
    end
    
    if neighbors.periodic[2]
        cellY = rawCellY .% neighbors.grid[2]
    else
        cellY = clamp.(rawCellY, 0, neighbors.grid[2] - 1)
    end
    
    if neighbors.periodic[3]
        cellZ = rawCellZ .% neighbors.grid[3]
    else
        cellZ = clamp.(rawCellZ, 0, neighbors.grid[3] - 1)
    end
    
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
    # cellOffset[i] should be the starting position for cell i (0-based)
    # For example: if cell 1 has 2 particles, cell 2 has 1 particle:
    # cellOffset[1] = 0, cellOffset[2] = 2, cellOffset[3] = 3
    
    temp_offset = 0
    for i in 1:nCells
        count = cellOffset[i]
        cellOffset[i] = temp_offset
        temp_offset += count
    end
    cellOffset[nCells + 1] = temp_offset  # Total count
    
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

# Iterator types for non-allocating neighbor iteration
struct NeighborIterator1D{NN, PT, CO}
    neighbors::NN
    permTable::PT
    cellOffset::CO
    cells::NTuple{3, Int}  # The 3 neighboring cells
    gridSize::Int
    periodic::Bool
end

struct NeighborIterator2D{NN, PT, CO}
    neighbors::NN
    permTable::PT
    cellOffset::CO
    centerCellX::Int
    centerCellY::Int
    gridX::Int
    gridY::Int
    periodicX::Bool
    periodicY::Bool
end

struct NeighborIterator3D{NN, PT, CO}
    neighbors::NN
    permTable::PT
    cellOffset::CO
    centerCellX::Int
    centerCellY::Int
    centerCellZ::Int
    gridX::Int
    gridY::Int
    gridZ::Int
    periodicX::Bool
    periodicY::Bool
    periodicZ::Bool
end

# Iterator state for tracking position
struct NeighborIteratorState
    cellIdx::Int      # Current cell being iterated
    particleIdx::Int  # Current particle within the cell
    dx::Int           # Offset in x direction (-1, 0, 1)
    dy::Int           # Offset in y direction (-1, 0, 1) 
    dz::Int           # Offset in z direction (-1, 0, 1)
end

# Iterator protocol implementations
Base.IteratorSize(::Type{<:NeighborIterator1D}) = Base.SizeUnknown()
Base.IteratorSize(::Type{<:NeighborIterator2D}) = Base.SizeUnknown()
Base.IteratorSize(::Type{<:NeighborIterator3D}) = Base.SizeUnknown()

Base.eltype(::Type{<:NeighborIterator1D}) = Int
Base.eltype(::Type{<:NeighborIterator2D}) = Int
Base.eltype(::Type{<:NeighborIterator3D}) = Int

# 1D Iterator implementation
function Base.iterate(iter::NeighborIterator1D)
    # Start with the first cell
    for (cellIdx, cell) in enumerate(iter.cells)
        if cell >= 1 && cell <= iter.gridSize
            startPos = iter.cellOffset[cell] + 1
            endPos = iter.cellOffset[cell + 1]
            if startPos <= endPos
                particleIdx = iter.permTable[startPos]
                state = NeighborIteratorState(cellIdx, startPos, 0, 0, 0)
                return (particleIdx, state)
            end
        end
    end
    return nothing
end

function Base.iterate(iter::NeighborIterator1D, state::NeighborIteratorState)
    cellIdx = state.cellIdx
    particlePos = state.particleIdx + 1
    
    # Try next particle in current cell
    if cellIdx <= length(iter.cells)
        cell = iter.cells[cellIdx]
        if cell >= 1 && cell <= iter.gridSize
            endPos = iter.cellOffset[cell + 1]
            if particlePos <= endPos
                particleIdx = iter.permTable[particlePos]
                newState = NeighborIteratorState(cellIdx, particlePos, 0, 0, 0)
                return (particleIdx, newState)
            end
        end
    end
    
    # Move to next cell
    for nextCellIdx in (cellIdx + 1):length(iter.cells)
        cell = iter.cells[nextCellIdx]
        if cell >= 1 && cell <= iter.gridSize
            startPos = iter.cellOffset[cell] + 1
            endPos = iter.cellOffset[cell + 1]
            if startPos <= endPos
                particleIdx = iter.permTable[startPos]
                newState = NeighborIteratorState(nextCellIdx, startPos, 0, 0, 0)
                return (particleIdx, newState)
            end
        end
    end
    
    return nothing
end

# 2D Iterator implementation
function Base.iterate(iter::NeighborIterator2D)
    # Start with dx=-1, dy=-1
    dx, dy = -1, -1
    neighborX = iter.centerCellX + dx
    neighborY = iter.centerCellY + dy
    
    # Apply periodic boundary conditions or bounds checking
    validX = true
    validY = true
    
    if iter.periodicX
        neighborX = ((neighborX % iter.gridX) + iter.gridX) % iter.gridX  # Proper modulo for negative numbers
    else
        validX = (0 <= neighborX < iter.gridX)
    end
    
    if iter.periodicY
        neighborY = ((neighborY % iter.gridY) + iter.gridY) % iter.gridY  # Proper modulo for negative numbers
    else
        validY = (0 <= neighborY < iter.gridY)
    end
    
    if validX && validY
        neighborCell = neighborY * iter.gridX + neighborX + 1
        startPos = iter.cellOffset[neighborCell] + 1
        endPos = iter.cellOffset[neighborCell + 1]
        if startPos <= endPos
            particleIdx = iter.permTable[startPos]
            state = NeighborIteratorState(neighborCell, startPos, dx, dy, 0)
            return (particleIdx, state)
        end
    end
    
    # If first cell is empty or invalid, find next valid cell/particle
    return iterate_next_2d(iter, dx, dy, 0)
end

function Base.iterate(iter::NeighborIterator2D, state::NeighborIteratorState)
    # Try next particle in current cell
    particlePos = state.particleIdx + 1
    endPos = iter.cellOffset[state.cellIdx + 1]
    if particlePos <= endPos
        particleIdx = iter.permTable[particlePos]
        newState = NeighborIteratorState(state.cellIdx, particlePos, state.dx, state.dy, 0)
        return (particleIdx, newState)
    end
    
    # Move to next cell
    return iterate_next_2d(iter, state.dx, state.dy, particlePos)
end

function iterate_next_2d(iter::NeighborIterator2D, start_dx::Int, start_dy::Int, start_pos::Int)
    # Continue from current position
    for dy in start_dy:1, dx in (dy == start_dy ? start_dx : -1):1
        if dy == start_dy && dx == start_dx && start_pos > 0
            continue  # Skip the current position we just finished
        end
        
        neighborX = iter.centerCellX + dx
        neighborY = iter.centerCellY + dy
        
        # Apply periodic boundary conditions or skip if out of bounds
        validX = true
        validY = true
        
        if iter.periodicX
            neighborX = ((neighborX % iter.gridX) + iter.gridX) % iter.gridX  # Proper modulo for negative numbers
        else
            validX = (0 <= neighborX < iter.gridX)
        end
        
        if iter.periodicY
            neighborY = ((neighborY % iter.gridY) + iter.gridY) % iter.gridY  # Proper modulo for negative numbers
        else
            validY = (0 <= neighborY < iter.gridY)
        end
        
        if validX && validY
            neighborCell = neighborY * iter.gridX + neighborX + 1
            startPos = iter.cellOffset[neighborCell] + 1
            endPos = iter.cellOffset[neighborCell + 1]
            if startPos <= endPos
                particleIdx = iter.permTable[startPos]
                state = NeighborIteratorState(neighborCell, startPos, dx, dy, 0)
                return (particleIdx, state)
            end
        end
    end
    return nothing
end

# 3D Iterator implementation
function Base.iterate(iter::NeighborIterator3D)
    # Start with dx=-1, dy=-1, dz=-1
    dx, dy, dz = -1, -1, -1
    neighborX = iter.centerCellX + dx
    neighborY = iter.centerCellY + dy
    neighborZ = iter.centerCellZ + dz
    
    # Apply periodic boundary conditions or bounds checking
    validX = true
    validY = true
    validZ = true
    
    if iter.periodicX
        neighborX = ((neighborX % iter.gridX) + iter.gridX) % iter.gridX  # Proper modulo for negative numbers
    else
        validX = (0 <= neighborX < iter.gridX)
    end
    
    if iter.periodicY
        neighborY = ((neighborY % iter.gridY) + iter.gridY) % iter.gridY  # Proper modulo for negative numbers
    else
        validY = (0 <= neighborY < iter.gridY)
    end
    
    if iter.periodicZ
        neighborZ = ((neighborZ % iter.gridZ) + iter.gridZ) % iter.gridZ  # Proper modulo for negative numbers
    else
        validZ = (0 <= neighborZ < iter.gridZ)
    end
    
    if validX && validY && validZ
        neighborCell = neighborZ * iter.gridX * iter.gridY + neighborY * iter.gridX + neighborX + 1
        startPos = iter.cellOffset[neighborCell] + 1
        endPos = iter.cellOffset[neighborCell + 1]
        if startPos <= endPos
            particleIdx = iter.permTable[startPos]
            state = NeighborIteratorState(neighborCell, startPos, dx, dy, dz)
            return (particleIdx, state)
        end
    end
    
    # If first cell is empty or invalid, find next valid cell/particle
    return iterate_next_3d(iter, dx, dy, dz, 0)
end

function Base.iterate(iter::NeighborIterator3D, state::NeighborIteratorState)
    # Try next particle in current cell
    particlePos = state.particleIdx + 1
    endPos = iter.cellOffset[state.cellIdx + 1]
    if particlePos <= endPos
        particleIdx = iter.permTable[particlePos]
        newState = NeighborIteratorState(state.cellIdx, particlePos, state.dx, state.dy, state.dz)
        return (particleIdx, newState)
    end
    
    # Move to next cell
    return iterate_next_3d(iter, state.dx, state.dy, state.dz, particlePos)
end

function iterate_next_3d(iter::NeighborIterator3D, start_dx::Int, start_dy::Int, start_dz::Int, start_pos::Int)
    # Continue from current position
    for dz in start_dz:1, dy in (dz == start_dz ? start_dy : -1):1, dx in (dz == start_dz && dy == start_dy ? start_dx : -1):1
        if dz == start_dz && dy == start_dy && dx == start_dx && start_pos > 0
            continue  # Skip the current position we just finished
        end
        
        neighborX = iter.centerCellX + dx
        neighborY = iter.centerCellY + dy
        neighborZ = iter.centerCellZ + dz
        
        # Apply periodic boundary conditions or bounds checking
        validX = true
        validY = true
        validZ = true
        
        if iter.periodicX
            neighborX = ((neighborX % iter.gridX) + iter.gridX) % iter.gridX  # Proper modulo for negative numbers
        else
            validX = (0 <= neighborX < iter.gridX)
        end
        
        if iter.periodicY
            neighborY = ((neighborY % iter.gridY) + iter.gridY) % iter.gridY  # Proper modulo for negative numbers
        else
            validY = (0 <= neighborY < iter.gridY)
        end
        
        if iter.periodicZ
            neighborZ = ((neighborZ % iter.gridZ) + iter.gridZ) % iter.gridZ  # Proper modulo for negative numbers
        else
            validZ = (0 <= neighborZ < iter.gridZ)
        end
        
        if validX && validY && validZ
            neighborCell = neighborZ * iter.gridX * iter.gridY + neighborY * iter.gridX + neighborX + 1
            startPos = iter.cellOffset[neighborCell] + 1
            endPos = iter.cellOffset[neighborCell + 1]
            if startPos <= endPos
                particleIdx = iter.permTable[startPos]
                state = NeighborIteratorState(neighborCell, startPos, dx, dy, dz)
                return (particleIdx, state)
            end
        end
    end
    return nothing
end

# Constructor functions for the iterators
function iterateNeighbors(mesh::UnstructuredMeshObject{P, 1, S, DT, NN}, symbol::Symbol, x) where {P, S, DT, NN<:NeighborsCellLinked}
    """
    Create a non-allocating iterator for particles in the 3 neighboring cells in 1D.
    """
    
    cell = assignCell(mesh._neighbors, x)
    cells = (cell - 1, cell, cell + 1)

    return NeighborIterator1D(
        mesh._neighbors,
        mesh._neighbors.permTable[symbol],
        mesh._neighbors.cellOffset[symbol],
        cells,
        mesh._neighbors.grid[1],
        mesh._neighbors.periodic[1]
    )
end

function iterateNeighbors(mesh::UnstructuredMeshObject{P, 2, S, DT, NN}, symbol::Symbol, x, y) where {P, S, DT, NN<:NeighborsCellLinked}
    """
    Create a non-allocating iterator for particles in the 9 neighboring cells in 2D.
    """

    cell = assignCell(mesh._neighbors, x, y)
    gridX, gridY = mesh._neighbors.grid
    
    # Convert linear cell index to 2D coordinates
    cellX = (cell - 1) % gridX
    cellY = div(cell - 1, gridX)

    return NeighborIterator2D(
        mesh._neighbors,
        mesh._neighbors.permTable[symbol],
        mesh._neighbors.cellOffset[symbol],
        cellX,
        cellY,
        gridX,
        gridY,
        mesh._neighbors.periodic[1],
        mesh._neighbors.periodic[2]
    )
end

function iterateNeighbors(mesh::UnstructuredMeshObject{P, 3, S, DT, NN}, symbol::Symbol, x, y, z) where {P, S, DT, NN<:NeighborsCellLinked}
    """
    Create a non-allocating iterator for particles in the 27 neighboring cells in 3D.
    """

    cell = assignCell(mesh._neighbors, x, y, z)
    gridX, gridY, gridZ = mesh._neighbors.grid
    
    # Convert linear cell index to 3D coordinates
    cellX = (cell - 1) % gridX
    temp = div(cell - 1, gridX)
    cellY = temp % gridY
    cellZ = div(temp, gridY)

    return NeighborIterator3D(
        mesh._neighbors,
        mesh._neighbors.permTable[symbol],
        mesh._neighbors.cellOffset[symbol],
        cellX,
        cellY,
        cellZ,
        gridX,
        gridY,
        gridZ,
        mesh._neighbors.periodic[1],
        mesh._neighbors.periodic[2],
        mesh._neighbors.periodic[3]
    )
end

# Convenience functions for backward compatibility
"""
    collectNeighbors(iterator) -> Vector{Int}
    
Convert a neighbor iterator to a vector of particle indices.
Use this when you need the old vector-based API.
"""
collectNeighbors(iterator) = collect(iterator)

"""
    collectNeighbors(mesh, symbol, coordinates...) -> Vector{Int}
    
Get all neighboring particles as a vector (allocating version).
"""
collectNeighbors(mesh, symbol, coords...) = collect(iterateNeighbors(mesh, symbol, coords...))