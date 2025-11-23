struct CellLinked{D, AC, AO, NC, NO, EC, EO, FC, FO, VC, VO, Bm, BM, CS} <: AbstractNeighbors{D}

    aCell::AC
    aOffset::AO

    nCell::NC
    nOffset::NO

    eCell::EC
    eOffset::EO

    fCell::FC
    fOffset::FO

    vCell::VC
    vOffset::VO

    _boundsMin::Bm
    _boundsMax::BM
    _cellsSize::CS
    _NCells::Int

end

function CellLinked(
        cellSize::Union{Real, Tuple},
        boundsMin::NTuple{D, <:Real},
        boundsMax::NTuple{D, <:Real},
    ) where {D}

    if cellSize isa Real
        cellSize = ntuple(_ -> cellSize, D)
    end

    for (mMin, mMax) in zip(boundsMin, boundsMax)
        if mMax <= mMin
            error("boundsMax must be greater than boundsMin for all dimensions.")
        end
    end

    _NCells = prod(ntuple(d -> ceil(Int, (boundsMax[d] - boundsMin[d]) / cellSize[d])+2, D))

    CellLinked{
        D,
        Nothing, Nothing,
        Nothing, Nothing, 
        Nothing, Nothing, 
        Nothing, Nothing, 
        Nothing, Nothing,  
        typeof(cellSize), typeof(boundsMin), typeof(boundsMax)}(
        nothing, nothing,
        nothing, nothing,
        nothing, nothing,
        nothing, nothing,
        nothing, nothing,
        boundsMin, boundsMax, cellSize, _NCells
        )

end

function CellLinked(
    mesh::UnstructuredMeshObject{P, D},
    cell::CellLinked{D}
) where {P, D}

 
    aCell = mesh.a !== nothing ? zeros(Int, lengthCache(mesh.a)) : nothing
    aOffset = mesh.a !== nothing ? zeros(Int, cell._NCells+1) : nothing

    nCell = mesh.n !== nothing ? zeros(Int, lengthCache(mesh.n)) : nothing
    nOffset = mesh.n !== nothing ? zeros(Int, cell._NCells+1) : nothing

    eCell = mesh.e !== nothing ? zeros(Int, lengthCache(mesh.e)) : nothing
    eOffset = mesh.e !== nothing ? zeros(Int, cell._NCells+1) : nothing

    fCell = mesh.f !== nothing ? zeros(Int, lengthCache(mesh.f)) : nothing
    fOffset = mesh.f !== nothing ? zeros(Int, cell._NCells+1) : nothing

    vCell = mesh.v !== nothing ? zeros(Int, lengthCache(mesh.v)) : nothing
    vOffset = mesh.v !== nothing ? zeros(Int, cell._NCells+1) : nothing

    return CellLinked{
        D,
        typeof(aCell), typeof(aOffset),
        typeof(nCell), typeof(nOffset),
        typeof(eCell), typeof(eOffset),
        typeof(fCell), typeof(fOffset),
        typeof(vCell), typeof(vOffset),
        typeof(cell.boundsMin), typeof(cell.boundsMax), typeof(cell.cellsSize)
    }(
        aCell, aOffset,
        nCell, nOffset,
        eCell, eOffset,
        fCell, fOffset,
        vCell, vOffset,
        cell.boundsMin, cell.boundsMax, cell.cellsSize, cell._NCells
    )

end