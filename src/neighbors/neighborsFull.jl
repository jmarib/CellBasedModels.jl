struct NeighborsFull{P, UM, PT} <: AbstractNeighbors 

    u::UM
    permTable::PT

end
Adapt.@adapt_structure NeighborsFull

NeighborsFull() = NeighborsFull(nothing, nothing)

function NeighborsFull(
    mesh,
    neighbors
)

    P = platform()
    NeighborsFull{P, typeof(mesh), typeof(neighbors)}(mesh, neighbors)

end

function initNeighbors(
        ::Int,
        ::NeighborsFull,
        meshParameters::NamedTuple
    )

    permTable = Dict()
    for (name, prop) in pairs(meshParameters)
        permTable[name] = zeros(Int, lengthCache(prop))
    end

    permTableNamed = NamedTuple{tuple(keys(permTable)...)}(values(permTable))

    NeighborsFull{platform(), typeof(meshParameters), typeof(permTableNamed)}(meshParameters, permTableNamed)

end

function update!(mesh::UnstructuredMeshObject{P, D, S, DT, NN, PAR}) where {P, D, S, DT, NN<:NeighborsFull, PAR}

    # Compaction
    for (name, prop) in pairs(mesh._p)
        N = lengthProperties(prop)
        @views cumsum!(mesh._neighbors.permTable[name][1:N], prop._FlagsSurvived[1:N])
        @views prop._FlagsSurvived[1:N] .= true
    end

    #Renaming TO DO
end

function iterateNeighbors(mesh::UnstructuredMeshObject{P, D, S, DT, NN}, symbol::Symbol, index::Int) where {P, D, S, DT, NN<:NeighborsFull}

    return 1:lengthProperties(mesh._p[symbol])

end

function getNeighbors(mesh::UnstructuredMeshObject{P, D, S, DT, NN}, symbol::Symbol, index::Int) where {P, D, S, DT, NN<:NeighborsFull}

    return collect(iterateNeighbors(mesh, symbol, index))

end


