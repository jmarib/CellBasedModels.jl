struct NeighborsFull <: AbstractNeighbors 
    neighbors::Vector{Int}
end

computeNeighbors!(neigh::NeighborsFull, comm::AbstractCommunity) = nothing

