abstract type AbstractNeighbors end

# Specialized no-op computeNeighbors! for Nothing
# computeNeighbors!(::Nothing, comm::AbstractCommunity) = nothing