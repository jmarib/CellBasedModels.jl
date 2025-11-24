function plotCellLinkedGrid(
         neighbors::UnstructuredMesh{D, PN},
         positions::AbstractMatrix{<:Real};
         kwargs...
     )
 
     dims = size(positions, 1)
     if dims == 2
         return plotCellLinkedNeighbors2D(neighbors, positions; kwargs...)
     elseif dims == 3
         return plotCellLinkedNeighbors3D(neighbors, positions; kwargs...)
     else
         error("Unsupported dimension $dims for plotting.")
     end
 end