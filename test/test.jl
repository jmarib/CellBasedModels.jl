using BenchmarkTools
import InteractiveUtils: @code_warntype
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector
using Profile
using Profile.Allocs
using PProf
import SparseConnectivityTracer, ADTypes
using CairoMakie  # Use CairoMakie instead of Plots
using Random

    props = (
            a = Parameter(Float64, description="param a", dimensions=:M, defaultValue=1.0),
            b = Parameter(Int, description="param b", dimensions=:count, defaultValue=0),
        )

show(typeof(UnstructuredMesh(2; n=Node(props))))

# Test cell-linked neighbor algorithm with visualization
println("\n=== Cell-Linked Neighbor Algorithm Test ===")

# Create a 2D mesh with random particles
dims = 2
N = 100  # Number of particles
box_size = 10.0
cell_size = (5., 1.5)  # Cell size for neighbor search
cutoff = 2.0     # Neighbor search radius

# Create mesh and object
mesh = UnstructuredMesh(dims; n=Node(props))
obj = UnstructuredMeshObject(mesh, n=(N, N))

# Initialize random positions
Random.seed!(42)  # For reproducible results
obj.n.x[1:N] .= rand(N) * box_size
obj.n.y[1:N] .= rand(N) * box_size

# Set up cell-linked neighbors
box = [0.0 box_size; 0.0 box_size]
neighbors = NeighborsCellLinked(;box=box, cellSize=cell_size)

# Create full object with neighbors
obj_with_neighbors = UnstructuredMeshObject(
    mesh, 
    n=(N, N), 
    neighbors=neighbors
)

# Copy position data
obj_with_neighbors.n.x[1:N] .= obj.n.x[1:N]
obj_with_neighbors.n.y[1:N] .= obj.n.y[1:N]

# Update cell-linked structure
CellBasedModels.update!(obj_with_neighbors)

x_coords = obj_with_neighbors.n.x[1:N]
y_coords = obj_with_neighbors.n.y[1:N]
    
# Create the figure and axis
fig = Figure(size = (800, 800))
ax = Axis(fig[1, 1], 
          title = "Cell-Linked Neighbor Algorithm Demo",
          xlabel = "X Position", 
          ylabel = "Y Position",
          aspect = AxisAspect(1))
    
# Draw grid cells
grid_x, grid_y = obj_with_neighbors._neighbors.grid
for i in 0:grid_x
    lines!(ax, [i * cell_size[1], i * cell_size[1]], [0, box_size], 
           color = :lightgray, alpha = 0.5, linewidth = 1)
end
for j in 0:grid_y
    lines!(ax, [0, box_size], [j * cell_size[2], j * cell_size[2]], 
           color = :lightgray, alpha = 0.5, linewidth = 1)
end
    
# Plot all particles
scatter!(ax, x_coords, y_coords, 
         color = :lightblue, 
         markersize = 8, 
         alpha = 0.7,
         label = "All Particles")

 # Highlight test particle
test_particle = 1

# Highlight neighbors
neighbor_indices = iterateNeighbors(obj_with_neighbors, :n, x_coords[test_particle], y_coords[test_particle])
neighbor_x = [x_coords[i] for i in neighbor_indices]
neighbor_y = [y_coords[i] for i in neighbor_indices]
scatter!(ax, neighbor_x, neighbor_y, 
         color = :green, 
         markersize = 10, 
         label = "Neighbors")

scatter!(ax, [x_coords[test_particle]], [y_coords[test_particle]], 
    color = :red, 
    markersize = 12, 
    label = "Query Particle $(test_particle)")


save("cell_linked_neighbors_demo.png", fig)