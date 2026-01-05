"""
    Mesh generation functions for various distribution patterns.
    
Returns: (nodes, edges, faces, volumes, agent_id)
where agent_id identifies the largest topological element each node belongs to.
"""

######################################################################################################
# 2D CUBIC PACKAGING
######################################################################################################

function cubic2D(radius::Float64, box::Tuple{Float64, Float64}, 
                 mask_distribution::Function)
    """
    2D cubic packaging of circular agents.
    
    Args:
        nagents: Number of agents
        radius: Radius of each agent
        box: (width, height) of the domain
        mask_distribution: Function that returns true if position is valid
    
    Returns:
        (nodes=N,), (nodes=(x,y),)
    """
    
    xvec = Vector{Float64}[]
    yvec = Vector{Float64}[]
    agent_id = Int[]
    agent_count = 0
    
    width, height = box
    spacing = 2 * radius
    
    nx = ceil(Int, width / spacing)
    ny = ceil(Int, height / spacing)
    
    for i in 1:nx
        for j in 1:ny
            x = (i - 0.5) * spacing
            y = (j - 0.5) * spacing
            
            if x <= width && y <= height && mask_distribution(x, y)
                agent_count += 1
                push!(xvec, x)
                push!(yvec, y)
                push!(agent_id, agent_count)
            end
        end
    end
    
    return (nodes=length(xvec),), (nodes=(x=xvec, y=yvec),)
end

######################################################################################################
# 3D CUBIC PACKAGING
######################################################################################################

function cubic3D(radius::Float64, box::Tuple{Float64, Float64, Float64},
                 mask_distribution::Function)
    """
    3D cubic packaging of spherical agents.
    
    Args:
        radius: Radius of each agent
        box: (width, height, depth) of the domain
        mask_distribution: Function that returns true if position is valid
    
    Returns:
        (nodes=N,), (nodes=(x,y,z),)
    """
    
    xvec = Vector{Float64}[]
    yvec = Vector{Float64}[]
    zvec = Vector{Float64}[]
    
    width, height, depth = box
    spacing = 2 * radius
    
    nx = ceil(Int, width / spacing)
    ny = ceil(Int, height / spacing)
    nz = ceil(Int, depth / spacing)
    
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz
                x = (i - 0.5) * spacing
                y = (j - 0.5) * spacing
                z = (k - 0.5) * spacing
                
                if x <= width && y <= height && z <= depth && mask_distribution(x, y, z)
                    push!(xvec, x)
                    push!(yvec, y)
                    push!(zvec, z)
                end
            end
        end
    end
    
    return (nodes=length(xvec),), (nodes=(x=xvec, y=yvec, z=zvec),)
end

######################################################################################################
# 2D HEXAGONAL PACKAGING
######################################################################################################

function hexagonal2D(radius::Float64, box::Tuple{Float64, Float64},
                     mask_distribution::Function)
    """
    2D hexagonal close packing of circular agents.
    
    Args:
        radius: Radius of each agent
        box: (width, height) of the domain
        mask_distribution: Function that returns true if position is valid
    
    Returns:
        (nodes=N,), (nodes=(x,y),)
    """
    
    xvec = Vector{Float64}[]
    yvec = Vector{Float64}[]
    
    width, height = box
    spacing = 2 * radius
    dy = spacing * sqrt(3) / 2
    
    nx = ceil(Int, width / spacing)
    ny = ceil(Int, height / dy)
    
    for i in 1:nx
        for j in 1:ny
            x = (i - 0.5) * spacing + (mod(j-1, 2) * spacing / 2)
            y = (j - 0.5) * dy
            
            if x <= width && y <= height && mask_distribution(x, y)
                push!(xvec, x)
                push!(yvec, y)
            end
        end
    end
    
    return (nodes=length(xvec),), (nodes=(x=xvec, y=yvec),)
end

######################################################################################################
# 3D HEXAGONAL PACKAGING
######################################################################################################

function hexagonal3D(radius::Float64, box::Tuple{Float64, Float64, Float64},
                     mask_distribution::Function)
    """
    3D hexagonal close packing (face-centered cubic) of spherical agents.
    
    Args:
        radius: Radius of each agent
        box: (width, height, depth) of the domain
        mask_distribution: Function that returns true if position is valid
    
    Returns:
        (nodes=N,), (nodes=(x,y,z),)
    """
    
    xvec = Vector{Float64}[]
    yvec = Vector{Float64}[]
    zvec = Vector{Float64}[]
    
    width, height, depth = box
    spacing = 2 * radius
    dy = spacing * sqrt(3) / 2
    dz = spacing * sqrt(2/3)
    
    nx = ceil(Int, width / spacing)
    ny = ceil(Int, height / dy)
    nz = ceil(Int, depth / dz)
    
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz
                x = (i - 0.5) * spacing + (mod(j-1, 2) * spacing / 2)
                y = (j - 0.5) * dy
                z = (k - 0.5) * dz + (mod(j-1, 2) * dz / 2)
                
                if x <= width && y <= height && z <= depth && mask_distribution(x, y, z)
                    push!(xvec, x)
                    push!(yvec, y)
                    push!(zvec, z)
                end
            end
        end
    end
    
    return (nodes=length(xvec),), (nodes=(x=xvec, y=yvec, z=zvec),)
end

######################################################################################################
# 2D RODS
######################################################################################################

function rods2D(nagents::Int, position_dist::Function, radius_dist::Function, 
                length_dist::Function, orientation_dist::Function)
    """
    2D rod packing with variable positions, radii, lengths and orientations.
    
    Args:
        nagents: Number of rods
        position_dist: Function returning (x, y) position
        radius_dist: Function returning radius
        length_dist: Function returning length
        orientation_dist: Function returning angle in radians
    
    Returns:
        (nodes=N,), (nodes=(x,y,r,l,px,py),)
    """
    
    xvec = Vector{Float64}[]
    yvec = Vector{Float64}[]
    rvec = Vector{Float64}[]
    lengthvec = Vector{Float64}[]
    orientationx = Vector{Int}[]
    orientationy = Vector{Int}[]
    
    for agent in 1:nagents
        x, y = position_dist()
        radius = radius_dist()
        length = length_dist()
        px, py = orientation_dist()
        
        push!(xvec, x)
        push!(yvec, y)
        push!(rvec, radius)
        push!(lengthvec, length)
        push!(orientationx, px)
        push!(orientationy, py)
    end
    
    return (nodes=nagents,), (nodes=(x=xvec, y=yvec, r=rvec, l=lengthvec, px=orientationx, py=orientationy),)
end

######################################################################################################
# 3D RODS
######################################################################################################

function rods3D(nagents::Int, position_dist::Function, radius_dist::Function,
                length_dist::Function, orientation_dist::Function)
    """
    3D rod packing with variable positions, radii, lengths and orientations.
    
    Args:
        nagents: Number of rods
        position_dist: Function returning (x, y, z) position
        radius_dist: Function returning radius
        length_dist: Function returning length
        orientation_dist: Function returning (θ, φ) angles in radians (spherical coords)
    
    Returns:
        (nodes=N,), (nodes=(x,y,z,r,l,px,py,pz),)
    """
    
    xvec = Vector{Float64}[]
    yvec = Vector{Float64}[]
    zvec = Vector{Float64}[]
    rvec = Vector{Float64}[]
    lengthvec = Vector{Float64}[]
    orientationx = Vector{Float64}[]
    orientationy = Vector{Float64}[]
    orientationz = Vector{Float64}[]

    for agent in 1:nagents
        x, y, z = position_dist()
        radius = radius_dist()
        length = length_dist()
        px, py, pz = orientation_dist()
        
        push!(xvec, x)
        push!(yvec, y)
        push!(zvec, z)
        push!(rvec, radius)
        push!(lengthvec, length)
        push!(orientationx, px)
        push!(orientationy, py)
        push!(orientationz, pz)
    end

    return (nodes=nagents,), (nodes=(x=xvec, y=yvec, z=zvec, r=rvec, l=lengthvec, px=orientationx, py=orientationy, pz=orientationz),)
end

######################################################################################################
# 2D FILAMENTS
######################################################################################################

function filaments2D(nfilaments::Int, nedges_per_filament::Function, position_dist::Function,
                     initial_angle_dist::Function, persistence_dist::Function)
    """
    2D filament networks with persistent random walk.
    
    Args:
        nfilaments: Number of filaments
        nedges_per_filament: Number of edges per filament
        position_dist: Function returning (x, y) starting position
        initial_angle_dist: Function returning initial angle in radians
        persistence_dist: Function returning persistence length
    
    Returns:
        (nodes=N,), (nodes=(x,y),), (edges=E,), (edges=(node1,node2,agent),)
    """
    
    xvec = Vector{Float64}[]
    yvec = Vector{Float64}[]
    edges = Vector{Tuple{Int,Int}}[]
    edge_agent = Vector{Int}[]
    
    node_count = 0
    segment_length = 1.0  # Unit segment length
    
    for fil in 1:nfilaments
        x, y = position_dist()
        angle = initial_angle_dist()
        persistence = persistence_dist()
        
        # Create nodes along filament
        filament_nodes = [node_count + 1]
        push!(xvec, x)
        push!(yvec, y)
        node_count += 1
        
        for step in 1:nedges_per_filament()
            # Update angle with persistence
            angle_change = randn() / persistence
            angle += angle_change
            
            # New position
            x += segment_length * cos(angle)
            y += segment_length * sin(angle)
            
            push!(xvec, x)
            push!(yvec, y)
            node_count += 1
            push!(filament_nodes, node_count)
        end
        
        # Create edges
        for i in 1:length(filament_nodes)-1
            node1 = filament_nodes[i]
            node2 = filament_nodes[i+1]
            
            push!(edges, (node1, node2))
            push!(edge_agent, fil)
        end
    end
    
    return (nodes=length(xvec),edges=length(edges)), (nodes=(x=xvec, y=yvec), edges=(node1=[e[1] for e in edges], node2=[e[2] for e in edges]), agents=(id=edge_agent),)
end

######################################################################################################
# 3D FILAMENTS
######################################################################################################

function filaments3D(nfilaments::Int, nedges_per_filament::Function, position_dist::Function,
                     initial_angle_dist::Function, persistence_dist::Function)
    """
    3D filament networks with persistent random walk.
    
    Args:
        nfilaments: Number of filaments
        nedges_per_filament: Number of edges per filament
        position_dist: Function returning (x, y, z) starting position
        initial_angle_dist: Function returning (θ, φ) angles in radians
        persistence_dist: Function returning persistence length
    
    Returns:
        (nodes=N,), (nodes=(x,y,z),), (edges=E,), (edges=(node1,node2,agent),)
    """
    
    xvec = Vector{Float64}[]
    yvec = Vector{Float64}[]
    zvec = Vector{Float64}[]
    edges = Vector{Tuple{Int,Int}}[]
    edge_agent = Vector{Int}[]
    
    node_count = 0
    segment_length = 1.0
    
    for fil in 1:nfilaments
        x, y, z = position_dist()
        theta, phi = initial_angle_dist()
        persistence = persistence_dist()
        
        filament_nodes = [node_count + 1]
        push!(xvec, x)
        push!(yvec, y)
        push!(zvec, z)
        node_count += 1
        
        for step in 1:nedges_per_filament()
            # Update direction with persistence
            theta_change = randn() / persistence
            phi_change = randn() / persistence
            theta += theta_change
            phi += phi_change
            
            # Ensure valid spherical coordinates
            theta = clamp(theta, 1e-6, π - 1e-6)
            
            # New position
            x += segment_length * sin(theta) * cos(phi)
            y += segment_length * sin(theta) * sin(phi)
            z += segment_length * cos(theta)
            
            push!(xvec, x)
            push!(yvec, y)
            push!(zvec, z)
            node_count += 1
            push!(filament_nodes, node_count)
        end
        
        # Create edges
        for i in 1:length(filament_nodes)-1
            node1 = filament_nodes[i]
            node2 = filament_nodes[i+1]
            
            push!(edges, (node1, node2))
            push!(edge_agent, fil)
        end
    end
    
    return (nodes=length(xvec),edges=length(edges)), (nodes=(x=xvec, y=yvec, z=zvec),), (edges=length(edges),), (edges=(node1=[e[1] for e in edges], node2=[e[2] for e in edges]), agents=(edge_agent,),)
end

######################################################################################################
# 2D TREES
######################################################################################################

function trees2D(ntrees::Int, max_steps::Function, splitting_dist::Function,
                 initial_angle_dist::Function, nsplits_dist::Function, 
                 persistence_dist::Function)
    """
    2D branching tree networks with random splitting.
    
    Args:
        ntrees: Number of trees
        max_steps: Maximum steps before stopping growth
        splitting_dist: Function returning step interval for splitting
        initial_angle_dist: Function returning initial angle in radians
        nsplits_dist: Function returning number of splits at each branching point
        persistence_dist: Function returning persistence length
    
    Returns:
        (nodes=N,), (nodes=(x,y),), (edges=E,), (edges=(node1,node2,agent),)
    """
    
    xvec = Vector{Float64}[]
    yvec = Vector{Float64}[]
    edges = Vector{Tuple{Int,Int}}[]
    edge_agent = Vector{Int}[]
    
    node_count = 0
    segment_length = 1.0
    agent_counter = 0
    
    for tree in 1:ntrees
        # Stack of (position, angle, steps_taken, parent_agent_id)
        growth_stack = [(0.0, 0.0, initial_angle_dist(), 0, -1)]
        
        while !isempty(growth_stack)
            x, y, step, angle, parent_agent = pop!(growth_stack)
            
            if step >= max_steps
                continue
            end
            
            agent_counter += 1
            persistence = persistence_dist()
            split_interval = splitting_dist()
            
            # Grow segment
            angle_change = randn() / persistence
            angle += angle_change
            
            x_new = x + segment_length * cos(angle)
            y_new = y + segment_length * sin(angle)
            
            node1 = node_count + 1
            node_count += 1
            push!(xvec, x)
            push!(yvec, y)
            
            node2 = node_count + 1
            node_count += 1
            push!(xvec, x_new)
            push!(yvec, y_new)
            
            push!(edges, (node1, node2))
            push!(edge_agent, agent_counter)
            
            # Check for splitting
            if step > 0 && mod(step, split_interval) == 0
                nsplits = nsplits_dist()
                for _ in 1:nsplits
                    split_angle = angle + (rand() - 0.5) * π / 2
                    push!(growth_stack, (x_new, y_new, step + 1, split_angle, agent_counter))
                end
            else
                push!(growth_stack, (x_new, y_new, step + 1, angle, agent_counter))
            end
        end
    end
    
    return (nodes=length(xvec), edges=length(edges)), (nodes=(x=xvec, y=yvec), edges=(node1=[e[1] for e in edges], node2=[e[2] for e in edges]), agent=(id=edge_agent,),)
end

######################################################################################################
# 3D TREES
######################################################################################################

function trees3D(ntrees::Int, max_steps::Function, splitting_dist::Function,
                 initial_angle_dist::Function, nsplits_dist::Function,
                 persistence_dist::Function)
    """
    3D branching tree networks with random splitting.
    
    Args:
        ntrees: Number of trees
        max_steps: Maximum steps before stopping growth
        splitting_dist: Function returning step interval for splitting
        initial_angle_dist: Function returning (θ, φ) angles in radians
        nsplits_dist: Function returning number of splits at each branching point
        persistence_dist: Function returning persistence length
    
    Returns:
        (nodes=N,), (nodes=(x,y,z),), (edges=E,), (edges=(node1,node2,agent),)
    """
    
    xvec = Vector{Float64}[]
    yvec = Vector{Float64}[]
    zvec = Vector{Float64}[]
    edges = Vector{Tuple{Int,Int}}[]
    edge_agent = Vector{Int}[]
    
    node_count = 0
    segment_length = 1.0
    agent_counter = 0
    
    for tree in 1:ntrees
        # Stack of (position, angles, steps_taken, parent_agent_id)
        growth_stack = [((0.0, 0.0, 0.0), initial_angle_dist(), 0, -1)]
        
        while !isempty(growth_stack)
            (x, y, z), (theta, phi), step, parent_agent = pop!(growth_stack)
            
            if step >= max_steps
                continue
            end
            
            agent_counter += 1
            persistence = persistence_dist()
            split_interval = splitting_dist()
            
            # Grow segment
            theta_change = randn() / persistence
            phi_change = randn() / persistence
            theta += theta_change
            phi += phi_change
            
            theta = clamp(theta, 1e-6, π - 1e-6)
            
            x_new = x + segment_length * sin(theta) * cos(phi)
            y_new = y + segment_length * sin(theta) * sin(phi)
            z_new = z + segment_length * cos(theta)
            
            node1 = node_count + 1
            node_count += 1
            push!(xvec, x)
            push!(yvec, y)
            push!(zvec, z)
            
            node2 = node_count + 1
            node_count += 1
            push!(xvec, x_new)
            push!(yvec, y_new)
            push!(zvec, z_new)
            
            push!(edges, (node1, node2))
            push!(edge_agent, agent_counter)
            
            # Check for splitting
            if step > 0 && mod(step, split_interval) == 0
                nsplits = nsplits_dist()
                for _ in 1:nsplits
                    split_theta = theta + (rand() - 0.5) * π / 4
                    split_phi = phi + (rand() - 0.5) * π / 2
                    push!(growth_stack, ((x_new, y_new, z_new), (split_theta, split_phi), step + 1, agent_counter))
                end
            else
                push!(growth_stack, ((x_new, y_new, z_new), (theta, phi), step + 1, agent_counter))
            end
        end
    end
    
    return (nodes=length(xvec), edges=length(edges)), (nodes=(x=xvec, y=yvec, z=zvec),), (edges=length(edges),), (edges=(node1=[e[1] for e in edges], node2=[e[2] for e in edges]), agents=(edge_agent,),)
end

######################################################################################################
# 2D VORONOI MESHES
######################################################################################################

function voronoi2D(ncells::Int, mesh_space::Mesh)
    """
    2D Voronoi tessellation mesh.
    
    Args:
        ncells: Number of Voronoi cells
        mesh_space: (width, height) of domain
        periodic_x: Whether to apply periodic boundary in x
        periodic_y: Whether to apply periodic boundary in y
    
    Returns:
        (nodes=N,), (nodes=(x,y),), (faces=F,), (faces=(cell_nodes,),)
    """
    
    error("Voronoi mesh generation requires DelaunayTriangulation.")

end

function voronoi2DPeriodic(ncells::Int, mesh_space::Tuple{Float64, Float64})
    """
    2D Voronoi tessellation mesh.
    
    Args:
        ncells: Number of Voronoi cells
        mesh_space: (width, height) of domain
        periodic_x: Whether to apply periodic boundary in x
        periodic_y: Whether to apply periodic boundary in y
    
    Returns:
        (nodes=N,), (nodes=(x,y),), (faces=F,), (faces=(cell_nodes,),)
    """
    
    error("Voronoi mesh generation requires DelaunayTriangulation.")

end

######################################################################################################
# 3D VORONOI MESHES
######################################################################################################

function voronoi3D(ncells::Int, mesh_space::Tuple{Float64, Float64, Float64},
                   periodic_x::Bool=false, periodic_y::Bool=false, periodic_z::Bool=false)
    """
    3D Voronoi tessellation mesh.
    
    Args:
        ncells: Number of Voronoi cells
        mesh_space: (width, height, depth) of domain
        periodic_x, periodic_y, periodic_z: Periodic boundary conditions
    
    Returns:
        (nodes=N,), (nodes=(x,y,z),), (volumes=V,), (volumes=(cell_nodes,),)
    """

    error("3D Voronoi mesh generation requires DelaunayTriangulation.jl.")
end
