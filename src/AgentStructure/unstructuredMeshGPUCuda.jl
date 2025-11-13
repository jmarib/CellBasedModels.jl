import CellBasedModels: DATATYPE
import CellBasedModels: lengthCache, lengthProperties, sizeFull, sizeFullCache, nCopyProperties
import CellBasedModels: UnstructuredMeshObjectField, UnstructuredMeshObjectFieldStyle, UnstructuredMeshObject, UnstructuredMeshObjectStyle, unpack_voa
import CellBasedModels: toCPU, toGPU, CPU, CPUSinglethread, CPUMultithreading

lengthCache(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuda} = CUDA.@allowscalar field._NCache[1]
lengthCache(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuDevice} = field._NCache[1]
lengthProperties(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuda} = CUDA.@allowscalar field._N[1]
lengthProperties(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuDevice} = field._N[1]
Base.length(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuda} = nCopyProperties(field) * CUDA.@allowscalar field._N[1]
Base.length(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuDevice} = nCopyProperties(field) * field._N[1]

sizeFull(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuda} = (field._NP, CUDA.@allowscalar field._N[1])
sizeFull(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuDevice} = (field._NP, field._N[1])
sizeFullCache(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuda} = (field._NP, CUDA.@allowscalar field._NCache[1])
sizeFullCache(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuDevice} = (field._NP, field._NCache[1])
Base.size(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuda} = (nCopyProperties(field), lengthProperties(field))
Base.size(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuDevice} = (nCopyProperties(field), lengthProperties(field))

# Base.length(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuda} = CUDA.@allowscalar field._N[1]
# Base.length(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuDevice} = field._N[1]
# lengthCache(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuda} = CUDA.@allowscalar field._NCache[1]
# lengthCache(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuDevice} = field._NCache[1]

########################################################################################
# broadcasting support
########################################################################################
for type in [
        Broadcast.Broadcasted{<:UnstructuredMeshObjectField},
        Broadcast.Broadcasted{<:UnstructuredMeshObjectFieldStyle},
    ]

    @eval @inline function Base.copyto!(
            dest::UnstructuredMeshObjectField{P, DT, PR, PRN, PRC},
            bc::$type) where {P<:Union{GPUCuda, GPUCuDevice}, DT, PR, PRN, PRC}
        bc = Broadcast.flatten(bc)
        N = lengthProperties(dest)
        @inbounds for i in 1:PRN
            if !dest._pReference[i]
                dest_ = @views dest._p[i][1:N]
                dest_ .= unpack_voa(bc, i)
            end
        end
        dest
    end
end

function unpack_voa(x::UnstructuredMeshObjectField{P}, i) where {P<:Union{GPUCuda, GPUCuDevice}}
    @views x._p[i][1:lengthProperties(x)]
end

for type in [
        Broadcast.Broadcasted{<:UnstructuredMeshObject},
        Broadcast.Broadcasted{<:UnstructuredMeshObjectStyle},
    ]

    @eval @inline function Base.copyto!(
            dest::UnstructuredMeshObject{P, A, N, E, F, V},
            bc::$type) where {P<:Union{GPUCuda, GPUCuDevice}, A, N, E, F, V}
        bc = Broadcast.flatten(bc)

        d = getfield(dest, :a)
        if d !== nothing
            np, n = sizeFull(d)
            for j in 1:np
                if !d._pReference[j]
                    dest_ = @views d._p[j][1:n]
                    dest_ .= unpack_voa(bc, :a, j, n)
                end
            end
        end

        d = getfield(dest, :n)
        if d !== nothing
            np, n = sizeFull(d)
            for j in 1:np
                if !d._pReference[j]
                    dest_ = @views d._p[j][1:n]
                    dest_ .= unpack_voa(bc, :n, j, n)
                end
            end
        end

        d = getfield(dest, :e)
        if d !== nothing
            np, n = sizeFull(d)
            for j in 1:np
                if !d._pReference[j]
                    dest_ = @views d._p[j][1:n]
                    dest_ .= unpack_voa(bc, :e, j, n)
                end
            end
        end

        d = getfield(dest, :f)
        if d !== nothing
            np, n = sizeFull(d)
            for j in 1:np
                if !d._pReference[j]
                    dest_ = @views d._p[j][1:n]
                    dest_ .= unpack_voa(bc, :f, j, n)
                end
            end
        end

        d = getfield(dest, :v)
        if d !== nothing
            np, n = sizeFull(d)
            for j in 1:np
                if !d._pReference[j]
                    dest_ = @views d._p[j][1:n]
                    dest_ .= unpack_voa(bc, :v, j, n)
                end
            end
        end
    end
end

function unpack_voa(x::UnstructuredMeshObject{P}, i, j, n) where {P<:Union{GPUCuda, GPUCuDevice}}
    @views x[i]._p[j][1:n]
    # x[i]._p[j]
end

########################################################################################
# to CPU / to GPU conversions
########################################################################################
toCPU(field::UnstructuredMeshObjectField{P}) where {P<:CPU} = field
toGPU(field::UnstructuredMeshObjectField{P}) where {P<:GPU} = field

function toCPU(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuda}
    UnstructuredMeshObjectField(
        field._p              === nothing ? nothing : Adapt.adapt(Array, field._p),
        field._NP             === nothing ? nothing : field._NP,
        field._pReference     === nothing ? nothing : SizedVector{field._NP, Bool}([field._pReference...]),
        field._id             === nothing ? nothing : Vector{Int}(field._id),
        field._idMax          === nothing ? nothing : Threads.Atomic{Int}(Array(field._idMax)[1]),
        field._N              === nothing ? nothing : Threads.Atomic{Int}(Array(field._N)[1]),
        field._NCache         === nothing ? nothing : Threads.Atomic{Int}(Array(field._NCache)[1]),
        field._FlagsRemoved   === nothing ? nothing : Vector{Bool}(field._FlagsRemoved),
        field._NRemoved       === nothing ? nothing : Threads.Atomic{Int}(0),
        field._NRemovedThread === nothing ? nothing : SizedVector{Threads.nthreads(), Int}(zeros(Int, Threads.nthreads())),
        field._NAdded         === nothing ? nothing : Threads.Atomic{Int}(0),
        field._NAddedThread   === nothing ? nothing : SizedVector{Threads.nthreads(), Int}(zeros(Int, Threads.nthreads())),
        field._AddedAgents    === nothing ? nothing : [Vector{NamedTuple{keys(field._p), Tuple{[CellBasedModels.standardDataType(eltype(i)) for i in values(field._p)]...}}}() for _ in 1:Threads.nthreads()],
        field._FlagOverflow   === nothing ? nothing : Threads.Atomic{Bool}(false),
    )
end

function toGPU(field::UnstructuredMeshObjectField{P}) where {P<:CPU}
    UnstructuredMeshObjectField(
        field._p              === nothing ? nothing : Adapt.adapt(CUDA.CuArray, field._p),
        field._NP             === nothing ? nothing : field._NP,
        field._pReference     === nothing ? nothing : tuple(field._pReference...),
        field._id             === nothing ? nothing : CUDA.CuArray(field._id),
        field._idMax          === nothing ? nothing : CUDA.CuArray([field._idMax[]]),
        field._N              === nothing ? nothing : CUDA.CuArray([field._N[]]),
        field._NCache         === nothing ? nothing : CUDA.CuArray([field._NCache[]]),
        field._FlagsRemoved   === nothing ? nothing : CUDA.CuArray(field._FlagsRemoved),
        field._NRemoved       === nothing ? nothing : CUDA.CuArray([0]),
        field._NRemovedThread === nothing ? nothing : CUDA.zeros(0),
        field._NAdded         === nothing ? nothing : CUDA.CuArray([0]),
        field._NAddedThread   === nothing ? nothing : CUDA.zeros(0),
        field._AddedAgents    === nothing ? nothing : CUDA.zeros(0),
        field._FlagOverflow   === nothing ? nothing : CUDA.CuArray([false]),
    )
end

toCPU(mesh::UnstructuredMeshObject{D, P}) where {D, P<:CPU} = mesh
toGPU(mesh::UnstructuredMeshObject{D, P}) where {D, P<:GPUCuda} = mesh

function toCPU(field::UnstructuredMeshObject{P, D, S, DT}) where {P<:GPUCuda, D, S, DT}

    PNew = platform()
    DTNew = DT <: AbstractFloat ? DATATYPE[AbstractFloat] : DT

    a = field.a === nothing ? nothing : toCPU(field.a)
    n = field.n === nothing ? nothing : toCPU(field.n)
    e = field.e === nothing ? nothing : toCPU(field.e)
    f = field.f === nothing ? nothing : toCPU(field.f)
    v = field.v === nothing ? nothing : toCPU(field.v)

    A = typeof(a)
    N = typeof(n)
    E = typeof(e)
    F = typeof(f)
    V = typeof(v)

    UnstructuredMeshObject{PNew, D, S, DTNew, A, N, E, F, V}(
        a,
        n,
        e,
        f,
        v,
    )
end

function toGPU(field::UnstructuredMeshObject{P, D, S, DT}) where {P<:CPU, D, S, DT}

    PNew = GPUCuda
    DTNew = DT <: AbstractFloat ? Float32 : DT

    a = field.a === nothing ? nothing : toGPU(field.a)
    n = field.n === nothing ? nothing : toGPU(field.n)
    e = field.e === nothing ? nothing : toGPU(field.e)
    f = field.f === nothing ? nothing : toGPU(field.f)
    v = field.v === nothing ? nothing : toGPU(field.v)

    A = typeof(a)
    N = typeof(n)
    E = typeof(e)
    F = typeof(f)
    V = typeof(v)

    UnstructuredMeshObject{PNew, D, S, DTNew, A, N, E, F, V}(
        a,
        n,
        e,
        f,
        v,
    )
end