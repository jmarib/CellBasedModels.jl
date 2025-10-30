import CellBasedModels: UnstructuredMeshObjectField, UnstructuredMeshObject
import CellBasedModels: toCPU, toGPU, CPU, CPUSinglethread, CPUMultithreading

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

Base.length(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuda} = CUDA.@allowscalar field._N[1]
Base.length(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuDevice} = field._N[1]
lengthCache(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuda} = CUDA.@allowscalar field._NCache[1]
lengthCache(field::UnstructuredMeshObjectField{P}) where {P<:GPUCuDevice} = field._NCache[1]

toCPU(mesh::UnstructuredMeshObject{D, P}) where {D, P<:CPU} = mesh
toGPU(mesh::UnstructuredMeshObject{D, P}) where {D, P<:GPUCuda} = mesh

function toCPU(field::UnstructuredMeshObject{P, D, S}) where {P<:GPUCuda, D, S}

    PNew = platform()

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

    UnstructuredMeshObject{PNew, D, S, A, N, E, F, V}(
        a,
        n,
        e,
        f,
        v,
    )
end

function toGPU(field::UnstructuredMeshObject{P, D, S}) where {P<:CPU, D, S}

    PNew = GPUCuda

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

    UnstructuredMeshObject{PNew, D, S, A, N, E, F, V}(
        a,
        n,
        e,
        f,
        v,
    )
end