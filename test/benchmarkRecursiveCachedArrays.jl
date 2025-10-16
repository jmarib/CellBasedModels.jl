using RecursiveArrayTools

N = 100000
ns_vec = [100, 20] #sizes of each array in the tuple
ns = (a=100, b=20) #sizes of each array in the tuple

f!(x,y,v) = v .= x .* 0.1 .+ y

function f_!(x::NTuple{N,Any}, y::NTuple{N,Any}, v::NTuple{N,Any}) where {N}
    for i in 1:N
        f!(x[i], y[i], v[i])
    end
    return nothing
end

function f2!(x, y, n)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    @inbounds for i in index:stride:n[1]
        y[i] = 1.0 * x[i]
    end
    return nothing
end

function f2!(vt)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    y = vt.b
    x = vt.a
    ns = vt._ns
    @inbounds for i in index:stride:ns.a
        y[i] = 1.0 * x[i]
    end
end

cache = (a = zeros(N), b = zeros(N))
xt = CellBasedModels.TupleOfCachedArrays(cache, ns)
for i in 1:length(ns)
    xt.a .= 1
end

cache = (a = zeros(N), b = zeros(N))
yt = CellBasedModels.TupleOfCachedArrays(cache, ns)
for i in 1:length(ns)
    yt.b .= 10
end

cache = (a = zeros(N), b = zeros(N))
vt = CellBasedModels.TupleOfCachedArrays(cache, ns)

# GPU
cache = (a = CUDA.zeros(N), b = CUDA.zeros(N))
xt_cuda = CellBasedModels.TupleOfCachedArrays(cache, ns)
for i in 1:length(ns)
    xt_cuda.a .= 1
end

cache = (a = CUDA.zeros(N), b = CUDA.zeros(N))
yt_cuda = CellBasedModels.TupleOfCachedArrays(cache, ns)
for i in 1:length(ns)
    yt_cuda.a .= 10
end

cache = (a = CUDA.zeros(N), b = CUDA.zeros(N))
vt_cuda = CellBasedModels.TupleOfCachedArrays(cache, ns)

# Benchmarking
cache = [zeros(N) for i in 1:length(ns)]
xv = VectorOfArray(cache)
for i in 1:length(ns)
    xv.u[i] .= 1
end

cache = [zeros(N) for i in 1:length(ns)]
yv = VectorOfArray(cache)
for i in 1:length(ns)
    yv.u[i] .= 10
end

cache = [zeros(N) for i in 1:length(ns)]
vv = VectorOfArray(cache)

x = tuple([zeros(N) .+ 1 for i in 1:length(ns)]...);
y = tuple([zeros(N) .+ 10 for i in 1:length(ns)]...);
v = tuple([zeros(N) for i in 1:length(ns)]...);

println("\nBenchmarking RecursiveCachedArrays N = $N with sizes = $ns")
println("=============================================================")

@printf("%-35s", "Tuple Arrays:")
@btime f_!(x,y,v)

@printf("%-35s", "ArrayOfVectors:")
@btime f!(xv,yv,vv)

@printf("%-35s", "Tuple Array Views:")
x_view = tuple([@views x[i][1:ns_vec[i]] for i in 1:length(ns)]...);
y_view = tuple([@views y[i][1:ns_vec[i]] for i in 1:length(ns)]...);
v_view = tuple([@views v[i][1:ns_vec[i]] for i in 1:length(ns)]...);
@btime @views f_!(x_view,y_view,v_view)

@printf("%-35s", "*TupleOfCachedArrays:")
@btime f!(xt,yt,vt)

x_cuda = tuple([CUDA.zeros(N).+ 1 for i in ns]...); 
y_cuda = tuple([CUDA.zeros(N).+ 10 for i in ns]...); 
v_cuda = tuple([CUDA.zeros(N) for i in ns]...);

println()
@printf("%-35s", "Tuple Array CUDA:")
@btime f_!(x_cuda,y_cuda,v_cuda)

@printf("%-35s", "Tuple Array Views CUDA:")
x_cuda_view = tuple([@views x_cuda[i][1:ns_vec[i]] for i in 1:length(ns)]...);
y_cuda_view = tuple([@views y_cuda[i][1:ns_vec[i]] for i in 1:length(ns)]...);
v_cuda_view = tuple([@views v_cuda[i][1:ns_vec[i]] for i in 1:length(ns)]...);
@btime @views f_!(x_cuda_view,y_cuda_view,v_cuda_view)

@printf("%-35s", "*TupleOfCachedArrays CUDA:")
@btime f!(xt_cuda,yt_cuda,vt_cuda)

cache = (a = CUDA.zeros(N), b = CUDA.ones(N)) 
vt = CellBasedModels.TupleOfCachedArrays(cache, ns)
x = CUDA.zeros(N) .+ 1; y = CUDA.zeros(N) .+ 10; n = CUDA.CuArray([100,20]);

println()
@printf("%-35s", "Kernel Array CUDA:")
@btime @cuda threads=254 f2!(x, y, n)

@printf("%-35s", "*Kernel TupleOfCachedArrays CUDA:")
@btime @cuda threads=254 f2!(vt)
