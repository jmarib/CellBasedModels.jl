using CUDA   
using RecursiveArrayTools
    
N = 100000
ns = [100,20]

f!(x,y,v) = v .= x .* 0.1 .+ y

cache = (zeros(N), zeros(N))
xt = CellBasedModels.TupleOfCachedArrays(cache, ns)
xt.u[1] .= 1; xt.u[2] .= 1

cache = (zeros(N), zeros(N))
yt = CellBasedModels.TupleOfCachedArrays(cache, ns)
yt.u[1] .= 10; yt.u[2] .= 10;

cache = (zeros(N), zeros(N))
vt = CellBasedModels.TupleOfCachedArrays(cache, ns)

# GPU
cache = (CUDA.zeros(N), CUDA.zeros(N))
xt_cuda = CellBasedModels.TupleOfCachedArrays(cache, ns)
xt_cuda.u[1] .= 1; xt_cuda.u[2] .= 1

cache = (CUDA.zeros(N), CUDA.zeros(N))
yt_cuda = CellBasedModels.TupleOfCachedArrays(cache, ns)
yt_cuda.u[1] .= 10; yt_cuda.u[2] .= 10;

cache = (CUDA.zeros(N), CUDA.zeros(N))
vt_cuda = CellBasedModels.TupleOfCachedArrays(cache, ns)

# Benchmarking
cache = [zeros(N), zeros(N)]
xv = VectorOfArray(cache)
xv.u[1] .= 1; xv.u[2] .= 1

cache = [zeros(N), zeros(N)]
yv = VectorOfArray(cache)
yv.u[1] .= 10; yv.u[2] .= 10;

cache = [zeros(N), zeros(N)]
vv = VectorOfArray(cache)
x = zeros(2*N) .+ 1; y = zeros(2*N) .+ 10; v = zeros(2*N);

println("Array: ")
@btime f!(x,y,v)
println("Array Views: ")
x_view = @views x[1:sum(ns)]; y_view = @views y[1:sum(ns)]; v_view = @views v[1:sum(ns)];
@btime @views f!(x_view,y_view,v_view)
println("ArrayOfVectors: ")
@btime f!(xv,yv,vv); 
println("TupleOfCachedArrays: ")
@btime f!(xt,yt,vt);

x_cuda = CUDA.zeros(2*N) .+ 1; y_cuda = CUDA.zeros(2*N) .+ 10; v_cuda = CUDA.zeros(2*N);
println("Array CUDA: ")
@btime f!(x_cuda,y_cuda,v_cuda)
println("Array Views CUDA: ")
x_cuda_view = @views x_cuda[1:sum(ns)]; y_cuda_view = @views y_cuda[1:sum(ns)]; v_cuda_view = @views v_cuda[1:sum(ns)];
@btime @views f!(x_cuda_view,y_cuda_view,v_cuda_view)
println("TupleOfCachedArrays CUDA: ")
@btime f!(xt_cuda,yt_cuda,vt_cuda);

ns = CUDA.CuArray(ns)

cache = (CUDA.zeros(N), CUDA.ones(N))
vt = CellBasedModels.TupleOfCachedArrays(cache, ns)
function f2!(vt)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    y = vt.u[1]
    x = vt.u[2]
    ns = vt.ns
    @inbounds for i in index:stride:ns[1]
        y[i] = 1.0 * x[i]
    end
end
@cuda threads=254 f2!(vt)
println("Kernel TupleOfCachedArrays CUDA: ")
@btime @cuda threads=254 f2!(vt)

function f2!(x, y, n)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    @inbounds for i in index:stride:n[1]
        y[i] = 1.0 * x[i]
    end
    return nothing
end

x = CUDA.zeros(2*N) .+ 1; y = CUDA.zeros(2*N) .+ 10; n = CUDA.CuArray([100,20]);
@cuda threads=254 f2!(x, y, n)
println("Kernel Array CUDA: ")
@btime @cuda threads=254 f2!(x, y, n)