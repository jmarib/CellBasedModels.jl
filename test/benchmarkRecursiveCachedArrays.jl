using CellBasedModels
using BenchmarkTools
using CUDA   
using RecursiveArrayTools
    
cache = (zeros(N), zeros(N))
xt = CellBasedModels.TupleOfCachedArrays(cache, ns)
xt.u[1] .= 1; xt.u[2] .= 1

cache = (zeros(N), zeros(N))
yt = CellBasedModels.TupleOfCachedArrays(cache, ns)
yt.u[1] .= 10; yt.u[2] .= 10;

cache = (zeros(N), zeros(N))
vt = CellBasedModels.TupleOfCachedArrays(cache, ns)

@test all(xt.ns .== ns) && all(yt.ns .== ns) && all(vt.ns .== ns)
@test all(xt.u[1] .== 1) && all(xt.u[2] .== 1)
@test all(yt.u[1] .== 10) && all(yt.u[2] .== 10)
@test all(vt.u[1] .== 0) && all(vt.u[2] .== 0)

f!(xt,yt,vt)
@test all(vt.u[1][1:ns[1]] .≈ 10.1) && all(vt.u[2][1:ns[2]] .≈ 10.1)
@test all(vt.u[1][ns[1]+1:end] .== 0) && all(vt.u[2][ns[2]+1:end] .== 0)

# GPU
cache = (CUDA.zeros(N), CUDA.zeros(N))
xt_cuda = CellBasedModels.TupleOfCachedArrays(cache, ns)
xt_cuda.u[1] .= 1; xt_cuda.u[2] .= 1

cache = (CUDA.zeros(N), CUDA.zeros(N))
yt_cuda = CellBasedModels.TupleOfCachedArrays(cache, ns)
yt_cuda.u[1] .= 10; yt_cuda.u[2] .= 10;

cache = (CUDA.zeros(N), CUDA.zeros(N))
vt_cuda = CellBasedModels.TupleOfCachedArrays(cache, ns)

@test all(vt_cuda.u[1][1:ns[1]] .≈ 10.1) && all(vt_cuda.u[2][1:ns[2]] .≈ 10.1)
@test all(vt_cuda.u[1][ns[1]+1:end] .== 0) && all(vt_cuda.u[2][ns[2]+1:end] .== 0)

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
println("CUDA Array: ")
@btime f!(x_cuda,y_cuda,v_cuda)
println("CUDA Array Views: ")
x_cuda_view = @views x_cuda[1:sum(ns)]; y_cuda_view = @views y_cuda[1:sum(ns)]; v_cuda_view = @views v_cuda[1:sum(ns)];
@btime @views f!(x_cuda_view,y_cuda_view,v_cuda_view)
println("CUDA TupleOfCachedArrays: ")
@btime f!(xt_cuda,yt_cuda,vt_cuda);
