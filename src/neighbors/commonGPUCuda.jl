import CellBasedModels: compactArray!, compactUnstructuredMeshField!

function compactUnstructuredMeshField!(prop::UnstructuredMeshField{P}, perm::CUDA.CuArray, aux::NamedTuple, NNew) where {P <: GPUCuda}
    N = lengthProperties(prop)
    for (fieldname, field) in pairs(prop._p)
        aux_field = getfield(aux, fieldname)
        compactArray!(field, aux_field, perm, N, NNew)
    end
    prop._N .= NNew
    return nothing
end

function compactArray_kernel!(data, auxBuffer, perm, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        if perm[i] != 0
            auxBuffer[perm[i]] = data[i]
        end
    end
    return
end

function compactArray!(data::CUDA.CuArray, auxBuffer::CUDA.CuArray, perm::CUDA.CuArray, N::Int, NNew::Int)
    # Copy data to target positions using auxiliary buffer (parallelizable)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks compactArray_kernel!(data, auxBuffer, perm, N)
    # Synchronize to ensure kernel completion
    CUDA.synchronize()
    # Copy back from auxiliary buffer to original array
    @inbounds @views data[1:NNew] .= auxBuffer[1:NNew]
    return nothing
end
