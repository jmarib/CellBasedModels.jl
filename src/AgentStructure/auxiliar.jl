function toCPU(x); @warn "No GPU found or no conversion for type $(typeof(x)). Fallback to CPU."; x end   # fallback definitions
function toGPU(x); @warn "No GPU found or no conversion for type $(typeof(x)). Fallback to CPU."; x end

# Generic unpackig for broadcasting

@inline function unpack_voa(bc::Broadcast.Broadcasted{Style}, i) where {Style}
    Broadcast.Broadcasted{Style}(bc.f, unpack_args_voa(i, bc.args))
end
unpack_voa(x, ::Any) = x

@inline function unpack_args_voa(i, args::Tuple)
    v = (unpack_voa(args[1], i), unpack_args_voa(i, Base.tail(args))...)
    return v
end
unpack_args_voa(::Any, args::Tuple{}) = ()

# # function unpack_voa(x::AbstractArray{T, N}, i) where {T, N}
# #     @view x[ntuple(x -> Colon(), N - 1)..., i]
# # end


# @inline function unpack_args_voa(i, args::NTuple{N, <:AbstractArray}) where {N}
#     ntuple(k -> unpack_voa(args[k], i), N)
# end
# unpack_args_voa(i, args::Tuple{Any}) = (unpack_voa(args[1], i),)

struct CommunityIndices{M}
    modules::M
end
function Base.iterate(ci::CommunityIndices, state=(1, 1, 1))
    (i_outer, i_inner, i_array) = state
    mods = ci.modules

    # stop when outer index past end
    if i_outer > length(mods)
        return nothing
    end

    # unpack inner structure
    (subkeys, N) = mods[i_outer]

    # if finished current N range, move to next inner key
    if i_array > N
        return iterate(ci, (i_outer, i_inner + 1, 1))
    end

    # if finished current subkey tuple, move to next module
    if i_inner > length(subkeys)
        return iterate(ci, (i_outer + 1, 1, 1))
    end

    # extract current names
    outer_key = keys(mods)[i_outer]
    inner_key = subkeys[i_inner]

    # yield current triple, and increment array index
    ((outer_key, inner_key, i_array), (i_outer, i_inner, i_array + 1))
end

import RecursiveArrayTools: recursivecopy!
import Base: copyto!

function recursivecopy!(b, a)
    copyto!(b, a)
end