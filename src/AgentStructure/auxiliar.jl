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