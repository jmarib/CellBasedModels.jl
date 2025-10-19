
iterator = CartesianIndices((1:3,1:2,1:5))
@btime for i in iterator
    
end

iterator = CellBasedModels.CommunityIndices((a=((:a1, :a2), 5), b=((:b1, :b2, :b3), 5)))
@btime for i in iterator
    
end

# for i in CartesianIndices((1:2,1:3,1:5))
#     @info "i: $i"
# end

# iterator = CellBasedModels.CommunityIndices(
#     (
#         a = ((:a1, :a2, :a3), 5),
#         b = ((:b1, :b2, :b3), 5)
#     )
# )

# @btime for i in iterator
#     nothing
# end

# for (outer, inner, i) in iterator
#     @info "Outer: $outer, Inner: $inner, Index: $i"
# end