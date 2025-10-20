CellBasedModels.toCPU(cp::CommunityPoint{CommunityPointMeta}) = Adapt.adapt(SizedVector, cp)
CellBasedModels.toCPU(cp::CommunityPoint{CommunityPointMeta{<:SizedVector, <:SizedVector}}) = cp

CellBasedModels.toGPU(cp::CommunityPoint{CommunityPointMeta}) = cp
CellBasedModels.toGPU(cp::CommunityPoint{B}) where {B<:CommunityPointMeta{<:SizedVector, <:SizedVector}} = Adapt.adapt(CuArray, cp)