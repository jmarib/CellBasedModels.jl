using RecursiveArrayTools
using Accessors
using DifferentialEquations

struct IntegratorParams
    _ns
    _params_cache
    _params
end

# Primary constructor
function IntegratorParams(ns::AbstractVector{<:Integer}, params_cache::VectorOfArray)
    params = VectorOfArray([@view params_cache.u[i][1:ns[i]] for i in eachindex(ns)])
    IntegratorParams(ns, params_cache, params)
end

function IntegratorParams(ns::AbstractVector{<:Integer}, params_cache)
    params = VectorOfArray([@view params_cache[i][1:ns[i]] for i in eachindex(ns)])
    IntegratorParams(ns, VectorOfArray(params_cache), params)
end

# “Update” views after ns or cache changed (mutating)
function update(s::IntegratorParams)
    return IntegratorParams(s._ns, s._params_cache)
end

function update(s::IntegratorParams, ns::AbstractVector{<:Integer})
    s._ns .= ns
    return IntegratorParams(s._ns, s._params_cache)
end

function update(integrator, ns)

    for name in propertynames(integrator)
        if typeof(getproperty(integrator, name)) <: IntegratorParams
            b = getproperty(integrator, name)
            lens = @optic _._params
            params_cache = getproperty(b, :_params_cache)
            vals = VectorOfArray([@view params_cache.u[i][1:ns[i]] for i in eachindex(ns)])
            set(b, lens, vals)
            b._ns .= ns
            println("Updated property $name with new sizes: $b")
        # else
        #     println("Warning: Property $name is not of type IntegratorParams, skipping update. Is type: $(typeof(getproperty(integrator, name)))")
        end
    end

end

# Treat IntegratorParams as a scalar for broadcasting
Base.length(x::IntegratorParams) = length(x._params)
Base.eltype(x::IntegratorParams) = eltype(x._params)
Base.size(x::IntegratorParams) = size(x._params)
Base.axes(x::IntegratorParams) = axes(x._params)
Base.ndims(x::IntegratorParams) = ndims(x._params)
Base.ndims(x::Type{<:IntegratorParams}) = 1

Base.:/(x::IntegratorParams, y::Number) = IntegratorParams(x._ns, x._params_cache, x._params ./ y)

Base.iterate(x::IntegratorParams) = iterate(x._params)
Base.iterate(x::IntegratorParams, state) = iterate(x._params, state)

Base.zero(x::IntegratorParams) = IntegratorParams(x._ns, zero(x._params_cache))

function Base.copy(x::IntegratorParams)
    return IntegratorParams(copy(x._ns), copy(x._params_cache))
end

function DifferentialEquations.recursivecopy!(dest::IntegratorParams, src::IntegratorParams)
    dest._ns .= src._ns
    for i in eachindex(dest._params.u)
        dest._params.u[i] .= src._params.u[i]
    end
end

# Base.broadcastable(x::IntegratorParams) = x._params
Broadcast.broadcastable(x::IntegratorParams) = x

# function Base.materialize!(dest::IntegratorParams, bc::Base.Broadcast.Broadcasted)
#     Base.materialize!(dest._params, bc)
#     return dest
# end

# Operations
for op in [:(Base.:-), :(Base.:+)]
    @eval function ($op)(A::IntegratorParams, B::IntegratorParams)
        ($op).(A, B)
    end
    @eval function ($op)(A::IntegratorParams, B::Number)
        ($op).(A, B)
    end
    @eval function ($op)(A::Number, B::IntegratorParams)
        ($op).(A, B)
    end
end

for op in [:(Base.:/), :(Base.:\), :(Base.:*)]
    if op !== :(Base.:/)
        @eval ($op)(A::Number, B::IntegratorParams) = ($op).(A, B)
    end
    if op !== :(Base.:\)
        @eval ($op)(A::IntegratorParams, B::Number) = ($op).(A, B)
    end
end