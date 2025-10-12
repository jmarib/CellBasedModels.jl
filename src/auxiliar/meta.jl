import MacroTools: @capture, postwalk, isexpr

function splitExpr(expr)
    if expr isa Expr && expr.head == :.
        lhs, rhs = expr.args
        return vcat(splitExpr(lhs), splitExpr(rhs))
    elseif expr isa Expr && expr.head == :ref
        base = expr.args[1]
        idxs = expr.args[2:end]
        if base isa Expr && base.head == :.
            left = splitExpr(base)
            left === nothing && return nothing
            lastfield = pop!(left)
            push!(left, (lastfield, idxs...))
            return left
        elseif base isa Symbol
            return [(base, idxs...)]
        else
            return nothing
        end
    elseif expr isa Symbol
        return [expr]
    else
        return nothing
    end
end

function hasParameter(code, target::Expr)
    env = Dict{Symbol,Any}()

    function resolve(x)
        if x isa Symbol
            seen = Set{Symbol}()
            while haskey(env, x) && !(x in seen)
                push!(seen, x)
                v = env[x]
                if v isa Symbol
                    x = v
                else
                    return v
                end
            end
            return x
        else
            return x
        end
    end

    function expand_node(node)
        if node isa Symbol
            return resolve(node)
        elseif node isa Expr && node.head == :.
            base, field = node.args
            return Expr(:., postwalk(expand_node, base), field)
        else
            return node
        end
    end

    function contains_target(node)
        node == target && return true
        node isa Expr || return false
        for a in node.args
            contains_target(a) && return true
        end
        return false
    end

    stmts = code isa Expr && code.head == :block ? code.args : [code]
    for st in stmts
        expanded = postwalk(expand_node, st)
        if contains_target(expanded)
            return true
        end
        if st isa Expr && @capture(st, lhs_ = rhs_)
            if lhs isa Symbol
                env[lhs] = postwalk(expand_node, rhs)
            end
        end
    end
    return false
end