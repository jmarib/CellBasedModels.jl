using MacroTools
using Symbolics

# --- normalize declared symbol forms (x, p.p1, p[i1], p[i1,i2]) -> valid Symbol + original Expr key
function _normalize_declared_symbol(symex)
    # plain Symbol
    if symex isa Symbol
        return (symex, symex)
    end

    # dotted: p.p1, q.r.s, ...
    if symex isa Expr && symex.head == :.
        parts = String[]
        cur = symex
        while cur isa Expr && cur.head == :.
            base = cur.args[1]
            field = cur.args[2]
            field_sym = field isa QuoteNode ? field.value : field
            pushfirst!(parts, string(field_sym))
            cur = base
        end
        base_sym = cur isa Symbol ? cur : Symbol(string(cur))
        valid = Symbol(string(base_sym), "__", join(parts, "__"))
        return (valid, symex)
    end

    # indexing: p[i1], p[i1,i2], ...
    if symex isa Expr && symex.head == :ref
        base = symex.args[1]
        idxs = symex.args[2:end]
        base_sym = base isa Symbol ? base : Symbol(string(base))
        idx_strs = map(i -> string(i), idxs)
        valid = Symbol(string(base_sym), "_", join(idx_strs, "_"), "_")
        return (valid, symex)
    end

    error("diffsym: unsupported symbol form in `symbols=`: $symex")
end

# collect LHS of assignments in a block
function _collect_assigned_lhs(ex)
    lhs = Any[]
    MacroTools.postwalk(ex) do node
        if node isa Expr && node.head == :(=)
            push!(lhs, node.args[1])
        end
        node
    end
    lhs
end

# replace nodes according to a Dict (Expr/Symbol keys)
_replace(ex, dict) = MacroTools.postwalk(ex) do node
    get(dict, node, node)
end

# ------------------------------------------------------------
# AST -> Symbolics interpreter (no eval)
# ------------------------------------------------------------

# resolve a called function symbol like :sqrt, :+, :*, :^, :-, :/
# without eval: only allow Base.<name> (extend if you want more)
function _resolve_base_fun(f::Symbol)
    isdefined(Base, f) || error("diffsym: unsupported function/operator `$f` in symbolic block (not in Base). This is most probably a custom function and symbolics cannot know its derivative, only functions from base are allowed (e.g. /, *, sqrt) inside the macro context.")
    return getfield(Base, f)
end

# Convert a Julia Expr into a Symbolics expression using an environment.
function _to_symbolics(ex, env::Dict{Symbol,Any})
    ex isa LineNumberNode && return nothing

    if ex isa Number
        return ex
    end

    if ex isa Symbol
        haskey(env, ex) || error("diffsym: symbol `$ex` not found in symbolic environment. Declare it in `symbols=` or define it in the block.")
        return env[ex]
    end

    if ex isa Expr
        if ex.head == :call
            f = ex.args[1]
            args = ex.args[2:end]

            if f isa Symbol
                fn = _resolve_base_fun(f)
                sargs = Any[]
                for a in args
                    sa = _to_symbolics(a, env)
                    sa === nothing && continue
                    push!(sargs, sa)
                end
                return fn(sargs...)
            else
                error("diffsym: unsupported call head `$f` in symbolic block (only Base symbols supported)")
            end

        elseif ex.head == :block
            last = nothing
            for stmt in ex.args
                val = _to_symbolics(stmt, env)
                val === nothing && continue
                last = val
            end
            return last
        else
            error("diffsym: unsupported expression head `$(ex.head)` in symbolic block")
        end
    end

    error("diffsym: unsupported node in symbolic block: $ex")
end

# Interpret a begin-block with assignments into env.
function _interpret_block(sym_block::Expr, declared_vars::Dict{Symbol,Any})
    env = Dict{Symbol,Any}(declared_vars)

    sym_block.head == :block || error("diffsym: expected a begin/end block")

    for stmt in sym_block.args
        stmt isa LineNumberNode && continue

        if stmt isa Expr && stmt.head == :(=)
            lhs = stmt.args[1]
            rhs = stmt.args[2]
            lhs isa Symbol || error("diffsym: only simple `name = expr` assignments supported in block for symbolic pass; got `$lhs`")
            env[lhs] = _to_symbolics(rhs, env)
        else
            _to_symbolics(stmt, env)
        end
    end

    return env
end

# Map a derivative "wrt" entry (Symbol or Expr like p.p1 / p[i]) to the internal valid Symbol.
function _wrt_to_valid(wrt, orig_to_valid::Dict{Any,Symbol}, valid_to_orig::Dict{Symbol,Any})
    if wrt isa Symbol
        haskey(valid_to_orig, wrt) || error("diffsym: differentiation variable `$wrt` must appear in `symbols=`")
        return wrt
    else
        haskey(orig_to_valid, wrt) || error("diffsym: differentiation variable `$wrt` must appear in `symbols=`")
        return orig_to_valid[wrt]
    end
end

# ------------------------------------------------------------
# Macro
# ------------------------------------------------------------
macro diffsym(block, args...)
    block isa Expr || error("diffsym: must pass a `begin ... end` block")

    # --- extract keywords
    symbols_ex = nothing
    derivs_ex  = nothing
    for a in args
        if a isa Expr && a.head == :(=) && a.args[1] == :symbols
            symbols_ex = a.args[2]
        elseif a isa Expr && a.head == :(=) && a.args[1] == :derivatives
            derivs_ex = a.args[2]
        else
            error("diffsym: expected `symbols=(...)` and `derivatives=(...)`")
        end
    end
    symbols_ex === nothing && error("diffsym: missing `symbols=(...)`")
    derivs_ex  === nothing && error("diffsym: missing `derivatives=(...)`")

    # --- validate symbols tuple
    (symbols_ex isa Expr && symbols_ex.head == :tuple) ||
        error("diffsym: `symbols=` must be a tuple, e.g. symbols=(x, y, p.p1)")

    # --- validate derivatives named tuple literal
    (derivs_ex isa Expr && derivs_ex.head == :tuple) ||
        error("diffsym: `derivatives=` must be a named tuple literal, e.g. derivatives=(dc_dx=(c,x),)")

    # --- build mappings for declared symbols
    declared = [_normalize_declared_symbol(s) for s in symbols_ex.args]  # (valid, original_expr)
    orig_to_valid = Dict{Any, Symbol}()   # :(p.p1) => :p__p1
    valid_to_orig = Dict{Symbol, Any}()   # :p__p1 => :(p.p1)

    for (v, o) in declared
        orig_to_valid[o] = v
        valid_to_orig[v] = o
    end

    # --- reject assignments to declared symbols inside the user block
    assigned = _collect_assigned_lhs(block)
    for L in assigned
        if haskey(orig_to_valid, L) || (L isa Symbol && haskey(valid_to_orig, L) && valid_to_orig[L] isa Symbol)
            error("diffsym: declared symbol `$L` is assigned inside the block; not allowed")
        end
    end

    # --- symbolic-pass rewrite: p.p1 -> p__p1, p[i] -> p_i_
    sym_block = _replace(block, orig_to_valid)

    # --- parse derivative requests
    # store (outname, f_sym, wrt_valid_sym)
    deriv_specs = Vector{Tuple{Symbol, Symbol, Symbol}}()
    for entry in derivs_ex.args
        (entry isa Expr && entry.head == :(=)) ||
            error("diffsym: each derivative entry must be like `dc_dx = (c, x)`")

        outname = entry.args[1]
        pair = entry.args[2]
        (pair isa Expr && pair.head == :tuple && length(pair.args) == 2) ||
            error("diffsym: `$outname` must be a 2-tuple `(f, x)`")

        f = pair.args[1]
        wrt = pair.args[2]

        (f isa Symbol) || error("diffsym: `$outname`: first element must be a Symbol (e.g. c)")

        wrt_valid = _wrt_to_valid(wrt, orig_to_valid, valid_to_orig)
        push!(deriv_specs, (outname, f, wrt_valid))
    end

    # ------------------------------------------------------------
    # NO EVAL: build Symbolics variables + interpret block into Symbolics IR
    # ------------------------------------------------------------
    declared_vars = Dict{Symbol,Any}()
    for (v, _) in declared
        declared_vars[v] = Symbolics.variable(v)
    end

    env = _interpret_block(sym_block, declared_vars)

    deriv_assign_exprs = Expr[]
    for (outname, f, wrt_valid) in deriv_specs
        haskey(env, f) || error("diffsym: requested derivative of `$f`, but `$f` is not defined in the block")

        fexpr = env[f]
        xvar  = declared_vars[wrt_valid]

        d = Symbolics.expand_derivatives(Symbolics.Differential(xvar)(fexpr))
        d = Symbolics.simplify(d)

        ex_d = Symbolics.toexpr(d)

        # substitute valid symbols back to original dotted/bracket expressions
        ex_d_orig = MacroTools.postwalk(ex_d) do node
            if node isa Symbol && haskey(valid_to_orig, node)
                valid_to_orig[node]
            else
                node
            end
        end

        push!(deriv_assign_exprs, :($outname = $ex_d_orig))
    end

    # --- final expansion: original user code + concrete derivative expressions
    return esc(quote
        begin
            $block
            $(deriv_assign_exprs...)
        end
    end)
end
