# module DiffSymMacro

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
    deriv_specs = Vector{Tuple{Symbol, Symbol, Symbol}}() # (outname, f_sym, x_sym)
    for entry in derivs_ex.args
        (entry isa Expr && entry.head == :(=)) ||
            error("diffsym: each derivative entry must be like `dc_dx = (c, x)`")

        outname = entry.args[1]
        pair = entry.args[2]
        (pair isa Expr && pair.head == :tuple && length(pair.args) == 2) ||
            error("diffsym: `$outname` must be a 2-tuple `(f, x)`")

        f = pair.args[1]
        x = pair.args[2]
        (f isa Symbol) || error("diffsym: `$outname`: first element must be a Symbol (e.g. c)")
        (x isa Symbol) || error("diffsym: `$outname`: second element must be a Symbol (e.g. x)")

        # x must be a declared symbol (after normalization it’s still the same for plain x,y)
        haskey(valid_to_orig, x) || error("diffsym: `$outname`: `$x` must appear in `symbols=`")

        push!(deriv_specs, (outname, f, x))
    end

    # --- build a macro-expansion-time eval that returns a NamedTuple of symbolic derivatives
    valid_syms = [v for (v, _) in declared]
    symvars_decl = Expr(:macrocall, Symbolics.var"@variables", __source__)
    append!(symvars_decl.args, valid_syms)

    # compute derivatives using Differential(x)(f) (Symbolics doc API) :contentReference[oaicite:2]{index=2}
    # return as NamedTuple to pull out easily
    returned_fields = Any[]
    for (outname, f, x) in deriv_specs
        push!(returned_fields,
            Expr(:(=), outname,
                :(Symbolics.simplify(Symbolics.expand_derivatives(Symbolics.Differential($x)($f))))
            )
        )
    end
    namedtuple_expr = Expr(:tuple, returned_fields...)

    eval_expr = quote
        let
            $symvars_decl
            $sym_block
            $namedtuple_expr
        end
    end

    deriv_nt = Base.eval(__module__, eval_expr)  # NamedTuple of Symbolics expressions

    # --- convert symbolic derivatives to Expr (Symbolics.toexpr documented) :contentReference[oaicite:3]{index=3}
    deriv_assign_exprs = Expr[]
    for (outname, _, _) in deriv_specs
        sym_d = getproperty(deriv_nt, outname)
        ex_d  = Symbolics.toexpr(sym_d)

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

# end # module
