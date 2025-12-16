using MacroTools
using ForwardDiff

# --------------------------
# Normalization utilities
# --------------------------

# (x, p.p1, p[i1]) -> (valid_symbol, original_expr_key)
function _normalize_declared_symbol_auto(symex)
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

    error("diffauto: unsupported symbol form: $symex")
end

_replace_auto(ex, dict) = MacroTools.postwalk(ex) do node
    get(dict, node, node)
end

# Collect plain-symbol LHS of assignments (x = ...)
function _collect_assigned_lhs_syms_auto(ex)
    lhs = Symbol[]
    MacroTools.postwalk(ex) do node
        if node isa Expr && node.head == :(=)
            L = node.args[1]
            L isa Symbol && push!(lhs, L)
        end
        node
    end
    lhs
end

# Infer free variables from the block:
# - Symbols that appear but are never assigned
# - Treat dotted/ref as atomic variables (p.p1, p[i])
# - Do NOT treat call-head symbols as variables (f in f(x) is ignored)
function _infer_free_symbols_from_block_auto(block::Expr)
    assigned = Set{Symbol}(_collect_assigned_lhs_syms_auto(block))
    seen = Set{Any}()
    ordered = Any[]  # Symbol or Expr(:.) or Expr(:ref)

    add(x) = (x in seen) ? nothing : (push!(ordered, x); push!(seen, x); nothing)

    function walk(ex)
        ex isa LineNumberNode && return
        ex isa Number && return

        if ex isa Symbol
            (ex in assigned) || add(ex)
            return
        end

        if ex isa Expr
            if ex.head == :. || ex.head == :ref
                add(ex)
                return
            end
            if ex.head == :(=)
                walk(ex.args[2]) # RHS only
                return
            end
            if ex.head == :call
                # skip function position ex.args[1]
                for a in ex.args[2:end]
                    walk(a)
                end
                return
            end
            if ex.head == :block
                for s in ex.args
                    walk(s)
                end
                return
            end
            for a in ex.args
                walk(a)
            end
        end
    end

    walk(block)
    # drop any assigned plain symbols defensively
    filter(x -> !(x isa Symbol && x in assigned), ordered)
end

# Map wrt (Symbol or Expr p.p1/p[i]) to internal valid symbol
function _wrt_to_valid_auto(wrt, orig_to_valid::Dict{Any,Symbol}, valid_to_orig::Dict{Symbol,Any})
    if wrt isa Symbol
        haskey(valid_to_orig, wrt) || error("diffauto: differentiation variable `$wrt` must be inferable or declared")
        return wrt
    else
        haskey(orig_to_valid, wrt) || error("diffauto: differentiation variable `$wrt` must be inferable or declared")
        return orig_to_valid[wrt]
    end
end

# --------------------------
# @diffauto
# --------------------------
macro diffauto(block, args...)
    block isa Expr || error("diffauto: must pass a `begin ... end` block")

    symbols_ex = nothing  # optional
    derivs_ex  = nothing  # required

    for a in args
        if a isa Expr && a.head == :(=) && a.args[1] == :symbols
            symbols_ex = a.args[2]
        elseif a isa Expr && a.head == :(=) && a.args[1] == :derivatives
            derivs_ex = a.args[2]
        else
            error("diffauto: expected `derivatives=(...)` and optionally `symbols=(...)`")
        end
    end
    derivs_ex === nothing && error("diffauto: missing `derivatives=(...)`")
    (derivs_ex isa Expr && derivs_ex.head == :tuple) ||
        error("diffauto: `derivatives=` must be a named tuple literal, e.g. derivatives=(dc_dx=(c,x),)")

    # infer symbols if not provided
    if symbols_ex === nothing
        inferred = _infer_free_symbols_from_block_auto(block)
        # ensure wrt variables are included even if not used
        for entry in derivs_ex.args
            (entry isa Expr && entry.head == :(=)) || error("diffauto: derivative entries must be `name=(f,x)`")
            pair = entry.args[2]
            (pair isa Expr && pair.head == :tuple && length(pair.args) == 2) ||
                error("diffauto: `$(entry.args[1])` must be `(f, x)`")
            wrt = pair.args[2]
            (wrt in inferred) || push!(inferred, wrt)
        end
        symbols_ex = Expr(:tuple, inferred...)
    end

    (symbols_ex isa Expr && symbols_ex.head == :tuple) ||
        error("diffauto: `symbols=` must be a tuple if provided")

    # build normalization maps
    declared = [_normalize_declared_symbol_auto(s) for s in symbols_ex.args]  # (valid, original)
    orig_to_valid = Dict{Any,Symbol}()
    valid_to_orig = Dict{Symbol,Any}()
    for (v, o) in declared
        orig_to_valid[o] = v
        valid_to_orig[v] = o
    end

    # reject assignments to inferred/declared *plain* symbols (function parameters)
    assigned_syms = Set{Symbol}(_collect_assigned_lhs_syms_auto(block))
    for (v, o) in declared
        if v isa Symbol && (v in assigned_syms)
            error("diffauto: free variable `$v` is assigned inside the block; not allowed")
        end
        # if user explicitly declared a dotted/ref symbol and assigns to it: also forbid
        if !(o isa Symbol)
            # check exact LHS nodes
            found = false
            MacroTools.postwalk(block) do node
                if node isa Expr && node.head == :(=) && node.args[1] == o
                    found = true
                end
                node
            end
            found && error("diffauto: declared symbol `$o` is assigned inside the block; not allowed")
        end
    end

    # rewrite dotted/ref uses in the wrapper function only
    block_for_fun = _replace_auto(block, orig_to_valid)

    # parse derivatives: (outname, f_sym, wrt_valid_sym)
    deriv_specs = Vector{Tuple{Symbol,Symbol,Symbol}}()
    for entry in derivs_ex.args
        (entry isa Expr && entry.head == :(=)) || error("diffauto: derivative entries must be `name=(f,x)`")
        outname = entry.args[1]
        pair = entry.args[2]
        (pair isa Expr && pair.head == :tuple && length(pair.args) == 2) ||
            error("diffauto: `$outname` must be a 2-tuple `(f, x)`")
        f = pair.args[1]
        wrt = pair.args[2]
        (f isa Symbol) || error("diffauto: `$outname`: first element must be a Symbol (e.g. c)")
        wrt_valid = _wrt_to_valid_auto(wrt, orig_to_valid, valid_to_orig)
        push!(deriv_specs, (outname, f, wrt_valid))
    end

    # function signature: all valid symbols as args (x, y, p__p1, p_i_, ...)
    valid_args = [v for (v, _) in declared]

    # in the wrapper, return the requested target `f` (per derivative spec we may need multiple)
    # easiest: generate one wrapper per target variable to keep it simple and robust.
    # We'll generate wrappers lazily per (f_sym).
    unique_targets = unique(t[2] for t in deriv_specs)

    wrappers = Expr[]
    wrapper_names = Dict{Symbol,Symbol}()

    for f_sym in unique_targets
        fname = gensym(Symbol("__diffauto_f_", f_sym, "__"))
        wrapper_names[f_sym] = fname

        # function body = rewritten block + return f_sym
        fbody = Expr(:block, block_for_fun.args...)  # assumes :block
        push!(fbody.args, :(return $f_sym))

        push!(wrappers, :(local function $fname($(valid_args...))
            $fbody
        end))
    end

    # build derivative assignments using ForwardDiff.derivative on 1D closures
    deriv_assigns = Expr[]
    for (outname, f_sym, wrt_valid) in deriv_specs
        fname = wrapper_names[f_sym]

        # call arguments: for each arg in valid_args, use its current runtime value.
        # - if arg corresponds to normalized dotted/ref, we must pass the original expression value (p.p1, p[i], ...)
        # - if plain symbol, pass itself
        base_call_args = Any[]
        for a in valid_args
            if haskey(valid_to_orig, a) && !(valid_to_orig[a] isa Symbol)
                push!(base_call_args, valid_to_orig[a])  # e.g. p.p1
            else
                push!(base_call_args, a)                 # e.g. x
            end
        end

        # build call with one position replaced by t
        call_args_t = copy(base_call_args)
        # find index of wrt_valid in valid_args
        idx = findfirst(==(wrt_valid), valid_args)
        idx === nothing && error("diffauto: internal error: wrt var not in args")
        call_args_t[idx] = :t

        deriv_expr = :(ForwardDiff.derivative(t -> $fname($(call_args_t...)), $(base_call_args[idx])))

        push!(deriv_assigns, :($outname = $deriv_expr))
    end

    return esc(quote
        begin
            # run original user code normally
            $block

            # runtime wrappers for AD (not CUDA-safe)
            $(wrappers...)

            # ForwardDiff derivatives
            $(deriv_assigns...)
        end
    end)
end
