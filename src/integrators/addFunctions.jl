using MacroTools: @capture, postwalk

"""
    analyze_rule_code(fdef::Expr; tracked_syms, protected_syms=Symbol[])

Analyze the body of a function definition (AST) to detect assignments involving the
given tracked symbols. Supports simple alias propagation and invalidation.

Returns a NamedTuple with fields:
- `assigns`     :: Vector{String}  — list of tracked assignments
- `violations`  :: Vector{String}  — assignments that modify protected symbols
"""
function lhs_chain_with_index(lhs)
    indexed = false
    function go(x)
        if x isa Expr
            if x.head == :ref
                indexed = true
                return go(x.args[1])
            elseif x.head == :.
                return vcat(go(x.args[1]), go(x.args[2]))
            else
                return [x]
            end
        elseif x isa QuoteNode
            return [x.value]
        elseif x isa Symbol
            return [x]
        else
            return [x]
        end
    end
    return go(lhs), indexed
end

# Keep for callers that still want just the chain (your other helpers may use it)
lhs_chain(lhs) = lhs_chain_with_index(lhs)[1]

function analyze_rule_code(kwargs, fdefs; type)
    assigns = Tuple[]

    # Heads we treat differently
    normal_heads     = (:(=), :+=, :-=, :*=, :/=)
    broadcast_heads  = (:.=, :.+=, :.-=, :.*=, :./=)

    for f in fdefs
        fdef = f.fdef

        # tracked: first arg (du), second (u) is protected
        tracked_syms   = f.args[1:2]
        du_sym         = tracked_syms[1]
        u_sym          = tracked_syms[2]
        protected_syms = [u_sym]

        @capture(fdef, function fname_(args__); body__ end) ||
            error("Expected a function definition expression")

        aliases = Dict{Symbol, Vector{Symbol}}()

        postwalk(fdef) do ex
            # Maintain alias map: x = du.a.a  (adds),  x = something_else (removes)
            if ex isa Expr && ex.head == :(=)
                lhs, rhs = ex.args
                if lhs isa Symbol && rhs isa Expr && rhs.head == :.
                    chain = lhs_chain(rhs)
                    if first(chain) in tracked_syms
                        aliases[lhs] = chain
                    end
                elseif lhs isa Symbol
                    delete!(aliases, lhs)
                end

            # Handle *all* assignment-like expressions (incl. broadcast)
            elseif ex isa Expr && (ex.head in (normal_heads..., broadcast_heads...))
                lhs = ex.args[1]
                chain, indexed = lhs_chain_with_index(lhs)

                # Resolve simple alias for the root
                if first(chain) in keys(aliases)
                    chain = vcat(aliases[first(chain)], chain[2:end])
                end

                root = first(chain)
                tail = chain[2:end]

                # 1) Any write to protected root (u) is forbidden
                if root in protected_syms
                    error("Assignment to protected symbol $(join(chain, '.')) is forbidden.")

                # 2) Broadcast writes on tracked state are forbidden
                elseif root == du_sym && (ex.head in broadcast_heads)
                    error("Broadcast assignment on tracked state is forbidden: $(join(chain, '.')) $(ex.head). Use explicit indexed updates (e.g. $(du_sym).scope.param[i] = ...).")

                # 3) Plain writes without indexing to vector-like fields are forbidden
                #    We consider du.* fields vector-like and require indexing.
                elseif root == du_sym
                    # direct write to `du` itself is also invalid
                    if length(chain) == 1
                        error("Direct assignment to $(du_sym) is forbidden; modify its fields via indexing.")
                    end
                    if !indexed
                        error("Assignment to vector-like field without indexing: $(join(chain, '.')). Use indexing (e.g. $(du_sym).scope.param[i] = ...).")
                    end

                    # Valid tracked indexed write -> record (tail...) as before
                    if length(tail) < 2
                        error("Tracked assignment too short: $(join(chain,'.')). Expected $(du_sym).SCOPE.PARAM[...]")
                    end
                    push!(assigns, tuple(tail...))
                end
            end
            ex
        end
    end

    unique_assigns = unique(assigns)
    println("Detected tracked (indexed) assignments: ", unique_assigns)

    # build emitted code (unchanged structure)
    fs = [f.fname for f in fdefs]
    assigns_code = :(CellBasedModels.addFunction!($(kwargs.mesh_name),
                                                 $(QuoteNode(type)),
                                                 $(QuoteNode(kwargs.scope_name)),
                                                 $unique_assigns,
                                                 ($(fs...),)))

    functions_code = [quote
        function $(f.fname)($(f.args...))
            $(f.fdef)
            $(f.fname)($(f.args...))
        end
    end for f in fdefs]

    quote
        $(functions_code...)
        $assigns_code
    end
end

# function lhs_chain(lhs)
#     if lhs isa Expr
#         if lhs.head == :ref
#             return lhs_chain(lhs.args[1])
#         elseif lhs.head == :.
#             return vcat(lhs_chain(lhs.args[1]), lhs_chain(lhs.args[2]))
#         else
#             return [lhs]
#         end
#     elseif lhs isa QuoteNode
#         return [lhs.value]
#     elseif lhs isa Symbol
#         return [lhs]
#     else
#         return [lhs]
#     end
# end

function extract_parameters(n, ex)

    functions = []
    for i in n-1:-1:0
        fdef = ex[end - i]
        @capture(fdef, function fname_(args__); body__ end) ||
            error("Last $n arguments of @addX macro should be followed by a function definition of the form `function f!(uNew, u, p, t) ... end`. Found: $fdef")

        arg_syms = [a isa Expr && a.head == :(::) ? a.args[1] : a for a in args]
        nargs = length(arg_syms)
        nargs == 4 || error("Function must have four arguments (uNew, u, p, t).")

        push!(functions, (fname=fname, args=arg_syms, fdef=fdef, body=body))
    end

    mesh_name = nothing
    scope_name = nothing
    overwrite = false
    for arg in ex[1:end - n]
        @capture(arg, kwarg_=value_)
        if kwarg == :model
           mesh_name = value
        elseif kwarg == :scope
           scope_name = value
        elseif kwarg == :overwrite
            overwrite = value
        else
            error("Unknown keyword argument '$kwarg'")
        end
    end

    if mesh_name === nothing
        error("Expected 'model' keyword argument.")
    elseif scope_name === nothing
        error("Expected 'scope' keyword argument.")
    elseif !(overwrite isa Bool)
        error("'overwrite' keyword argument must be a Bool.")
    end

    return (mesh_name=mesh_name, scope_name=scope_name, overwrite=overwrite), functions

end

macro addRule(ex...)

    kwargs, functions = extract_parameters(1, ex)
    code = analyze_rule_code(kwargs, functions; type=:RULE)

    return esc(code)
end

macro addODE(ex...)

    kwargs, functions = extract_parameters(1, ex)
    code = analyze_rule_code(kwargs, functions; type=:ODE)

    return esc(code)

end

macro addSDE(ex...)
    kwargs, functions = extract_parameters(2, ex)
    code = analyze_rule_code(kwargs, functions; type=:SDE)

    return esc(code)
end