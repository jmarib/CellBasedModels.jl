using MacroTools: @capture, postwalk

"""
    analyze_rule_code(fdef::Expr; tracked_syms, protected_syms=Symbol[])

Analyze the body of a function definition (AST) to detect assignments involving the
given tracked symbols. Supports simple alias propagation and invalidation.

Returns a NamedTuple with fields:
- `assigns`     :: Vector{String}  — list of tracked assignments
- `violations`  :: Vector{String}  — assignments that modify protected symbols
"""
function analyze_rule_code(mesh::Symbol, fdef::Expr; tracked_syms::Vector{Symbol}, protected_syms::Vector{Symbol}=Symbol[])
    @capture(fdef, function fname_(args__); body__ end) ||
        error("Expected a function definition expression")

    aliases = Dict{Symbol, Vector{Symbol}}()
    assigns = String[]
    violations = String[]

    postwalk(fdef) do ex
        if ex isa Expr && ex.head == :(=)
            lhs, rhs = ex.args

            # Case 1: aliasing
            if lhs isa Symbol && rhs isa Expr && rhs.head == :.
                chain = lhs_chain(rhs)
                if first(chain) in tracked_syms
                    aliases[lhs] = chain
                end

            # Case 2: reassignment invalidates alias
            elseif lhs isa Symbol
                delete!(aliases, lhs)
            end

        elseif ex isa Expr && ex.head in (:(=), :+=, :-=, :*=, :/=)
            lhs = ex.args[1]
            chain = lhs_chain(lhs)

            # resolve alias if any
            if first(chain) in keys(aliases)
                chain = vcat(aliases[first(chain)], chain[2:end])
            end

            # track or flag violation
            root = first(chain)
            if root in protected_syms
                push!(violations, join(chain, "."))
            elseif root in tracked_syms
                push!(assigns, join(chain, "."))
            end
        end
        ex
    end

    return (assigns = assigns, violations = violations)
end

function lhs_chain(lhs)
    if lhs isa Expr
        if lhs.head == :ref
            return lhs_chain(lhs.args[1])
        elseif lhs.head == :.
            return vcat(lhs_chain(lhs.args[1]), lhs_chain(lhs.args[2]))
        else
            return [lhs]
        end
    elseif lhs isa QuoteNode
        return [lhs.value]
    elseif lhs isa Symbol
        return [lhs]
    else
        return [lhs]
    end
end

macro addRule(mesh_name, fdef)
    # Verify the user passed a function definition
    @capture(fdef, function fname_(args__); body__ end) ||
        error("@addRule must be followed by a function definition of the form `function f!(uNew, u, p) ... end`")

    arg_syms = [a isa Expr && a.head == :(::) ? a.args[1] : a for a in args]
    nargs = length(arg_syms)
    nargs == 3 || error("Function must have three arguments (uNew, u, p).")

    # First two arguments are tracked (uNew)
    tracked_syms = [arg_syms[1]]

    # Protected symbols — here, second argument (u) cannot be modified
    protected_syms = [arg_syms[2]]

    analyze_rule_code(mesh_name, fdef; tracked_syms=tracked_syms, protected_syms=protected_syms)
end

macro addODE(mesh_name, fdef)
    # Verify the user passed a function definition
    @capture(fdef, function fname_(args__); body__ end) ||
        error("@addRule must be followed by a function definition of the form `function f!(uNew, u, p, t) ... end`")

    arg_syms = [a isa Expr && a.head == :(::) ? a.args[1] : a for a in args]
    nargs = length(arg_syms)
    nargs == 4 || error("Function must have four arguments (du, u, p, t)")

    # First two arguments are tracked (uNew)
    tracked_syms = [arg_syms[1]]

    # Protected symbols — here, second argument (u) cannot be modified
    protected_syms = [arg_syms[2]]

    analyze_rule_code(mesh_name, fdef; tracked_syms=tracked_syms, protected_syms=protected_syms)
end

macro addSDE(mesh_name, fdef, gdef)
    # Verify the user passed a function definition
    @capture(fdef, function fname_(args__); body__ end) ||
        error("@addRule must be followed by a first function definition of the form `function f!(uNew, u, p, t) ... end`")
    @capture(gdef, function fname_(args__); body__ end) ||
        error("@addRule must be followed by a second function definition of the form `function g!(uNew, u, p, t) ... end`")

    arg_syms = [a isa Expr && a.head == :(::) ? a.args[1] : a for a in args]
    nargs = length(arg_syms)
    nargs == 4 || error("Function must have four arguments (du, u, p, t)")

    # First two arguments are tracked (uNew)
    tracked_syms = [arg_syms[1]]

    # Protected symbols — here, second argument (u) cannot be modified
    protected_syms = [arg_syms[2]]

    analyze_rule_code(mesh_name, fdef; tracked_syms=tracked_syms, protected_syms=protected_syms)
end

