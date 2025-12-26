macro parameters(expr)
    # Extract parameter assignments from the expression
    assignments = []
    named_tuple_pairs = []
    
    # If expr is a block, process each line
    if isa(expr, Expr) && expr.head == :block
        for arg in expr.args
            if isa(arg, Expr) && arg.head == :(=) && length(arg.args) == 2
                # Extract variable name and value
                var_name = arg.args[1]
                var_value = arg.args[2]
                
                # Create individual assignment
                push!(assignments, :($var_name = $var_value))
                
                # Create named tuple pair
                push!(named_tuple_pairs, :($var_name = $var_name))
            elseif isa(arg, Expr) && arg.head == :(=) && length(arg.args) >= 2
                # Handle more complex assignments
                var_name = arg.args[1]
                var_value = length(arg.args) == 2 ? arg.args[2] : Expr(:tuple, arg.args[2:end]...)
                
                push!(assignments, :($var_name = $var_value))
                push!(named_tuple_pairs, :($var_name = $var_name))
            end
        end
    end
    
    # Create the named tuple expression
    named_tuple_expr = Expr(:tuple, named_tuple_pairs...)
    
    # Combine all assignments and return the named tuple
    result = quote
        $(assignments...)
        $named_tuple_expr
    end
    
    return esc(result)
end