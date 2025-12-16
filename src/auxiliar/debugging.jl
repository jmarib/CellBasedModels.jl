using ForwardDiff
using Printf

macro debugAutodiff(expr, pairs...)
    quote
        # First run: execute the expression to get all variables computed
        $(esc(expr))
        
        # Define tolerance for "approximately zero"
        local tolerance = 1e-10
        
        # Print table header
        @printf("┌%s┬%s┬%s┬%s┐\n", "─"^15, "─"^15, "─"^15, "─"^15)
        @printf("│%15s│%15s│%15s│%15s│\n", "Variable", "Analytic", "Autodiff", "Difference")
        @printf("├%s┼%s┼%s┼%s┤\n", "─"^15, "─"^15, "─"^15, "─"^15)
        
        # Process each variable-derivative pair
        $(map(pairs) do pair
            if isa(pair, Expr) && pair.head == :tuple && length(pair.args) == 2
                deriv_var = pair.args[1]
                result_var = pair.args[2]
                
                # Create modified expression with the variable definition commented out
                modified_expr = filter_assignments(expr, deriv_var)
                
                quote
                    # Capture the analytic value before any ForwardDiff computation
                    local analytic_value = $(esc(result_var))
                    
                    # Create a function that wraps the expression for this variable
                    local autodiff_func = function(var)
                        # Replace the derivative variable with the input
                        local $(esc(deriv_var)) = var
                        # Evaluate the modified expression (with variable definition filtered out)
                        $(esc(modified_expr))
                    end
                    
                    # Get the current value of the derivative variable
                    local current_value = $(esc(deriv_var))
                    
                    # Compute and display the derivative comparison
                    local autodiff_result = ForwardDiff.derivative(autodiff_func, ForwardDiff.value(current_value))
                    
                    # Extract numerical values from any dual numbers
                    local analytic_num = ForwardDiff.value(analytic_value)
                    local autodiff_num = ForwardDiff.value(autodiff_result)
                    local diff = abs(autodiff_num - analytic_num)
                    
                    # Color the difference based on tolerance and zero values
                    local diff_color = if analytic_num == 0.0 || autodiff_num == 0.0
                        "\033[33m"  # Yellow for warning (possible missing derivative)
                    elseif diff > tolerance
                        "\033[31m"  # Red for significant differences
                    else
                        "\033[32m"  # Green for good matches
                    end
                    
                    @printf("│%15s│%15.6g│%15.6g│%s%15.6g\033[0m│\n", 
                            $(string(deriv_var)), analytic_num, autodiff_num, diff_color, diff)
                end
            else
                error("Each pair must be in the form (variable, derivative_result)")
            end
        end...)
        
        # Print table footer
        @printf("└%s┴%s┴%s┴%s┘\n", "─"^15, "─"^15, "─"^15, "─"^15)
        
        # Final run: re-execute the expression cleanly to reset ALL intermediate variables
        $(esc(expr))
    end
end

# Helper function to filter out assignment statements for a specific variable
function filter_assignments(expr, target_var)
    if isa(expr, Expr) && expr.head == :block
        # Filter out assignment statements that define the target variable
        filtered_args = []
        for arg in expr.args
            if isa(arg, Expr) && arg.head == :(=) && length(arg.args) >= 2
                # Check if this assignment defines our target variable
                if arg.args[1] == target_var
                    # Skip this assignment (effectively commenting it out)
                    continue
                end
            end
            # Keep other statements and recursively filter nested blocks
            if isa(arg, Expr) && arg.head == :block
                push!(filtered_args, filter_assignments(arg, target_var))
            else
                push!(filtered_args, arg)
            end
        end
        return Expr(:block, filtered_args...)
    else
        return expr
    end
end