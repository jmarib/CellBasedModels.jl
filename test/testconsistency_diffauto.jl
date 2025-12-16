using CellBasedModels
using CUDA
using ForwardDiff

f(x) = x^2

@testset verbose = verbose "Auxiliar - DiffSym" begin
    
    x = 2
    y = 4
    p = (p1=1,)

    @consistency_diffauto begin
        
        a = x ^ 2
        b = f(y)
        da_dx = 2*x

        c = a + b

        dc_dx = 2*x
        dc_dy = 2*y

    end derivatives=(dc_dx = (c, x), dc_dy = (c, y), da_dx = (a,x))

end