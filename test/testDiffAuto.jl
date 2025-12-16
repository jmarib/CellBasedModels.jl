using CellBasedModels
using CUDA
using ForwardDiff

f(x) = x^2

@testset verbose = verbose "Auxiliar - DiffSym" begin
    
    x = 2
    y = 4
    p = (p1=1,)

    @diffauto begin
        
        a = x ^ 2
        b = f(y)

        c = a + b

    end derivatives=(dc_dx = (c, x), dc_dy = (c, y))

    @test dc_dx == 2 * x
    @test dc_dy == 2 * y

    @diffauto begin
        
        a = x ^ 2
        b = y ^ 2

        c = a * b * p.p1

    end derivatives=(dc_dp1 = (c, p.p1),)

    @test dc_dp1 == x^2 * y ^2

end