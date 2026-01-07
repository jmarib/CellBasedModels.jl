@testset verbose=verbose "specializations - AgentPoint" begin

    #Define the model
    model = AgentPoint(
        (
            a = AbstractFloat, 
            a2 = AbstractFloat,
            b = Integer
        ),
    )

    #v1
    @addODE model=model scope=globalEvolution function globalEvolution!(du, u, p, t)
        du.g.a[1] = -0.1 * u.g.a[1]
    end

    #v2
    @addODE model=model scope=globalEvolution2 function globalEvolution2!(du, u, p, t)
        @. du.g.a2 = -0.1 * u.g.a2
    end

    @addRule model=model scope=globalReset function globalReset!(uNew, u, p, t)
        uNew.g.b[1] += 2
        if uNew.g.b[1] > 10
            uNew.g.b[1] = 0
        end
    end

    println(model)

    #Initialize the object
    obj = createObject(model)
    obj.g.a .= 1.0
    obj.g.a2 .= 1.0
    obj.g.b .= 0

    #Define the problem
    problem = CBProblem(
        model,
        obj
    )

    integrator = init(problem, dt=0.1)

    for i in 1:100
        step!(integrator)
        @test integrator.u.g.a[1] ≈ exp(-0.1 * integrator.t)
        @test integrator.u.g.a2[1] ≈ exp(-0.1 * integrator.t)
        @test integrator.u.g.b[1] ≤ 10
    end

end