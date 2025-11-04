import DifferentialEquations: ODEProblem, SDEProblem
import DifferentialEquations

struct CBProblem

    mesh
    u0
    tspan
    p
    u
    _DEProblems

end

function CBProblem(mesh::CellBasedModels.AbstractMesh, meshObject0::CellBasedModels.AbstractMeshObject, tspan::Tuple{T,T}, p::Tuple=tuple()) where T<:Real

    u0 = meshObject0
    u = deepcopy(u0)

    DEProblemsDict = Dict{Symbol, Any}()
    for scope in keys(mesh._functions)
        args = modifiedInScope(mesh, scope)
        type, fs = mesh._functions[scope]

        println(args)
        if type == :RULE
            DEProblemsDict[scope] = RuleProblem(fs[1], partialCopy(u, args), tspan, p)
        elseif type == :ODE
            DEProblemsDict[scope] = ODEProblem(fs[1], partialCopy(u, args), tspan, p)
        elseif type == :SDE
            DEProblemsDict[scope] = SDEProblem(fs[1], fs[2], partialCopy(u, args), tspan, p)
        else
            error("Unknown function type: $type")
        end
    end

    return CBProblem(mesh, u0, tspan, p, u, DEProblemsDict)

end

mutable struct CBIntegrator

    u
    integratorsDict::Dict{Symbol, Any}
    
end

function DifferentialEquations.init(problem::CBProblem; dt::Real)

    # kwargs_ = deepcopy(kwargs)

    # for (k, v) in pairs(kwargs_)
        
    #     if !(k in keys(problem._DEProblems))
    #        error("Unknown scope $k for CBProblem. Available scopes: $(keys(problem._DEProblems))")
    #     end

    # end

    integratorsDict = Dict{Symbol, Any}()
    for (scope, deproblem) in problem._DEProblems
        if typeof(deproblem) == RuleProblem
            integratorsDict[scope] = DifferentialEquations.init(deproblem; dt=dt)
        else
            integratorsDict[scope] = DifferentialEquations.init(deproblem, DifferentialEquations.Euler(); dt=dt)
        end
    end

    return CBIntegrator(problem.u, integratorsDict)

end

function DifferentialEquations.step!(integrator::CBIntegrator)

    for (scope, deintegrator) in integrator.integratorsDict
        if typeof(deintegrator) != Rule
            DifferentialEquations.step!(deintegrator)
        end
    end

    for (scope, deintegrator) in integrator.integratorsDict
        if typeof(deintegrator) == Rule
            DifferentialEquations.step!(deintegrator)
        end
    end

    for (scope, deintegrator) in integrator.integratorsDict
        if scope == :biochemistry
            println(deintegrator)
            println(deintegrator.u.n.x)
        end
        copyto!(integrator.u, deintegrator.u)
    end

    return nothing

end