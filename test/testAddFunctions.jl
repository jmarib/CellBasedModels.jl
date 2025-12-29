using BenchmarkTools
using CUDA
using DifferentialEquations
using Adapt
import StaticArrays: SizedVector
using MacroTools: @capture, postwalk

@testset verbose = verbose "ABM - AddFunctions" begin

    @testset "Capturing patterns" begin
        props = (a = Float64, b = Int)
        mesh = UnstructuredMesh(2; n=Node(props))

        # Direct assignment to protected symbol (u)
        kwargs, functions = CellBasedModels.extract_parameters(1,
            ( 
                :(model = mesh),
                :(scope = :mechanics),
                :(
                    function rule_bad1!(du, u, p, t)
                        du = 5
                    end 
                )
            )
        )
        @test_throws ErrorException CellBasedModels.analyze_rule_code(kwargs, functions; type=:RULE)

        # Assignment to protected symbol (u)
        kwargs, functions = CellBasedModels.extract_parameters(1,
            ( 
                :(model = mesh),
                :(scope = :mechanics),
                :(
                    function rule_bad2!(du, u, p, t)
                        u.a.a[1] = 1.0
                    end
                ),
            )
        )
        # @test_logs :warn CellBasedModels.analyze_rule_code(kwargs, functions; type=:RULE)        # Test that it runs without error and produces a warning
        @test_nowarn CellBasedModels.analyze_rule_code(kwargs, functions; type=:RULE)

        # Direct assignment to tracked symbol (du)
        kwargs, functions = CellBasedModels.extract_parameters(1,
            ( 
                :(model = mesh),
                :(scope = :mechanics),
                :(
                    function rule_bad3!(du, u, p, t)
                        du = zeros(10)
                    end
                ),
            )
        )
        @test_throws ErrorException CellBasedModels.analyze_rule_code(kwargs, functions; type=:RULE)

        # Valid indexed assignment when not broadcasting
        kwargs, functions = CellBasedModels.extract_parameters(1,
            ( 
                :(model = mesh),
                :(scope = :mechanics),
                :(
                    function rule_good!(du, u, p, t)
                        for i in 1:length(u.a.a)
                            du.a.a[i] = 1.0
                        end
                    end
                ),
            )
        )
        @test_nowarn CellBasedModels.analyze_rule_code(kwargs, functions; type=:RULE)

        # Valid broadcast assignment when broadcasting=true
        kwargs, functions = CellBasedModels.extract_parameters(1,
            ( 
                :(model = mesh),
                :(scope = :mechanics),
                :(
                    function rule_good!(du, u, p, t)
                        du.a.a .= 1.0
                    end
                ),
            )
        )
        @test_nowarn CellBasedModels.analyze_rule_code(kwargs, functions; type=:RULE)

        # Valid alias usage
        kwargs, functions = CellBasedModels.extract_parameters(1,
            ( 
                :(model = mesh),
                :(scope = :mechanics),
                :(
                    function rule_alias1!(du, u, p, t)
                        x = du.a.a
                        x[1] = 1.0
                    end
                ),
            )
        )
        @test_nowarn CellBasedModels.analyze_rule_code(kwargs, functions; type=:RULE)

        # Alias reassignment
        kwargs, functions = CellBasedModels.extract_parameters(1,
            ( 
                :(model = mesh),
                :(scope = :mechanics),
                :(
                    function rule_alias2!(du, u, p, t)
                        x = du.a.a
                        x = u.a.b  # Should invalidate alias
                        # No assignment using x after this
                    end
                ),
            )
        )
        @test_nowarn CellBasedModels.analyze_rule_code(kwargs, functions; type=:RULE)

        # Protected symbol through alias
        kwargs, functions = CellBasedModels.extract_parameters(1,
            ( 
                :(model = mesh),
                :(scope = :mechanics),
                :(
                    function rule_alias_bad1!(du, u, p, t)
                        x = u.a.a
                        x[1] = 1.0
                    end
                ),
            )
        )
        @test_nowarn CellBasedModels.analyze_rule_code(kwargs, functions; type=:RULE)

    end

    @testset "Complex assignment patterns" begin

        kwargs, functions = CellBasedModels.extract_parameters(1,
            ( 
                :(model = mesh),
                :(scope = :mechanics),
                :(
                    function rule_alias_bad1!(du, u, p, t)
                        x = du.a
                        x.a[1] = 1.0
                        x = 9
                        x.b[1] = 2
                        y = du
                        z = y.a
                        z.c[1] = 2
                        for i in 1:10
                            du.a.d[i] = 3.0
                        end
                    end
                ),
            )
        )
        code = CellBasedModels.analyze_rule_code(kwargs, functions; type=:RULE)
        @test occursin("Tuple[(:a, :a), (:a, :c), (:a, :d)]", "$code")

        kwargs, functions = CellBasedModels.extract_parameters(1,
            ( 
                :(model = mesh),
                :(scope = :mechanics),
                :(
                    function rule_alias_bad1!(du, u, p, t)
                        @kernel function f(du, u, p, t)
                            x = du.a
                            x.a[1] = 1.0
                            x = 9
                            x.b[1] = 2
                            y = du
                            z = y.a
                            z.c[1] = 2
                            for i in 1:10
                                du.a.d[i] = 3.0
                            end
                        end

                        backend = get_backend(du)
                        mul2_kernel(backend, 64)(A, ndrange=size(du))
                        synchronize(backend)
                    end
                ),
            )
        )
        code = CellBasedModels.analyze_rule_code(kwargs, functions; type=:RULE)
        @test occursin("Tuple[(:a, :a), (:a, :c), (:a, :d)]", "$code")


    end

    #######################################################################
    # Macro tests: @addRule, @addODE, @addSDE
    #######################################################################

    @testset "@addRule macro" begin

        @testset "@addRule - basic functionality" begin
            props = (a = Float64, b = Int)

            mesh = UnstructuredMesh(2; n=Node(props))
            # Valid rule without broadcasting
            @test_nowarn @addRule(model=mesh, scope=mechanics,
                function rule1!(du, u, p, t)
                    for i in 1:length(u.n)
                        du.n.a[i] = u.n.a[i] + 1.0
                        du.n.b[i] = u.n.b[i] + 1
                    end
                end
            )
            # Valid rule with broadcasting
            @test_nowarn @addRule(model=mesh, scope=mechanics2,
                function rule2!(du, u, p, t)
                    du.n.a .= u.n.a .+ 1.0
                    du.n.b .= u.n.b .+ 1
                end
            )
            # Test that functions are properly added to mesh
            @test haskey(mesh._functions, :mechanics)
            @test mesh._functions[:mechanics][1] === :RULE
            @test mesh._functions[:mechanics][2] isa Tuple{<:Function}
        end

        @testset "@addRule - parameter validation" begin
            props = (a = Float64, b = Int)

            mesh = UnstructuredMesh(2; n=Node(props))
            # Missing model parameter
            @test_throws LoadError @eval @addRule(scope=mechanics,
                function rule_bad1!(du, u, p, t)
                    du.n.a[1] = 1.0
                end
            )
            # Missing scope parameter
            @test_throws LoadError @eval @addRule(model=mesh,
                function rule_bad2!(du, u, p, t)
                    du.n.a[1] = 1.0
                end
            )
            # Wrong number of function arguments
            @test_throws LoadError @eval @addRule(model=mesh, scope=mechanics,
                function rule_bad4!(du, u, p)
                    du.n.a[1] = 1.0
                end
            )
        end

    end

    @testset "@addODE macro" begin

        @testset "@addODE - basic functionality" begin
            props = (a = Float64, b=Float64)
            mesh = UnstructuredMesh(2; n=Node(props))
            # Valid ODE without broadcasting
            @test_nowarn @addODE(model=mesh, scope=biochemistry,
                function ode1!(du, u, p, t)
                    for i in 1:length(u.n.a)
                        du.n.a[i] = -u.n.a[i]
                    end
                end
            )
            # Valid ODE with broadcasting
            @test_nowarn @addODE(model=mesh, scope=biochemistry2,
                function ode2!(du, u, p, t)
                    du.n.b .= -u.n.a
                end
            )
            # Test that functions are properly added to mesh
            @test haskey(mesh._functions, :biochemistry)
            @test mesh._functions[:biochemistry][1] === :ODE
            @test mesh._functions[:biochemistry][2] isa Tuple{<:Function}
        end

        @testset "@addODE - parameter validation" begin
            props = (a = Float64, b=Float64)
            mesh = UnstructuredMesh(2; n=Node(props))
            # Missing model parameter
            @test_throws LoadError @eval @addODE(scope=mechanics,
                function rule_bad1!(du, u, p, t)
                    du.n.a[1] = 1.0
                end
            )
            # Missing scope parameter
            @test_throws LoadError @eval @addODE(model=mesh,
                function rule_bad2!(du, u, p, t)
                    du.n.a[1] = 1.0
                end
            )
            # Wrong number of function arguments
            @test_throws LoadError @eval @addODE(model=mesh, scope=mechanics,
                function rule_bad4!(du, u, p)
                    du.n.a[1] = 1.0
                end
            )
        end

    end

    # @testset "@addSDE macro" begin
    @testset "@addSDE macro" begin

        @testset "@addSDE - basic functionality" begin
            props = (a = Float64, b=Float64)
            mesh = UnstructuredMesh(2; n=Node(props))
            # Valid SDE without broadcasting
            @test_nowarn @addSDE(model=mesh, scope=biochemistry,
                function sde1!(du, u, p, t)
                    for i in 1:length(u.n.a)
                        du.n.a[i] = -u.n.a[i]
                    end
                end,
                function sde_noise1!(du, u, p, t)
                    for i in 1:length(u.n.a)
                        du.n.a[i] = 0.1 * randn()
                    end
                end
            )
            # Valid ODE with broadcasting
            @test_nowarn @addSDE(model=mesh, scope=biochemistry2,
                function ode2!(du, u, p, t)
                    du.n.b .= -u.n.a
                end,
                function ode_noise2!(du, u, p, t)
                    du.n.b .= 0.1 .* randn.(length(u.n.a))
                end
            )
            # Test that functions are properly added to mesh
            @test haskey(mesh._functions, :biochemistry)
            @test mesh._functions[:biochemistry][1] === :SDE
            @test mesh._functions[:biochemistry][2] isa Tuple{<:Function, <:Function}
        end

        @testset "@addSDE - parameter validation" begin
            props = (a = Float64, b=Float64)
            mesh = UnstructuredMesh(2; n=Node(props))
            # Missing model parameter
            @test_throws LoadError @eval @addSDE(scope=mechanics,
                function rule_bad1!(du, u, p, t)
                    du.a.a[1] = 1.0
                end,
                function rule_noise_bad1!(du, u, p, t)
                    du.a.a[1] = 0.1 * randn()
                end
            )
            # Missing scope parameter
            @test_throws LoadError @eval @addSDE(model=mesh,
                function rule_bad2!(du, u, p, t)
                    du.a.a[1] = 1.0
                end,
                function rule_noise_bad2!(du, u, p, t)
                    du.a.a[1] = 0.1 * randn()
                end
            )
            # Wrong number of function arguments
            @test_throws LoadError @eval @addSDE(model=mesh, scope=mechanics,
                function rule_bad4!(du, u, p)
                    du.a.a[1] = 1.0
                end
            )
        end
    end

end