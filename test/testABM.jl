@testset verbose=verbose "ABM - CommunityABM" begin

    @testset "struct - Global" begin

        environment = AgentGlobal(
                    properties=
                        (
                            temperature = Parameter(Float64; defaultValue=37.0, description="temp", _scope=:global, _updated=true),
                            pressure    = Parameter(Float64; defaultValue=1.0,  description="press", _scope=:global, _DE=true),
                            note        = Parameter(Bool;  defaultValue=false, description="note",  _scope=:agent),
                        )
                )

        # Build ABM
        model = ABM(
            3,
            
            agents = (
                environment = environment,
            ),

            rules = (
                ode_env = ODE(:c, :dt,
                    quote
                        dt.environment.pressure[1] = 0.1
                    end
                ),

                rule_env = Rule(:c, :cn,
                    quote
                        if c.environment.temperature[1] > 1.0
                            cn.environment.temperature[1] = 0
                        end
                    end
                ),
            )

        )

        @test model.environment.temperature._updated == true
        @test model.environment.temperature._DE == false

        @test model.environment.pressure._updated == false
        @test model.environment.pressure._DE == true

        @test model.environment.note._updated == false
        @test model.environment.note._DE == false

        # Build ABM
        model = ABM(
            3,
            
            agents = (
                environment = environment,
            ),

            rules = (
                ode_env2 = ODE(:c, :dt,
                    quote
                        env = dt.environment
                        for i in eachindex(env.pressure)
                            @views env.pressure[i] = 0.1
                        end
                    end
                ),

                rule_env2 = Rule(:c, :cn,
                    quote
                        d = cn
                        g = c.environment
                        if g.temperature[1] > 1.0
                            d.environment.temperature[1] = 0
                        end
                    end
                ),
            )

        )

        @test model.environment.temperature._updated == true
        @test model.environment.temperature._DE == false

        @test model.environment.pressure._updated == false
        @test model.environment.pressure._DE == true

        @test model.environment.note._updated == false
        @test model.environment.note._DE == false

        model = ABM(
            3,
            
            agents = (
                environment = environment,
            ),

            rules = (
                rule_env2 = Rule(:c, :cn,
                    quote
                        cn.environment.temperature[1] += 0.2
                    end
                ),
            )

        )

        @test typeof(model) === ABM{3, (:environment,), Tuple{AgentGlobal{(:temperature, :pressure, :note), Tuple{Float64, Float64, Bool}},}}
        @test haskey(model._agents, :environment)
        @test typeof(model._agents.environment) === AgentGlobal{(:temperature, :pressure, :note), Tuple{Float64, Float64, Bool}}

        community = CommunityABM(
            model;            
            environment = 
                CommunityGlobal(
                    model.environment;
                )
        )

        CellBasedModels.CustomFunctions.rule_env(community)
        model._functions.rule_env2(community)

        @test community.environment.temperature[1] == 0.0
        @test community._parametersNew.environment.temperature[1] == 0.2

        update!(community)

        @test community.environment.temperature[1] == 0.2
        @test community._parametersNew.environment.temperature[1] == 0.2

    end

end