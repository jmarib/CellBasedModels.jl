function generateCode!(f, nameFunction::Symbol, rule::Union{Rule}, specialization)

    for neighbors in NEIGHBORS_ALGS

        code = quote $(rule.rules...) end
        code_ = postwalk(x-> @capture(x, a_) && typeof(a) == Symbol ? (startswith(string(a), "@loopOverNeighbors") ? Symbol("$a$neighbors") : x) : x, code)
        
        code__ = quote
            function $nameFunction(community::CommunityABM{ABM{$(specialization...)}, CPU, $neighbors{$(specialization...)}}) 
                function kernel!($(rule.dt), $(rule.c)) 
                    $code_
                end
                kernel!(community._parametersNew, community._parameters)
            end
                    
        end
        
        CellBasedModels.CustomFunctions.eval(code__)
        
        if hasCuda()
        
            codef_ = quote end
            for (i,code) in enumerate(integrator.f)
                kernelname = Meta.parse("kernel$(i)!")
                codef_ = quote
                    function $(kernelname)($(rule.dt), $(rule.c)) 
                        $code
                    end
                    @cuda threads=256 blocks=ceil(Int,size(community.agents,1)/256) $(kernelname)(community._parametersNew, community._parameters)
                end

            end

                code__ = quote
                function $nameFunction(community::CommunityABM{ABM{$(specialization...)}, GPU, $neighbors{$(specialization...)}})
                    $codef_
                end                        
            end
            CellBasedModels.CustomFunctions.eval(code__)
        end
    end

    f[nameFunction] = eval(Meta.parse("CellBasedModels.CustomFunctions.$nameFunction"))

    return NamedTuple(f)

end

function generateCode!(f, nameFunction::Symbol, rule::Union{ODE}, specialization)

    for neighbors in NEIGHBORS_ALGS

        code = quote $(rule.f...) end
        code_ = postwalk(x-> @capture(x, a_) && typeof(a) == Symbol ? (startswith(string(a), "@loopOverNeighbors") ? Symbol("$a$neighbors") : x) : x, code)
        
        code__ = quote
            function $nameFunction($(rule.dt), $(rule.c), dt)
                $code_
            end
            function $nameFunction(community::CommunityABM{ABM{$(specialization...)}, CPU, $neighbors{$(specialization...)}}) 
                kernel!(community._parametersNew, community._parameters)
            end
        end
    
        CellBasedModels.CustomFunctions.eval(code__)
        
        if hasCuda()
        
            codef_ = quote end
            for (i,code) in enumerate(integrator.f)
                kernelname = Meta.parse("kernel$(i)!")
                codef_ = quote
                    function $(kernelname)($(rule.dt), $(rule.c)) 
                        $code
                    end
                    @cuda threads=256 blocks=ceil(Int,size(community.agents,1)/256) $(kernelname)(community._parametersNew, community._parameters)
                end

            end

                code__ = quote
                function $nameFunction(community::CommunityABM{ABM{$(specialization...)}, GPU, $neighbors{$(specialization...)}})
                    $codef_
                end                        
            end
            CellBasedModels.CustomFunctions.eval(code__)
        end
    end

    f[nameFunction] = eval(Meta.parse("CellBasedModels.CustomFunctions.$nameFunction"))

    return NamedTuple(f)

end

