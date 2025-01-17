using Fermions, Plots, ProgressMeter


function SpecFunc2(
    eigVals::Vector{Float64},
    eigVecs::Vector{Dict{BitVector,Float64}},
    probes::Dict{String,Vector{Tuple{String,Vector{Int64},Float64}}},
    freqValues::Vector{Float64},
    basisStates::Vector{Dict{BitVector,Float64}},
    standDev::Union{Vector{Float64}, Float64};
    degenTol::Float64=0.,
    normalise::Bool=true,
    silent::Bool=false,
    broadFuncType::String="lorentz",
)
    eigenStates = Vector{Float64}[] 
    @time for vector in eigVecs
        push!(eigenStates, ExpandIntoBasis(vector, basisStates))
    end
    probeMatrices = Dict{String,Matrix{Float64}}(name => zeros(length(basisStates), length(basisStates)) for name in keys(probes))
    @time for (name,probe) in collect(probes)
        probeMatrices[name] = OperatorMatrix(basisStates, probe)
    end

    return SpecFunc2(eigVals, eigenStates, probeMatrices, freqValues, standDev;
                    normalise=normalise, degenTol=degenTol, silent=silent, broadFuncType=broadFuncType)
end

function SpecFunc2(
    eigVals::Vector{Float64},
    eigVecs::Vector{Vector{Float64}},
    probes::Dict{String,Matrix{Float64}},
    freqValues::Vector{Float64},
    standDev::Union{Vector{Float64}, Float64};
    degenTol::Float64=0.,
    normalise::Bool=true,
    silent::Bool=false,
    broadFuncType::String="lorentz",
    )

    @assert length(eigVals) == length(eigVecs)
    @assert broadFuncType ∈ ("lorentz", "gauss")
    @assert issorted(freqValues)

    broadeningFunc(x, standDev) = ifelse(broadFuncType=="lorentz", standDev ./ (x .^ 2 .+ standDev .^ 2), exp.(-0.5 .* ((x ./ standDev).^2)) ./ (standDev .* ((2π)^0.5)))

    energyGs = minimum(eigVals)
    specFunc = 0 .* freqValues

    degenerateManifold = eigVals .≤ energyGs + degenTol
    if !silent
        println("Degeneracy = ", length(eigVals[degenerateManifold]), "; Range=[$(eigVals[degenerateManifold][1]), $(eigVals[degenerateManifold][end])]")
    end

    @showprogress desc="$(broadFuncType)" for groundState in eigVecs[degenerateManifold]
        excitationCreate = probes["create"] * groundState
        excitationDestroy = probes["destroy"] * groundState
        excitationCreateBra = groundState' * probes["create"]
        excitationDestroyBra = groundState' * probes["destroy"]
        for index in eachindex(eigVals)
            excitedState = eigVecs[index]
            spectralWeights = [(excitationDestroyBra * excitedState) * (excitedState' * excitationCreate),
                               (excitedState' * excitationDestroy) * (excitationCreateBra * excitedState)
                              ]
            specFunc .+= spectralWeights[1] * broadeningFunc(freqValues .+ energyGs .- eigVals[index], standDev)
            specFunc .+= spectralWeights[2] * broadeningFunc(freqValues .- energyGs .+ eigVals[index], standDev)
        end
    end
    areaSpecFunc = sum(specFunc .* (maximum(freqValues) - minimum(freqValues[1])) / (length(freqValues) - 1))
    if areaSpecFunc > 1e-10 && normalise
        specFunc = specFunc ./ areaSpecFunc
    end
    return specFunc
end

V = 0.0
hubbardU = 2.
epsilond = -hubbardU/2
numBathSites = 5
dispersion = 0 .* 10 .^ range(-1, stop=-4, length=numBathSites)
SIAMHamiltonian = SiamKSpace(dispersion, V, epsilond, hubbardU; globalField=1e-8)
basis = BasisStates(2 * (1 + numBathSites))
E, X = Spectrum(SIAMHamiltonian, basis, ['N', 'Z'])
println(E[1])
freqValues = collect(-3:0.01:3)
standDev = 0.001
probes = Dict("create" => [("+", [1], 1.)], "destroy" => [("-", [3], 1.)])
specfunc1 = SpecFunc2(E, X, probes, freqValues, basis, standDev)
p = Plots.plot(freqValues[freqValues .|> abs .≤ 0.3], specfunc1[freqValues .|> abs .≤ 0.3])
savefig(p, "plot1-$(V)-$(numBathSites).pdf")
p = Plots.plot(freqValues, specfunc1)
savefig(p, "plot12-$(V)-$(numBathSites).pdf")
#=probes = Dict("create" => [("+", [1], 1.)], "destroy" => [("-", [1], 1.)])=#
#=specfunc2 = SpecFunc(E, X, probes, freqValues, basis, standDev)=#
#=p = plot(freqValues[freqValues .|> abs .≤ 0.5], specfunc2[freqValues .|> abs .≤ 0.5])=#
#=savefig(p, "plot2-$(V)-$(numBathSites).pdf")=#
#=p = plot(freqValues, specfunc2)=#
#=savefig(p, "plot22-$(V)-$(numBathSites).pdf")=#
