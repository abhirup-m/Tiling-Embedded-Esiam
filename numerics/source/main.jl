using JLD2
using ProgressMeter
include("./constants.jl")
include("./rgFlow.jl")
include("./probes.jl")

# file name format for saving plots and animations
animName(orbitals, size_BZ, omega_by_t, scale, W_by_J_min, W_by_J_max, J_val) = "$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(W_by_J_min, digits=4))_$(round(W_by_J_max, digits=4))_$(round(J_val, digits=4))_$(FIG_SIZE[1] * scale)x$(FIG_SIZE[2] * scale)"


function rgFlowData(size_BZ::Int64, omega_by_t::Float64, J_val::Float64, W_by_J_range::Vector{Float64}, orbitals::Tuple{String,String}; saveDir::String="./data/")
    # determines whether the inner loops (RG iteration, for a single value of W)
    # should show progress bars.
    progressbarEnabled = length(W_by_J_range) == 1 ? true : false

    # file paths for saving RG flow results, specifically bare and fixed point values of J as well as the dispersion
    savePaths = [saveDir * "$(orbitals)_$(size_BZ)_$(round(J_val, digits=5))_$(round(W_by_J, digits=5)).jld2" for W_by_J in W_by_J_range]

    # create the data save directory if it doesn't already exist
    isdir(saveDir) || mkdir(saveDir)

    # loop over all given values of W/J, get the fixed point distribution J(k1,k2) and save them in files
    @showprogress for (j, W_by_J) in collect(enumerate(W_by_J_range))
        W_val = J_val * W_by_J
        kondoJArray, dispersion = momentumSpaceRG(size_BZ, omega_by_t, J_val, W_val, orbitals; progressbarEnabled=progressbarEnabled)
        averageKondoScale = sum(abs.(kondoJArray[:, :, 1])) / length(kondoJArray[:, :, 1])
        @assert averageKondoScale > RG_RELEVANCE_TOL
        kondoJArray[:, :, end] .= ifelse.(abs.(kondoJArray[:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, kondoJArray[:, :, end], 0)
        jldopen(savePaths[j], "w") do file
            file["kondoJArray"] = kondoJArray
            file["dispersion"] = dispersion
            file["W_val"] = W_val
            file["size_BZ"] = size_BZ
            file["orbitals"] = orbitals
        end
    end
    return savePaths
end


function getCorrelations(size_BZ::Int64, dispersion::Vector{Float64}, kondoJArray::Array{Float64, 3}, W_val::Float64, orbitals::Tuple{String,String}, corrDefArray, cutoffFraction::Float64; twoParticle=0, tile=false)
    correlationResults = [[] for _ in corrDefArray]
    basis, suitableIndices, uniqueSequences, eigenSet = getBlockSpectrum(size_BZ, dispersion, kondoJArray, W_val , orbitals, cutoffFraction)
    for (i, correlationDefinition) in enumerate(corrDefArray)
        if tile
            results = correlationMap(size_BZ, basis, dispersion, suitableIndices, uniqueSequences, eigenSet, correlationDefinition; twoParticle=twoParticle)
        else
            results = correlationMap(size_BZ, basis, dispersion, suitableIndices, uniqueSequences, eigenSet, correlationDefinition; twoParticle=twoParticle)
        end
        push!(correlationResults[i], results...)
    end
    return correlationResults
end

