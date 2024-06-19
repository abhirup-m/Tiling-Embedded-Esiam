using JLD2
using ProgressMeter
include("./constants.jl")
include("./rgFlow.jl")
include("./probes.jl")

# file name format for saving plots and animations
animName(orbitals, size_BZ, omega_by_t, scale, W_by_J_min, W_by_J_max, J_val) = "$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(W_by_J_min, digits=4))_$(round(W_by_J_max, digits=4))_$(round(J_val, digits=4))_$(FIG_SIZE[1] * scale)x$(FIG_SIZE[2] * scale)"


function getCorrelations(size_BZ::Int64, dispersion::Vector{Float64}, kondoJArray::Array{Float64, 3}, W_val::Float64, orbitals::Tuple{String,String}, corrDefArray, cutoffFraction::Float64; twoParticle=0, tile=false)
    correlationResults = [[] for _ in corrDefArray]
    basis, suitableIndices, uniqueSequences, gstatesSet = getBlockSpectrum(size_BZ, dispersion, kondoJArray, W_val , orbitals, cutoffFraction)
    for (i, correlationDefinition) in enumerate(corrDefArray)
        if tile
            results = correlationMap(size_BZ, basis, dispersion, suitableIndices, uniqueSequences, gstatesSet, correlationDefinition; twoParticle=twoParticle)
        else
            results = correlationMap(size_BZ, basis, dispersion, suitableIndices, uniqueSequences, gstatesSet, correlationDefinition; twoParticle=twoParticle)
        end
        push!(correlationResults[i], results...)
    end
    return correlationResults
end

