##### Functions for calculating various probes          #####
##### (correlation functions, Greens functions, etc)    #####

using LinearAlgebra
using Combinatorics
using fermions
using ProgressMeter

"""
Function to calculate the total Kondo scattering probability Γ(k) = ∑_q J(k,q)^2
at the RG fixed point.
"""
function scattProb(kondoJArray::Array{Float64,3}, size_BZ::Int64, dispersion::Vector{Float64})

    # allocate zero arrays to store Γ at fixed point and for the bare Hamiltonian.
    results = zeros(size_BZ^2)
    results_bare = zeros(size_BZ^2)

    # loop over all points k for which we want to calculate Γ(k).
    Threads.@threads for point in 1:size_BZ^2

        # check if the point is one of the four corners or the 
        # center. If it is, then don't bother. These points are
        # not affected by the RG and therefore not of interest.
        if point ∉ [1, size_BZ, size_BZ^2 - size_BZ + 1, size_BZ^2, trunc(Int, 0.5 * (size_BZ^2 + 1))]
            targetStatesForPoint = collect(1:size_BZ^2)[abs.(dispersion).<=abs(dispersion[point])]

            # calculate the sum over q
            results[point] = sum(kondoJArray[point, targetStatesForPoint, end] .^ 2)
            results_bare[point] = sum(kondoJArray[point, targetStatesForPoint, 1] .^ 2)
        end

    end

    # get a boolean representation of results for visualisation, using the mapping
    results_scaled = results ./ results_bare
    results_bool = ifelse.(abs.(results_scaled) .> 0, 1, 0)

    return results_scaled, results_bool
end


"""
Return the map of Kondo couplings M_k(q) = J^2_{k,q} given the state k.
Useful for visualising how a single state k interacts with all other states q.
"""
function kondoCoupMap(k_vals::Tuple{Float64,Float64}, size_BZ::Int64, kondoJArrayFull::Array{Float64,3})
    kspacePoint = map2DTo1D(k_vals..., size_BZ)
    results = kondoJArrayFull[kspacePoint, :, end] .^ 2
    results_bare = kondoJArrayFull[kspacePoint, :, 1] .^ 2
    results_bool = [r / r_b <= RG_RELEVANCE_TOL ? -1 : 1 for (r, r_b) in zip(results, results_bare)]
    results[results_bool.<0] .= NaN
    return results, results_bare, results_bool
end


function correlationMap(size_BZ, basis, dispersion, uniqueSequences, eigenSet, correlationDefinition)
    suitableIndices = getUpperQuadrantLowerIndices(size_BZ)

    # initialise zero array for storing correlations
    results = zeros(size_BZ^2)

    # initialise zero array to count the number of times a particular k-state
    # appears in the computation. Needed to finally average over all combinations.
    contributorCounter = zeros(size_BZ^2)

    correlationResults = fetch.([Threads.@spawn fermions.gstateCorrelation(basis, eigenvals, eigenstates, correlationDefinition.(1:TRUNC_DIM)) 
                                 for (eigenvals, eigenstates) in eigenSet])

    # calculate the correlation for all such configurations.
    for (sequenceSet, correlationResult) in zip(uniqueSequences, correlationResults)
        for sequence in sequenceSet
            maxE = maximum(abs.(dispersion[collect(sequence)]))
            for (i, index) in enumerate(sequence)
                if abs(abs(dispersion[index]) - maxE) < TOLERANCE
                    results[index] = ifelse(correlationResult[i] > results[index], correlationResult[i], results[index])
                    contributorCounter[index] += 1
                end
            end
        end
    end

    @assert all(contributorCounter[suitableIndices] .> 0)

    # average over all sequences
    # results[suitableIndices] ./= contributorCounter[suitableIndices]
    results = propagateIndices(suitableIndices, size_BZ, results)
    results_bool = [r <= 0 ? -1 : 1 for r in results]
    results[0 .< abs.(results) .< 1e-2] .= 1e-2
    return results, results_bool
end


function tiledCorrelationMap(size_BZ, energyContours, spectrumSet, sequenceSets, correlationDefinition, tiler)
    results = zeros(size_BZ^2, size_BZ^2)
    correlationmap, _ = correlationMap(size_BZ, energyContours, spectrumSet, sequenceSets, correlationDefinition)
    results = tiler(correlationmap)
    results[abs.(results) .< TOLERANCE ^ 0.5] .= TOLERANCE ^ 0.5
    results_bool = [r <= RG_RELEVANCE_TOL ? -1 : 1 for r in results]
    return results, results_bool
end


function transitionPoints(size_BZ_max::Int64, W_by_J_max::Float64, omega_by_t::Float64, J_val::Float64, orbitals::Tuple{String,String}; figScale::Float64=1.0, saveDir::String="./data/")
    size_BZ_min = 5
    size_BZ_vals = size_BZ_min:4:size_BZ_max
    antinodeTransition = Float64[]
    nodeTransition = Float64[]
    for (kvals, array) in zip([(-pi / 2, -pi / 2), (0.0, -pi)], [nodeTransition, antinodeTransition])
        W_by_J_bracket = [0, W_by_J_max]
        @showprogress for (i, size_BZ) in enumerate(size_BZ_vals)
            if i > 1
                W_by_J_bracket = [array[i-1], W_by_J_max]
            end
            kpoint = map2DTo1D(kvals..., size_BZ)
            while maximum(W_by_J_bracket) - minimum(W_by_J_bracket) > 0.1
                bools = []
                for W_by_J in [W_by_J_bracket[1], sum(W_by_J_bracket) / 2, W_by_J_bracket[2]]
                    @time kondoJArrayFull, dispersion = momentumSpaceRG(size_BZ, omega_by_t, J_val, J_val * W_by_J, orbitals)
                    @time results, results_bool = mapProbeNameToProbe("scattProb", size_BZ, kondoJArrayFull, W_by_J * J_val, dispersion, orbitals)
                    push!(bools, results_bool[kpoint] == 0)
                end
                if bools[1] == false && bools[3] == true
                    if bools[2] == true
                        W_by_J_bracket[2] = sum(W_by_J_bracket) / 2
                    else
                        W_by_J_bracket[1] = sum(W_by_J_bracket) / 2
                    end
                else
                    W_by_J_bracket[2] = W_by_J_bracket[1]
                    W_by_J_bracket[1] = 0
                end
            end
            push!(array, sum(W_by_J_bracket) / 2)
        end
    end
end
