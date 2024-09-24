##### Functions for calculating various probes          #####
##### (correlation functions, Greens functions, etc)    #####


"""
Function to calculate the total Kondo scattering probability Γ(k) = ∑_q J(k,q)^2
at the RG fixed point.
"""
function scattProb(kondoJArray::Array{Float64,3}, size_BZ::Int64, dispersion::Vector{Float64}, BZfraction::Float64)

    # allocate zero arrays to store Γ at fixed point and for the bare Hamiltonian.
    results_scaled = zeros(size_BZ^2)

    # loop over all points k for which we want to calculate Γ(k).
    Threads.@threads for point in 1:size_BZ^2

        # check if the point is one of the four corners or the 
        # center. If it is, then don't bother. These points are
        # not affected by the RG and therefore not of interest.
        if abs(dispersion[point] / maximum(dispersion)) < BZfraction
            targetStatesForPoint = collect(1:size_BZ^2)[abs.(dispersion).<=abs(dispersion[point])]

            # calculate the sum over q
            results_scaled[point] = sum(kondoJArray[point, targetStatesForPoint, end] .^ 2) ^ 0.5 / sum(kondoJArray[point, targetStatesForPoint, 1] .^ 2) ^ 0.5
        end

    end

    # get a boolean representation of results for visualisation, using the mapping
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


@everywhere function iterDiagResults(
        hamiltDetails::Dict,
        correlationFuncDict::Dict,
        maxSize::Int64,
        pivotLoc::Int64,
        pivotPointsArr::Vector{Vector{Int64}},
    )
    corrResults = Dict(k => zeros(repeat([hamiltDetails["size_BZ"]^2], numLegs)...) for (k, (numLegs, _)) in correlationFuncDict)
    pivotPoints = pivotPointsArr[pivotLoc]
    newStatesArr = [pivotPointsArr[pivotLoc:pivotLoc]; pivotPointsArr[1:pivotLoc-1]; pivotPointsArr[pivotLoc+1:end]]
    activeStatesArr = Vector{Int64}[pivotPoints]
    for newStates in newStatesArr[2:end]
        push!(activeStatesArr, [activeStatesArr[end]; newStates])
    end

    mapCorrNameToIndex = Dict()
    correlationDefDict = Dict{String, Vector{Tuple{String, Vector{Int64}, Float64}}}()
    for (name, (numLegs, func)) in correlationFuncDict
        for k_inds in combinations(eachindex(activeStatesArr[end]), numLegs)
            if !isempty(intersect(activeStatesArr[end][k_inds...], pivotPoints))
                correlationDefDict[name * join(k_inds)] = func(k_inds...)
                mapCorrNameToIndex[name * join(k_inds)] = (name, activeStatesArr[end][k_inds])
            end
        end
    end

    hamiltonianFamily = fetch.([
                                Threads.@spawn kondoKSpace(activeStates,
                                                           hamiltDetails["dispersion"], 
                                                           hamiltDetails["kondoJArray"],
                                                           states -> bathIntForm(hamiltDetails["W_val"], hamiltDetails["orbitals"][2], hamiltDetails["size_BZ"], states);
                                                           impField=1e-4,
                                                           specialIndices=newStates,
                                                          )
                                for (activeStates, newStates) in zip(activeStatesArr, newStatesArr)
                               ]
                              )

    savePaths, iterDiagResults = IterDiag(
                                 hamiltonianFamily, 
                                 maxSize;
                                 symmetries=Char['N'],
                                 occReq=(x, N) -> div(N, 2) - 3 ≤ x ≤ div(N, 2) + 3,
                                 correlationDefDict=correlationDefDict,
                                 silent=true,
                                )

    for (k, v) in iterDiagResults
        if k ∉ keys(mapCorrNameToIndex)
            continue
        end
        name, k_inds = mapCorrNameToIndex[k]
        corrResults[name][k_inds...] += v / correlationFuncDict[name][1]
    end
    return corrResults
end


function correlationMap(
        hamiltDetails::Dict,
        numShells::Int64,
        correlationFuncDict,
        maxSize::Int64,
    )

    size_BZ = hamiltDetails["size_BZ"]

    # initialise zero array for storing correlations
    corrResults = Dict(k => zeros(repeat([size_BZ^2], numLegs)...) for (k, (numLegs, _)) in correlationFuncDict)
    corrResultsBool = Dict(k => zeros(repeat([size_BZ^2], numLegs)...) for (k, (numLegs, _)) in correlationFuncDict)

    pivotPointsArr = Vector{Int64}[]

    energyVals = dispersion[1:div(size_BZ+1,2)] .|> abs |> unique |> sort
    energyPartitions = [[0]; (energyVals[2:end] .+ energyVals[1:end-1]) ./ 2; energyVals[end]]
    oppositePoints = Dict{Int64, Vector{Int64}}()

    # pick out k-states from the southwest quadrant that have positive energies 
    # (hole states can be reconstructed from them (p-h symmetry))
    SWIndices = [p for p in 1:size_BZ^2 if map1DTo2D(p, size_BZ)[1] <= 0 && map1DTo2D(p, size_BZ)[2] <= 0 && energyPartitions[numShells+1] ≥ dispersion[p]]

    energyShells = Vector{Int64}[]
    for shell in 1:numShells
        push!(energyShells, SWIndices[(abs.(dispersion[SWIndices]) .≥ energyPartitions[shell]) .& (abs.(dispersion[SWIndices]) .≤ energyPartitions[shell+1])])
        particleStates = energyShells[end][dispersion[energyShells[end]] .≥ 0]
        distancesFromNodeParticle = [sum((map1DTo2D(p, size_BZ) .- (-π/2, -π/2)) .^ 2)^0.5 for p in particleStates]
        holeStates = energyShells[end][dispersion[energyShells[end]] .≤ 0]
        distancesFromNodeHole = [sum((map1DTo2D(p, size_BZ) .- (-π/2, -π/2)) .^ 2)^0.5 for p in holeStates]
        
        for distance in sort(unique(distancesFromNodeParticle))
            push!(pivotPointsArr, particleStates[distancesFromNodeParticle .== distance])
            for point in pivotPointsArr[end]
                oppositePoints[point] = holeStates[isapprox.(distancesFromNodeHole, distance, atol=1e-10)]
            end
        end
    end

    corrResults = @showprogress @distributed (d1, d2) -> mergewith(+, d1, d2) for pivotLoc in eachindex(pivotPointsArr)
        iterDiagResults(hamiltDetails, correlationFuncDict, maxSize, pivotLoc, copy(pivotPointsArr))
    end

    corrResults = propagateIndices(vcat(pivotPointsArr...), corrResults, size_BZ, oppositePoints)
    # for pivotPoint in vcat(pivotPointsArr...)
    #     newPoints = propagateIndices(pivotPoint, size_BZ, oppositePoints[pivotPoint])
    #     println(pivotPoint, newPoints, oppositePoints[pivotPoint])
    #     for (k, correlation) in corrResults
    #         corrResults[k][newPoints] .= correlation[pivotPoint]
    #     end
    # end

    for (name, results) in corrResults
        @assert !any(isnan.(results))
        corrResultsBool[name] = [abs(r) ≤ 1e-6 ? -1 : 1 for r in results]
    end
    return corrResults, corrResultsBool
end


function entanglementMap(size_BZ::Int64, basis::Dict{Tuple{Int64, Int64}, Vector{BitVector}}, dispersion::Vector{Float64}, suitableIndices::Vector{Int64}, uniqueSequences::Vector{Vector{NTuple{TRUNC_DIM, Int64}}}, gstatesSet::Vector{Vector{Dict{BitVector, Float64}}}, mutInfoIndices::Vector{Int64})

    # initialise zero array for storing correlations
    # results = ifelse(twoParticle == 0, zeros(size_BZ^2), zeros(size_BZ^2, size_BZ^2))
    resultsVNE = zeros(size_BZ^2)
    resultsMI = Dict(zip(mutInfoIndices, [zeros(size_BZ^2) for _ in mutInfoIndices]))

    # initialise zero array to count the number of times a particular k-state
    # appears in the computation. Needed to finally average over all combinations.
    contributorCounterVNE = fill(0, size_BZ^2)

    correlationVNE = [fetch.([Threads.@spawn fermions.vnEntropy(gstates, [2 * i + 1, 2 * i + 2]) for i in 1:TRUNC_DIM]) for gstates in gstatesSet]
    # calculate the correlation for all such configurations.
    for (sequenceSet, vne) in zip(uniqueSequences, correlationVNE)
        for sequence in sequenceSet
            maxE = maximum(abs.(dispersion[collect(sequence)]))
            for (i, index) in enumerate(sequence)
                if abs(abs(dispersion[index]) - maxE) < TOLERANCE
                    resultsVNE[index] += vne[i]
                    contributorCounterVNE[index] += 1
                end
            end
        end
    end

    correlationMI = [fetch.([Threads.@spawn fermions.mutInfo(gstates, ([2 * i + 1, 2 * i + 2], [2 * j + 1, 2 * j + 2])) for (i,j) in Iterators.product(1:TRUNC_DIM, 1:TRUNC_DIM)]) for gstates in gstatesSet]
    contributorCounterMI = Dict(zip(mutInfoIndices, [zeros(size_BZ^2) for _ in mutInfoIndices]))
    for (sequenceSet, mutInfo) in zip(uniqueSequences, correlationMI)
        for sequence in sequenceSet
            if isnothing(intersect(sequence, mutInfoIndices))
                continue
            end
            maxE = maximum(abs.(dispersion[collect(sequence)]))
            for (i, (index1, index2)) in enumerate(Iterators.product(sequence, sequence))
                if (index1 ∈ mutInfoIndices || index2 ∈ mutInfoIndices ) && (abs(abs(dispersion[index1]) - maxE) < TOLERANCE || abs(abs(dispersion[index2]) - maxE) < TOLERANCE)
                    if index1 ∈ mutInfoIndices
                        resultsMI[index1][index2] += mutInfo[i]
                        contributorCounterMI[index1][index2] += 1
                    end
                    if index2 ∈ mutInfoIndices && index1 ≠ index2
                        resultsMI[index2][index1] += mutInfo[i]
                        contributorCounterMI[index2][index1] += 1
                    end
    end end end end

    @assert all(x -> x > 0, contributorCounterVNE[suitableIndices])
    resultsVNE[suitableIndices] ./= contributorCounterVNE[suitableIndices]
    Threads.@threads for index in suitableIndices
        newPoints = propagateIndices(index, size_BZ)
        resultsVNE[newPoints] .= resultsVNE[index]
    end
    @assert !any(isnan.(resultsVNE))

    for (index, counter) in contributorCounterMI
        @assert all(x -> x > 0, counter[suitableIndices])
        resultsMI[index][suitableIndices] ./= counter[suitableIndices]
    end

    Threads.@threads for index in suitableIndices
        newPoints = propagateIndices(index, size_BZ)
        resultsVNE[newPoints] .= resultsVNE[index]
        for mutInfoIndex in keys(resultsMI)
            resultsMI[mutInfoIndex][newPoints] .= resultsMI[mutInfoIndex][index]
        end
    end
    for index in keys(resultsMI)
        @assert !any(isnan.(resultsMI[index]))
    end

    resultsVNE[resultsVNE .< 1e-3] .= 0
    for k in keys(resultsMI)
        resultsMI[k][resultsMI[k] .< 1e-3] .= 0
    end
    # resultsVNE_bool = [r <= 0 ? -1 : 1 for r in resultsVNE]
    return resultsVNE, resultsMI
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
