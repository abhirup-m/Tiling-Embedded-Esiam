##### Functions for calculating various probes          #####
##### (correlation functions, Greens functions, etc)    #####

@everywhere using ProgressMeter, Combinatorics
@everywhere using fermions
@everywhere include("models.jl")

"""
Function to calculate the total Kondo scattering probability Γ(k) = ∑_q J(k,q)^2
at the RG fixed point.
"""
function scattProb(
        size_BZ::Int64,
        kondoJArray::Array{Float64,3},
        dispersion::Vector{Float64},
    )

    # allocate zero arrays to store Γ at fixed point and for the bare Hamiltonian.
    results_scaled = zeros(size_BZ^2)

    # loop over all points k for which we want to calculate Γ(k).
    Threads.@threads for point in 1:size_BZ^2

        targetStatesForPoint = collect(1:size_BZ^2)[abs.(dispersion).<=abs(dispersion[point])]

        # calculate the sum over q
        results_scaled[point] = sum(kondoJArray[point, targetStatesForPoint, end] .^ 2) ^ 0.5 / sum(kondoJArray[point, targetStatesForPoint, 1] .^ 2) ^ 0.5

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
        vneFuncDict::Dict,
        maxSize::Int64,
        pivotLoc::Int64,
        pivotPointsArr::Vector{Vector{Int64}},
    )
    corrResults = Dict{String, Vector{Float64}}()
    for (k, (numLegs, _)) in correlationFuncDict 
        corrResults[k] = zeros(repeat([hamiltDetails["size_BZ"]^2], numLegs)...) 
    end
    for (k, (numLegs, _)) in vneFuncDict 
    corrResults[k] = zeros(repeat([hamiltDetails["size_BZ"]^2], numLegs)...) 
    end

    pivotPoints = pivotPointsArr[pivotLoc]
    newStatesArr = Vector{Int64}[pivotPointsArr[pivotLoc:pivotLoc]; pivotPointsArr[1:pivotLoc-1]; pivotPointsArr[pivotLoc+1:end]]
    newStatesArr = [newStates for newStates in newStatesArr if all(p -> hamiltDetails["dispersion"][p] ≤ maximum(hamiltDetails["dispersion"][pivotPoints]), newStates)]
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

    vneDefDict = Dict{String, Vector{Int64}}()
    for (name, (numLegs, func)) in vneFuncDict
        for k_inds in combinations(eachindex(activeStatesArr[end]), numLegs)
            if !isempty(intersect(activeStatesArr[end][k_inds...], pivotPoints))
                vneDefDict[name * join(k_inds)] = func(k_inds...)
                mapCorrNameToIndex[name * join(k_inds)] = (name, activeStatesArr[end][k_inds])
            end
        end
    end


    hamiltonianFamily = fetch.([
                                Threads.@spawn kondoKSpace(activeStates,
                                                           hamiltDetails["dispersion"], 
                                                           hamiltDetails["kondoJArray"],
                                                           states -> hamiltDetails["bathIntForm"](hamiltDetails["W_val"], hamiltDetails["orbitals"][2], hamiltDetails["size_BZ"], states);
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
                                 vneDefDict=vneDefDict,
                                 silent=true,
                                )

    for (k, v) in iterDiagResults
        if k ∉ keys(mapCorrNameToIndex)
            continue
        end
        name, k_inds = mapCorrNameToIndex[k]
        if name ∈ keys(correlationFuncDict)
            corrResults[name][k_inds...] += v[end] / correlationFuncDict[name][1]
        else
            corrResults[name][k_inds...] += v[end] / vneFuncDict[name][1]
        end
    end
    return corrResults
end


function correlationMap(
        hamiltDetails::Dict,
        numShells::Int64,
        correlationFuncDict::Dict,
        maxSize::Int64;
        vneFuncDict::Dict=Dict(),
    )

    size_BZ = hamiltDetails["size_BZ"]

    # initialise zero array for storing correlations
    
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

    desc = "W=$(round(hamiltDetails["W_val"], digits=3))"
    corrResults = @showprogress desc=desc @distributed (d1, d2) -> mergewith(+, d1, d2) for pivotLoc in eachindex(pivotPointsArr)
        iterDiagResults(hamiltDetails, correlationFuncDict, vneFuncDict, maxSize, pivotLoc, copy(pivotPointsArr))
    end

    corrResults = propagateIndices(vcat(pivotPointsArr...), corrResults, size_BZ, oppositePoints)

    corrResultsBool = Dict()
    for (name, results) in corrResults
        @assert !any(isnan.(results))
        corrResultsBool[name] = [abs(r) ≤ 1e-6 ? -1 : 1 for r in results]
    end
    return corrResults, corrResultsBool
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


function PhaseDiagram(J_vals_arr::Vector{Float64}, W_val_arr::Vector{Float64}, numPoints::Int64, phaseMaps::Dict{String, Int64})
    function PhaseStrip(J_val::Float64, W_val_arr::Vector{Float64}, phaseMaps::Dict{String, Int64})
        phaseFlags = 0 .* collect(W_val_arr)
        trackPoints = Dict(
                           "N" => map2DTo1D(π/2, π/2, size_BZ),
                           "AN" => map2DTo1D(π/1, 0.0, size_BZ),
                           "M" => map2DTo1D(0.75 * π, 0.25 * π, size_BZ),
                          )

        gapTrackers = Dict("N" => NaN, "AN" => NaN, "M" => NaN)
        for (i, W_val) in enumerate(W_val_arr)
            kondoJArray, dispersion = momentumSpaceRG(size_BZ, omega_by_t, J_val, W_val, orbitals)
            averageKondoScale = sum(abs.(kondoJArray[:, :, 1])) / length(kondoJArray[:, :, 1])
            @assert averageKondoScale > RG_RELEVANCE_TOL
            kondoJArray[:, :, end] .= ifelse.(abs.(kondoJArray[:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, kondoJArray[:, :, end], 0)
            scattProbBool = scattProb(kondoJArray, size_BZ, dispersion, fractionBZ)[2]
            if all(>(0), scattProbBool[fermiPoints])
                phaseFlags[i] = phaseMaps["FL"]
            elseif !all(==(0), scattProbBool[fermiPoints])
                phaseFlags[i] = phaseMaps["PG"]
            else
                phaseFlags[i] = phaseMaps["MI"]
            end
            for (point, kpoint) in trackPoints
                if isnan(gapTrackers[point]) && scattProbBool[kpoint] == 0
                    gapTrackers[point] = W_val
                end
            end
        end
        return phaseFlags, gapTrackers
    end

    densityOfStates, dispersionArray = getDensityOfStates(tightBindDisp, size_BZ)
    fermiPoints = unique(getIsoEngCont(dispersionArray, 0.0))
    @assert length(fermiPoints) == 2 * size_BZ - 2
    @assert all(==(0), dispersionArray[fermiPoints])

    phaseDiagram = zeros(numPoints, numPoints)
    nodeGap = zeros(numPoints)
    antiNodeGap = zeros(numPoints)
    midPointGap = zeros(numPoints)
    @time results = fetch.(@showprogress [Threads.@spawn PhaseStrip(J_val, W_val_arr, phaseMaps) for J_val in J_val_arr])
    for (i, (phaseResult, gapTrackers)) in enumerate(results)
        phaseDiagram[i, :] = phaseResult
        nodeGap[i] = gapTrackers["N"]
        antiNodeGap[i] = gapTrackers["AN"]
        midPointGap[i] = gapTrackers["M"]
    end
    return phaseDiagram, nodeGap, antiNodeGap, midPointGap
end
