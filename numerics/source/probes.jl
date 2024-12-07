##### Functions for calculating various probes          #####
##### (correlation functions, Greens functions, etc)    #####

@everywhere using ProgressMeter, Combinatorics
@everywhere using Fermions
# @everywhere include("models.jl")

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
        maxSize::Int64,
        pivotPoints::Vector{Int64},
        sortedPoints::Vector{Int64},
        correlationFuncDict::Dict,
        vneFuncDict::Dict,
        mutInfoFuncDict::Dict,
        bathIntLegs::Int64,
        noSelfCorr::Vector{String},
        addPerStep::Int64,
    )
    allKeys = vcat(keys(correlationFuncDict)..., keys(vneFuncDict)..., keys(mutInfoFuncDict)...)
    corrResults = Dict{String, Vector{Float64}}(k => zeros(hamiltDetails["size_BZ"]^2) for k in allKeys)

    nonPivotPoints = filter(∉(pivotPoints), sortedPoints)
    pointsSequence = vcat(pivotPoints, nonPivotPoints)

    mapCorrNameToIndex = Dict()
    correlationDefDict = Dict{String, Vector{Tuple{String, Vector{Int64}, Float64}}}()
    pivotIndices = findall(∈(pivotPoints), pointsSequence)
    for (name, (secondMomentum, func)) in correlationFuncDict
        secondIndex = isnothing(secondMomentum) ? nothing : findfirst(==(secondMomentum), pointsSequence)
        for pivotIndex in pivotIndices
            if name ∈ noSelfCorr && secondIndex == pivotIndex
                continue
            end
            correlationDefDict[name * string(pivotIndex)] = func(pivotIndex, secondIndex)
            mapCorrNameToIndex[name * string(pivotIndex)] = (name, pointsSequence[pivotIndex])
        end
    end

    vneDefDict = Dict{String, Vector{Int64}}()
    for (name, func) in vneFuncDict
        for pivotIndex in pivotIndices
            vneDefDict[name * string(pivotIndex)] = func(pivotIndex)
            mapCorrNameToIndex[name * string(pivotIndex)] = (name, pointsSequence[pivotIndex])
        end
    end

    mutInfoDefDict = Dict{String, NTuple{2, Vector{Int64}}}()
    for (name, (secondMomentum, func)) in mutInfoFuncDict
        secondIndex = isnothing(secondMomentum) ? nothing : findfirst(==(secondMomentum), pointsSequence)
        for pivotIndex in pivotIndices
            partyA, partyB = func(pivotIndex, secondIndex)
            if sort(partyA) ≠ sort(partyB)
                mutInfoDefDict[name * join(pivotIndex)] = (partyA, partyB)
                mapCorrNameToIndex[name * join(pivotIndex)] = (name, pointsSequence[pivotIndex])
            end
        end
    end

    bathIntFunc = points -> hamiltDetails["bathIntForm"](hamiltDetails["W_val"], 
                                                         hamiltDetails["orbitals"][2],
                                                         hamiltDetails["size_BZ"],
                                                         points)
                            
    hamiltonian = KondoModel(
                             hamiltDetails["dispersion"][pointsSequence],
                             hamiltDetails["kondoJArray"][pointsSequence, pointsSequence],
                             pointsSequence,
                             bathIntFunc;
                             bathIntLegs=bathIntLegs,
                             globalField=1e-8,
                             couplingTolerance=1e-10,
                            )
    indexPartitions = [2 + 2 * length(pivotPoints)]
    while indexPartitions[end] < 2 + 2 * length(pointsSequence)
        push!(indexPartitions, indexPartitions[end] + 2 * addPerStep)
    end
    hamiltonianFamily = MinceHamiltonian(hamiltonian, indexPartitions)
    iterDiagResults = nothing
    id = nothing
    while true
        output = IterDiag(
                          hamiltonianFamily, 
                          maxSize;
                          symmetries=Char['N', 'S'],
                          magzReq=(m, N) -> -1 ≤ m ≤ 2,
                          occReq=(x, N) -> div(N, 2) - 3 ≤ x ≤ div(N, 2) + 3,
                          correlationDefDict=correlationDefDict,
                          vneDefDict=vneDefDict,
                          mutInfoDefDict=mutInfoDefDict,
                          silent=true,
                          maxMaxSize=maxSize,
                         )
        exitCode = 0
        if length(output) == 2
            savePaths, iterDiagResults = output
        else
            savePaths, iterDiagResults, exitCode = output
        end
                          

        if exitCode > 0
            id = rand()
            println("Error code $(exitCode). Retry id=$(id).")
        else
            if !isnothing(id)
                println("Passed $(id).")
            end
            break
        end
    end

    for (k, v) in iterDiagResults
        if k ∉ keys(mapCorrNameToIndex)
            continue
        end
        name, k_ind = mapCorrNameToIndex[k]
        if name ∈ allKeys
            corrResults[name][k_ind] = ifelse(abs(v[end]) < 1e-10, 0, v[end])
        end
    end
    return corrResults

end


function iterDiagSpecFunc(
        hamiltDetails::Dict,
        maxSize::Int64,
        sortedPoints::Vector{Int64},
        specFuncDict::Dict,
        bathIntLegs::Int64,
        addPerStep::Int64,
        freqValues::Vector{Float64},
        standDev::Union{Vector{Float64}, Float64},
    )

    bathIntFunc = points -> hamiltDetails["bathIntForm"](hamiltDetails["W_val"], 
                                                         hamiltDetails["orbitals"][2],
                                                         hamiltDetails["size_BZ"],
                                                         points)
                            
    hamiltonian = KondoModel(
                             hamiltDetails["dispersion"][sortedPoints],
                             hamiltDetails["kondoJArray"][sortedPoints, sortedPoints],
                             sortedPoints,
                             bathIntFunc;
                             bathIntLegs=bathIntLegs,
                             globalField=hamiltDetails["globalField"],
                             couplingTolerance=1e-10,
                            )
    append!(hamiltonian, [("n", [1], -6), ("n", [2], -6), ("nn", [1, 2], 12.)])
    indexPartitions = [2]
    while indexPartitions[end] < 2 + 2 * length(sortedPoints)
        push!(indexPartitions, indexPartitions[end] + 2 * addPerStep)
    end
    hamiltonianFamily = MinceHamiltonian(hamiltonian, indexPartitions)

    savePaths, resultsDict, specFuncOperators = IterDiag(
                                                         hamiltonianFamily, 
                                                         maxSize;
                                                         symmetries=Char['N', 'S'],
                                                         #=magzReq=(m, N) -> -1 ≤ m ≤ 2,=#
                                                         occReq=(x, N) -> div(N, 2) - 3 ≤ x ≤ div(N, 2) + 3,
                                                         silent=false,
                                                         maxMaxSize=maxSize,
                                                         specFuncDefDict=specFuncDict,
                                                        ) 
    totalSpecFunc = IterSpecFunc(savePaths, specFuncOperators, freqValues, standDev;
                                 normEveryStep=true, degenTol=1e-10,
                           )
    return totalSpecFunc

end
function correlationMap(
        hamiltDetails::Dict,
        numShells::Int64,
        correlationFuncDict::Dict,
        maxSize::Int64;
        vneFuncDict::Dict=Dict(),
        mutInfoFuncDict::Dict=Dict(),
        bathIntLegs::Int64=2,
        noSelfCorr::Vector{String}=String[],
        addPerStep::Int64=1,
    )
    size_BZ = hamiltDetails["size_BZ"]

    cutoffEnergy = hamiltDetails["dispersion"][div(size_BZ - 1, 2) + 2 - numShells]

    # pick out k-states from the southwest quadrant that have positive energies 
    # (hole states can be reconstructed from them (p-h symmetry))
    SWIndices = [p for p in 1:size_BZ^2 if map1DTo2D(p, size_BZ)[1] < 0
                 && map1DTo2D(p, size_BZ)[2] ≤ 0 
                 && abs(cutoffEnergy) ≥ abs(dispersion[p])
                ]

    calculatePoints = filter(p -> map1DTo2D(p, size_BZ)[1] ≤ map1DTo2D(p, size_BZ)[2], SWIndices)

    distancesFromNode = [sum((map1DTo2D(p, size_BZ) .- (-π/2, -π/2)) .^ 2)^0.5 for p in SWIndices]
    symmetricPairsNode = SWIndices[sortperm(distancesFromNode)]
    distancesFromAntiNode = [minimum([sum((map1DTo2D(p, size_BZ) .- (-π, 0.)) .^ 2)^0.5,
                                      sum((map1DTo2D(p, size_BZ) .- (0., -π)) .^ 2)^0.5])
                                     for p in SWIndices]
    symmetricPairsAntiNode = SWIndices[sortperm(distancesFromAntiNode)]
    oppositePoints = Dict{Int64, Vector{Int64}}()
    for point in calculatePoints
        reflectDiagonal = map2DTo1D(reverse(map1DTo2D(point, size_BZ))..., size_BZ)
        reflectFS = map2DTo1D((-1 .* reverse(map1DTo2D(point, size_BZ)) .+ [-π, -π])..., size_BZ)
        reflectBoth = map2DTo1D((-1 .* map1DTo2D(point, size_BZ) .+ [-π, -π])..., size_BZ)
        oppositePoints[point] = [reflectDiagonal, reflectFS, reflectBoth]
    end
    
    desc = "W=$(round(hamiltDetails["W_val"], digits=3))"
    corrResults = @showprogress desc=desc @distributed (d1, d2) -> mergewith(+, d1, d2) for pivotPoint in calculatePoints
        corrNode = iterDiagResults(hamiltDetails, maxSize, [pivotPoint], symmetricPairsNode, copy(correlationFuncDict), copy(vneFuncDict), copy(mutInfoFuncDict), bathIntLegs, noSelfCorr, addPerStep)
        corrAntiNode = iterDiagResults(hamiltDetails, maxSize, [pivotPoint], symmetricPairsAntiNode, copy(correlationFuncDict), copy(vneFuncDict), copy(mutInfoFuncDict), bathIntLegs, noSelfCorr, addPerStep)
        avgCorr = mergewith(+, corrNode, corrAntiNode)
        map!(v -> v ./ 2, values(avgCorr))
        avgCorr
    end

    corrResults = PropagateIndices(calculatePoints, corrResults, size_BZ, oppositePoints)

    corrResultsBool = Dict()
    for (name, results) in corrResults
        @assert !any(isnan.(results))
        corrResultsBool[name] = [abs(r) ≤ 1e-6 ? -1 : 1 for r in results]
    end
    return corrResults, corrResultsBool
end


function localSpecFunc(
        hamiltDetails::Dict,
        numShells::Int64,
        specFuncDictFunc::Function,
        freqValues::Vector{Float64},
        standDev::Union{Vector{Float64}, Float64},
        maxSize::Int64;
        bathIntLegs::Int64=2,
        addPerStep::Int64=1,
    )
    size_BZ = hamiltDetails["size_BZ"]

    cutoffEnergy = hamiltDetails["dispersion"][div(size_BZ - 1, 2) + 2 - numShells]

    # pick out k-states from the southwest quadrant that have positive energies 
    # (hole states can be reconstructed from them (p-h symmetry))
    SWIndices = [p for p in 1:size_BZ^2 if map1DTo2D(p, size_BZ)[1] < 0
                 && map1DTo2D(p, size_BZ)[2] ≤ 0 
                 && abs(cutoffEnergy) ≥ abs(dispersion[p])
                ]

    distancesFromNode = [sum((map1DTo2D(p, size_BZ) .- (-π/2, -π/2)) .^ 2)^0.5 for p in SWIndices]
    sortedPoints = SWIndices[sortperm(distancesFromNode)]
    
    siamSpecDict, kondoSpecDict = specFuncDictFunc(length(sortedPoints))
    specFunc1 = iterDiagSpecFunc(hamiltDetails, maxSize, sortedPoints, siamSpecDict, 
                                    bathIntLegs, addPerStep, freqValues, standDev)
    specFunc2 = iterDiagSpecFunc(hamiltDetails, maxSize, sortedPoints, kondoSpecDict, 
                                 bathIntLegs, addPerStep, freqValues, standDev) / length(sortedPoints)
    specFunc = specFunc1 + specFunc2
    specFunc ./= sum(specFunc .* (maximum(freqValues) - minimum(freqValues)) / (length(freqValues)-1))

    return specFunc
end


function correlationMap2Point(
        hamiltDetails::Dict,
        numShells::Int64,
        correlationFuncDict::Dict,
        maxSize::Int64;
        probePoints::Vector{Int64}=Int64[],
        bathIntLegs::Int64=2,
    )
    size_BZ = hamiltDetails["size_BZ"]

    # initialise zero array for storing correlations
    
    cutoffEnergy = hamiltDetails["dispersion"][div(size_BZ - 1, 2) + 2 - numShells]

    # pick out k-states from the southwest quadrant that have positive energies 
    # (hole states can be reconstructed from them (p-h symmetry))
    SWIndices = [p for p in 1:size_BZ^2 if map1DTo2D(p, size_BZ)[1] < 0
                 && map1DTo2D(p, size_BZ)[2] ≤ 0 
                 && cutoffEnergy ≥ dispersion[p] ≥ 0
                ]

    distancesFromNode = [sum((map1DTo2D(p, size_BZ) .- (-π/2, -π/2)) .^ 2)^0.5 for p in SWIndices]
    symmetricPairsNode = SWIndices[sortperm(distancesFromNode)]
    distancesFromAntiNode = [minimum([sum((map1DTo2D(p, size_BZ) .- (-π, 0.)) .^ 2)^0.5,
                                      sum((map1DTo2D(p, size_BZ) .- (0., -π)) .^ 2)^0.5])
                                     for p in SWIndices]
    symmetricPairsAntiNode = SWIndices[sortperm(distancesFromAntiNode)]
    calculatePoints = filter(p -> map1DTo2D(p, size_BZ)[1] ≤ map1DTo2D(p, size_BZ)[2], SWIndices)
    
    oppositePoints = Dict{Int64, Vector{Int64}}()
    for point in calculatePoints
        reflectDiagonal = map2DTo1D(reverse(map1DTo2D(point, size_BZ))..., size_BZ)
        reflectFS = map2DTo1D((-1 .* reverse(map1DTo2D(point, size_BZ)) .+ [-π, -π])..., size_BZ)
        reflectBoth = map2DTo1D((-1 .* map1DTo2D(point, size_BZ) .+ [-π, -π])..., size_BZ)
        oppositePoints[point] = [reflectDiagonal, reflectFS, reflectBoth]
    end

    twoPointResults = Dict(k => zeros(length(probePoints), size_BZ^2) for k in keys(correlationFuncDict))

    if isempty(probePoints)
        probePoints = sort(calculatePoints)
    end
    for pivotIndex in eachindex(probePoints)
        desc = "W=$(round(hamiltDetails["W_val"], digits=3))"
        results = @showprogress desc=desc @distributed (d1, d2) -> mergewith(+, d1, d2) for pivotPoint2 in calculatePoints
            for k in keys(correlationFuncDict)
                correlationFuncDict[k][1] = probePoints[pivotIndex]
            end
            corrNode = iterDiagResults(hamiltDetails, maxSize, [pivotPoint2], symmetricPairsNode, copy(correlationFuncDict), Dict(), Dict(), bathIntLegs)
            corrAntiNode = iterDiagResults(hamiltDetails, maxSize, [pivotPoint2], symmetricPairsAntiNode, copy(correlationFuncDict), Dict(), Dict(), bathIntLegs)
            avgCorr = mergewith(+, corrNode, corrAntiNode)
            map!(v -> v ./ 2, values(avgCorr))
            avgCorr
        end
        results = PropagateIndices(calculatePoints, results, size_BZ, oppositePoints)
        for (k, v) in results
            twoPointResults[k][pivotIndex, :] .= v
        end
    end

    twoPointResultsBool = Dict()
    for (name, results) in twoPointResults
        @assert !any(isnan.(results))
        twoPointResultsBool[name] = map(r -> abs(r) ≤ 1e-6 ? -1 : 1, results)
    end
    return twoPointResults, twoPointResultsBool
end


function CorrelationTiling(
        corrResults::Vector{Float64}
    )
    tiledResults = zeros(repeat(length(corrResults), 2)...)
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
                    kondoJArrayFull, dispersion = momentumSpaceRG(size_BZ, omega_by_t, J_val, J_val * W_by_J, orbitals)
                    results, results_bool = mapProbeNameToProbe("scattProb", size_BZ, kondoJArrayFull, W_by_J * J_val, dispersion, orbitals)
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


function PhaseIndex(
        J_val::Float64,
        W_val::Float64,
        fermiPoints::Vector{Int64},
    )
    kondoJArray, dispersion = momentumSpaceRG(size_BZ, omega_by_t, J_val, W_val, orbitals)
    averageKondoScale = sum(abs.(kondoJArray[:, :, 1])) / length(kondoJArray[:, :, 1])
    @assert averageKondoScale > RG_RELEVANCE_TOL
    kondoJArray[:, :, end] .= ifelse.(abs.(kondoJArray[:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, kondoJArray[:, :, end], 0)
    scattProbBool = scattProb(size_BZ, kondoJArray, dispersion)[2]
    if all(>(0), scattProbBool[fermiPoints])
        return 1
    elseif !all(==(0), scattProbBool[fermiPoints])
        return 2
    else
        return 3
    end
end


function PhaseBounds(
        J_val::Float64,
        W_val_bounds::Vector{Float64},
        fermiPoints::Vector{Int64},
        phaseIndices::NTuple{2, Int64},
        tolerance::Float64;
        maxIter=100,
    )
    @assert tolerance > 0
    currentPhaseIndices = [PhaseIndex(J_val, W_val_bounds[1], fermiPoints), PhaseIndex(J_val, W_val_bounds[2], fermiPoints)]
    @assert currentPhaseIndices[1] ≤ phaseIndices[1] && currentPhaseIndices[2] ≥ phaseIndices[2]
    @assert 2 ∈ phaseIndices
    numIter = 1
    while abs(W_val_bounds[1] - W_val_bounds[2]) > tolerance && numIter < maxIter
        new_W_val = 0.5 * sum(W_val_bounds)
        newPhaseIndex = PhaseIndex(J_val, new_W_val, fermiPoints)
        if newPhaseIndex == currentPhaseIndices[1] || newPhaseIndex == phaseIndices[1]
            currentPhaseIndices[1] = newPhaseIndex
            W_val_bounds[1] = new_W_val
        else
            currentPhaseIndices[2] = newPhaseIndex
            W_val_bounds[2] = new_W_val
        end
        numIter += 1
    end
    return 0.5 * sum(W_val_bounds)
end

function PhaseDiagram(kondoJVals::Vector{Float64}, bathIntVals::Vector{Float64}, phaseMaps::Dict{String, Int64})
    @assert issorted(bathIntVals, rev=true)
    tolerance = (maximum(bathIntVals) - minimum(bathIntVals)) / (length(bathIntVals) - 1)
    densityOfStates, dispersionArray = getDensityOfStates(tightBindDisp, size_BZ)
    fermiPoints = unique(getIsoEngCont(dispersionArray, 0.0))
    @assert length(fermiPoints) == 2 * size_BZ - 2
    @assert all(==(0), dispersionArray[fermiPoints])

    phaseDiagram = fill(0, (length(kondoJVals), length(bathIntVals)))
    @showprogress Threads.@threads for (i, kondoJ) in collect(enumerate(kondoJVals))
        bathIntPGStart = PhaseBounds(kondoJ, [bathIntVals[1], bathIntVals[end]], fermiPoints, (1, 2), 1e-3)
        bathIntPGStop = PhaseBounds(kondoJ, [bathIntVals[1], bathIntVals[end]], fermiPoints, (2, 3), 1e-3)
        phaseDiagram[i, bathIntVals .≥ bathIntPGStart] .= phaseMaps["L-FL"]
        phaseDiagram[i, bathIntPGStart .≥ bathIntVals .≥ bathIntPGStop] .= phaseMaps["L-PG"]
        phaseDiagram[i, bathIntPGStop .≥ bathIntVals] .= phaseMaps["LM"]
    end
    return phaseDiagram
end
