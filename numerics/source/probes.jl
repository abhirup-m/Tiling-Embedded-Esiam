##### Functions for calculating various probes          #####
##### (correlation functions, Greens functions, etc)    #####

@everywhere using ProgressMeter, Combinatorics, Fermions
#=include("/home/abhirup/storage/programmingProjects/fermions.jl/src/iterDiag.jl")=#

"""
Function to calculate the total Kondo scattering probability Γ(k) = ∑_q J(k,q)^2
at the RG fixed point.
"""

@everywhere function scattProb(
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
function kondoCoupMap(
        kx_ky::Tuple{Float64,Float64},
        size_BZ::Int64,
        kondoJArrayFull::Array{Float64,3};
        mapAmong::Union{Function, Nothing}=nothing
    )
    kspacePoint = map2DTo1D(kx_ky..., size_BZ)
    otherPoints = collect(1:size_BZ^2)
    if !isnothing(mapAmong)
        filter!(p -> mapAmong(map1DTo2D(p, size_BZ)...), otherPoints)
    end
    results = zeros(size_BZ^2) 
    results[otherPoints] .= kondoJArrayFull[kspacePoint, otherPoints, end]
    results_bare = kondoJArrayFull[kspacePoint, :, 1]
    results_bool = [r / r_b <= RG_RELEVANCE_TOL ? -1 : 1 for (r, r_b) in zip(results, results_bare)]
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
                             globalField=hamiltDetails["globalField"],
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
        freqValues::Vector{Float64},
        standDev::Union{Float64, Vector{Float64}};
        addPerStep::Int64=2,
        silent::Bool=false,
        broadFuncType::String="gauss",
        normEveryStep::Bool=true,
    )

    bathIntFunc = points -> hamiltDetails["bathIntForm"](hamiltDetails["W_val"], 
                                                         hamiltDetails["orbitals"][2],
                                                         hamiltDetails["size_BZ"],
                                                         points)
                            
    #=hybridisationArray = 0.0 .* [(sort(hamiltDetails["kondoJArray"][point, sortedPoints], by=abs, rev=true)[1] * impCorr) for point in sortedPoints]=#
    #=localField = ifelse(hamiltDetails["kondoJArray"][sortedPoints, sortedPoints] .|> abs |> maximum == 0, -4., 0.)=#
    hamiltonian = KondoModel(
                             hamiltDetails["dispersion"][sortedPoints],
                             hamiltDetails["kondoJArray"][sortedPoints, sortedPoints],
                             sortedPoints, bathIntFunc;
                             bathIntLegs=bathIntLegs,
                             globalField=hamiltDetails["globalField"],
                             #=impurityField=localField,=#
                             couplingTolerance=1e-10,
                            )

    # impurity local terms
    impCorr = 14.
    push!(hamiltonian, ("n",  [1], -impCorr/2)) # Ed nup
    push!(hamiltonian, ("n",  [2], -impCorr/2)) # Ed ndown
    push!(hamiltonian, ("nn",  [1, 2], impCorr)) # U nup ndown

    indexPartitions = [10]
    while indexPartitions[end] < 2 + 2 * length(sortedPoints)
        push!(indexPartitions, indexPartitions[end] + 2 * addPerStep)
    end
    hamiltonianFamily = MinceHamiltonian(hamiltonian, indexPartitions)

    savePaths, resultsDict, specFuncOperators = IterDiag(
                                                         hamiltonianFamily, 
                                                         maxSize;
                                                         symmetries=Char['N', 'S'],
                                                         #=magzReq=(m, N) -> -1 ≤ m ≤ 2,=#
                                                         #=occReq=(x, N) -> div(N, 2) - 3 ≤ x ≤ div(N, 2) + 3,=#
                                                         silent=silent,
                                                         maxMaxSize=maxSize,
                                                         specFuncDefDict=specFuncDict,
                                                        ) 
    totalSpecFunc, specFuncMatrix = IterSpecFunc(savePaths, specFuncOperators, 
                                 freqValues, standDev;
                                 normEveryStep=normEveryStep, degenTol=1e-10, 
                                 silent=true, broadFuncType=broadFuncType,
                                 returnEach=true,
                           )
    return totalSpecFunc

end

@everywhere function correlationMap(
        hamiltDetails::Dict,
        numShells::Int64,
        correlationFuncDict::Dict,
        maxSize::Int64,
        savePath::String;
        vneFuncDict::Dict=Dict(),
        mutInfoFuncDict::Dict=Dict(),
        bathIntLegs::Int64=2,
        noSelfCorr::Vector{String}=String[],
        addPerStep::Int64=1,
        numProcs::Int64=1,
        loadData::Bool=false,
    )

    size_BZ = hamiltDetails["size_BZ"]

    cutoffEnergy = hamiltDetails["dispersion"][div(size_BZ - 1, 2) + 2 - numShells]

    # pick out k-states from the southwest quadrant that have positive energies 
    # (hole states can be reconstructed from them (p-h symmetry))
    SWIndices = [p for p in 1:size_BZ^2 if map1DTo2D(p, size_BZ)[1] < 0
                 && map1DTo2D(p, size_BZ)[2] ≤ 0 
                 && abs(cutoffEnergy) ≥ abs(hamiltDetails["dispersion"][p])
                ]

    calculatePoints = filter(p -> map1DTo2D(p, size_BZ)[1] ≤ map1DTo2D(p, size_BZ)[2], SWIndices)

    oppositePoints = Dict{Int64, Vector{Int64}}()
    for point in calculatePoints
        reflectDiagonal = map2DTo1D(reverse(map1DTo2D(point, size_BZ))..., size_BZ)
        reflectFS = map2DTo1D((-1 .* reverse(map1DTo2D(point, size_BZ)) .+ [-π, -π])..., size_BZ)
        reflectBoth = map2DTo1D((-1 .* map1DTo2D(point, size_BZ) .+ [-π, -π])..., size_BZ)
        oppositePoints[point] = [reflectDiagonal, reflectFS, reflectBoth]
    end

    if isfile(savePath) && loadData
        #=corrResults = deserialize(savePath)=#
        corrResults = Dict{String, Vector{Float64}}()
        for (k, v) in load(savePath)
            corrResults[k] = v
        end
        @assert issetequal(collect(keys(corrResults)), vcat(([correlationFuncDict, vneFuncDict, mutInfoFuncDict] .|> keys .|> collect)...))
        corrResults = PropagateIndices(calculatePoints, corrResults, size_BZ, oppositePoints)
        corrResultsBool = Dict()
        for (name, results) in corrResults
            @assert !any(isnan.(results))
            corrResultsBool[name] = [abs(r) ≤ 1e-6 ? -1 : 1 for r in results]
        end
        return corrResults, corrResultsBool
    end

    distancesFromNode = [sum((map1DTo2D(p, size_BZ) .- (-π/2, -π/2)) .^ 2)^0.5 for p in SWIndices]
    symmetricPairsNode = SWIndices[sortperm(distancesFromNode)]
    distancesFromAntiNode = [minimum([sum((map1DTo2D(p, size_BZ) .- (-π, 0.)) .^ 2)^0.5,
                                      sum((map1DTo2D(p, size_BZ) .- (0., -π)) .^ 2)^0.5])
                                     for p in SWIndices
                                    ]
    symmetricPairsAntiNode = SWIndices[sortperm(distancesFromAntiNode)]
    
    desc = "W=$(round(hamiltDetails["W_val"], digits=3))"

    corrNode = @showprogress pmap(pivotPoint -> iterDiagResults(hamiltDetails, maxSize, [pivotPoint], symmetricPairsNode, 
                                                                correlationFuncDict, vneFuncDict, mutInfoFuncDict, 
                                                                bathIntLegs, noSelfCorr, addPerStep),
                                  WorkerPool(1:numProcs), calculatePoints)
    corrAntiNode = @showprogress pmap(pivotPoint -> iterDiagResults(hamiltDetails, maxSize, [pivotPoint], symmetricPairsAntiNode, 
                                                                    correlationFuncDict, vneFuncDict, mutInfoFuncDict, bathIntLegs, 
                                                                    noSelfCorr, addPerStep), 
                                      WorkerPool(1:numProcs), calculatePoints)
    corrResults = mergewith(+, corrNode..., corrAntiNode...)
    map!(v -> v ./ 2, values(corrResults))

    if !isempty(savePath)
        mkpath(SAVEDIR)
        jldopen(savePath, "w"; compress = true) do file
            for (k, v) in corrResults
                file[k] = v
            end
        end
    end

    corrResults = PropagateIndices(calculatePoints, corrResults, size_BZ, oppositePoints)

    corrResultsBool = Dict()
    for (name, results) in corrResults
        @assert !any(isnan.(results))
        corrResultsBool[name] = [abs(r) ≤ 1e-6 ? -1 : 1 for r in results]
    end
    return corrResults, corrResultsBool
end


function localSpecFuncAverage(
        argsNode,
        argsAntinode
    )
    specFuncOuter = iterDiagSpecFunc(argsNode...)
    specFuncOuter .+= iterDiagSpecFunc(argsAntinode...)
    return specFuncOuter
end


function localSpecFunc(
        hamiltDetails::Dict,
        numShells::Int64,
        specFuncDictFunc::Function,
        freqValues::Vector{Float64},
        standDevInner::Union{Vector{Float64}, Float64},
        standDevOuter::Union{Vector{Float64}, Float64},
        maxSize::Int64;
        resonanceHeight::Float64=0.,
        heightTolerance::Float64=1e-4,
        bathIntLegs::Int64=2,
        addPerStep::Int64=1,
        maxIter::Int64=20,
        broadFuncType::String="gauss",
    )
    if resonanceHeight < 0
        resonanceHeight = 0
    end
    if resonanceHeight > 0
        @assert typeof(standDevInner) == Float64
    end
    size_BZ = hamiltDetails["size_BZ"]
    cutoffEnergy = hamiltDetails["dispersion"][div(size_BZ - 1, 2) + 2 - numShells]

    # pick out k-states from the southwest quadrant that have positive energies 
    # (hole states can be reconstructed from them (p-h symmetry))
    SWIndices = [p for p in 1:size_BZ^2 if 
                 #=map1DTo2D(p, size_BZ)[1] ≤ 0 &&=#
                 #=map1DTo2D(p, size_BZ)[2] ≤ 0 &&=#
                 map1DTo2D(p, size_BZ)[1] ≤ map1DTo2D(p, size_BZ)[2] &&
                 abs(cutoffEnergy) ≥ abs(hamiltDetails["dispersion"][p])
                ]
    distancesFromNode = [minimum([sum((map1DTo2D(p, size_BZ) .- node) .^ 2)^0.5 for node in NODAL_POINTS])
                        for p in SWIndices
                       ]
    distancesFromAntiNode = [minimum([sum((map1DTo2D(p, size_BZ) .- antinode) .^ 2)^0.5
                                     for antinode in ANTINODAL_POINTS])
                            for p in SWIndices
                           ]
    sortedPointsNode = SWIndices[sortperm(distancesFromNode)]
    sortedPointsAntiNode = SWIndices[sortperm(distancesFromAntiNode)]
    @assert length(sortedPointsAntiNode) == length(sortedPointsNode)
    
    siamSpecDictNode, kondoSpecDictNode = specFuncDictFunc(length(sortedPointsNode))
    siamSpecDictAntiNode, kondoSpecDictAntiNode = specFuncDictFunc(length(sortedPointsAntiNode))
    specFunc = zeros(length(freqValues))
    specFuncArgsNode = [
                    hamiltDetails, maxSize, sortedPointsNode,
                    siamSpecDictNode, bathIntLegs, addPerStep,
                    freqValues, standDevOuter, true, "gauss"
                   ]
    specFuncArgsAntinode = deepcopy(specFuncArgsNode)
    specFuncArgsAntinode[3] = sortedPointsAntiNode
    specFuncOuter = localSpecFuncAverage(specFuncArgsNode, specFuncArgsAntinode)

    error = 1.
    numIter = 0
    increment = 0.
    specFuncArgsNode[4], specFuncArgsNode[end] = kondoSpecDictNode, "lorentz"
    specFuncArgsAntinode[4], specFuncArgsAntinode[end] = kondoSpecDictAntiNode, "lorentz"
    while abs(error) > heightTolerance && numIter < maxIter
        numIter += 1
        specFuncArgsNode[end-2], specFuncArgsAntinode[end-2] = standDevInner, standDevInner
        specFuncInner = localSpecFuncAverage(specFuncArgsNode, specFuncArgsAntinode) / length(sortedPointsNode)
        specFunc = specFuncInner + specFuncOuter
        specFunc ./= sum(specFunc .* (maximum(freqValues) - minimum(freqValues)) / (length(freqValues)-1))
        if resonanceHeight == 0
            break
        end

        newError = (specFunc[freqValues .≥ 0][1] - resonanceHeight) / resonanceHeight
        if numIter == 1
            standDevInner = (1 + newError) * standDevInner
            increment = 0.1 * standDevInner
            error = newError
            display((standDevInner, error, increment))
            numIter += 1
            continue
        end

        if abs(error - newError) > newError || error * newError < 0
            increment /= 2
        end
        error = newError
        if error > 0
            if standDevInner ≤ increment
                increment /= 2
            end
            standDevInner += increment
        else
            standDevInner -= increment
        end
        display((standDevInner, error, increment))
    end

    if resonanceHeight > 0 && (specFunc[freqValues .≥ 0][1] - resonanceHeight) / resonanceHeight < heightTolerance
        println("Converged in $(numIter-1) runs: η=$(standDevInner)")
    end
    if resonanceHeight > 0 && (specFunc[freqValues .≥ 0][1] - resonanceHeight) / resonanceHeight > heightTolerance
        println("Failed to converge: error=$((specFunc[freqValues .≥ 0][1] - resonanceHeight) / resonanceHeight)")
    end

    return specFunc, standDevInner
end


function kspaceLocalSpecFunc(
        hamiltDetails::Dict,
        numShells::Int64,
        specFuncDictFunc::Function,
        freqValues::Vector{Float64},
        standDevInner::Union{Vector{Float64}, Float64},
        standDevOuter::Union{Vector{Float64}, Float64},
        maxSize::Int64,
        kspacePoint::Vector{Float64};
        bathIntLegs::Int64=2,
        addPerStep::Int64=1,
        maxIter::Int64=20,
        broadFuncType::String="gauss",
    )
    size_BZ = hamiltDetails["size_BZ"]
    cutoffEnergy = hamiltDetails["dispersion"][div(size_BZ - 1, 2) + 2 - numShells]

    # pick out k-states from the southwest quadrant that have positive energies 
    # (hole states can be reconstructed from them (p-h symmetry))
    SWIndices = [p for p in 1:size_BZ^2 if 
                 #=map1DTo2D(p, size_BZ)[1] ≤ 0 &&=#
                 #=map1DTo2D(p, size_BZ)[2] ≤ 0 &&=#
                 map1DTo2D(p, size_BZ)[1] ≤ map1DTo2D(p, size_BZ)[2] &&
                 abs(cutoffEnergy) ≥ abs(hamiltDetails["dispersion"][p])
                ]
    rotationMatrix(rotateAngle) = [cos(rotateAngle) -sin(rotateAngle); sin(rotateAngle) cos(rotateAngle)]
    equivalentPoints = [rotationMatrix(rotateAngle) * kspacePoint for rotateAngle in (0, π/2, π, 3π/2)]
    distancesFromPivot = [minimum([sum((map1DTo2D(p, size_BZ) .- pivot) .^ 2)^0.5 for pivot in equivalentPoints])
                        for p in SWIndices
                       ]
    sortedPoints = SWIndices[sortperm(distancesFromPivot)]
    
    specDictSet = specFuncDictFunc(length(sortedPoints), (3, 4))
    if hamiltDetails["kondoJArray"][sortedPoints[1], sortedPoints] .|> abs |> maximum == 0 
        delete!(specDictSet, "Sd+")
        delete!(specDictSet, "Sd-")
    end

    specFunc = zeros(length(freqValues))
    for (name, specFuncDict) in specDictSet
        standDev = ifelse(name ∈ ("Sd+", "Sd-"), standDevInner, standDevOuter)
        broadType = ifelse(name ∈ ("Sd+", "Sd-"), "lorentz", "gauss")
        specFunc .+= iterDiagSpecFunc(hamiltDetails, maxSize, sortedPoints,
                                      specFuncDict, bathIntLegs, freqValues, 
                                      standDev; addPerStep=addPerStep,
                                      silent=false, broadFuncType=broadType,
                                      normEveryStep=false,
                                    )
    end
    specFunc ./= sum(specFunc .* (maximum(freqValues) - minimum(freqValues)) / (length(freqValues)-1))

    return specFunc
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


@everywhere function PhaseIndex(
        size_BZ::Int64,
        omega_by_t::Float64,
        J_val::Float64,
        W_val::Float64,
        fermiPoints::Vector{Int64},
    )
    kondoJArray, dispersion = momentumSpaceRG(size_BZ, omega_by_t, J_val, W_val, orbitals; saveData=false)
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


@everywhere function CriticalBathInt(
        size_BZ::Int64,
        omega_by_t::Float64,
        kondoJ::Float64,
        transitionWindow::Vector{Float64},
        fermiPoints::Vector{Int64},
        tolerance::Float64;
        maxIter=100,
        loadData::Bool=false,
    )
    @assert tolerance > 0
    criticalBathInt = Float64[]
    @assert issorted(transitionWindow, rev=true)
    for phaseBoundType in [(1, 2), (2, 3)]
        currentTransitionWindow = copy(transitionWindow)
        currentPhaseIndices = [PhaseIndex(size_BZ, omega_by_t, kondoJ, W_val, fermiPoints) for W_val in currentTransitionWindow]
        @assert currentPhaseIndices[1] ≤ phaseBoundType[1] && currentPhaseIndices[2] ≥ phaseBoundType[2]
        @assert 2 ∈ phaseBoundType
        numIter = 1
        while abs(currentTransitionWindow[1] - currentTransitionWindow[2]) > tolerance && numIter < maxIter
            updatedEdge = 0.5 * sum(currentTransitionWindow)
            newPhaseIndex = PhaseIndex(size_BZ, omega_by_t, kondoJ, updatedEdge, fermiPoints)
            if newPhaseIndex == currentPhaseIndices[1] || newPhaseIndex == phaseBoundType[1]
                currentPhaseIndices[1] = newPhaseIndex
                currentTransitionWindow[1] = updatedEdge
            else
                currentPhaseIndices[2] = newPhaseIndex
                currentTransitionWindow[2] = updatedEdge
            end
            numIter += 1
        end
        push!(criticalBathInt, 0.5 * sum(currentTransitionWindow))
    end
    return criticalBathInt
end

function PhaseDiagram(
        size_BZ::Int64,
        omega_by_t::Float64,
        kondoJVals::Vector{Float64}, 
        bathIntVals::Vector{Float64}, 
        tolerance::Float64,
        phaseMaps::Dict{String, Int64};
        loadData::Bool=false,
    )
    @assert issorted(kondoJVals)
    densityOfStates, dispersionArray = getDensityOfStates(tightBindDisp, size_BZ)
    fermiPoints = unique(getIsoEngCont(dispersionArray, 0.0))
    @assert length(fermiPoints) == 2 * size_BZ - 2
    @assert all(==(0), dispersionArray[fermiPoints])

    phaseDiagram = fill(0, (length(kondoJVals), length(bathIntVals)))
    savePath = joinpath(SAVEDIR, "crit-bathint-$(size_BZ).jld2") 
    if loadData && ispath(savePath)
        loadedData = load(savePath)
    else
        loadedData = Dict()
    end
    keyFunc(kondoJ) = "$(tolerance)-$(kondoJ)"
    criticalBathIntResults = @showprogress pmap(kondoJ -> keyFunc(kondoJ) ∈ keys(loadedData) ? loadedData[keyFunc(kondoJ)] : CriticalBathInt(size_BZ, omega_by_t, kondoJ, [maximum(bathIntVals), minimum(bathIntVals)], fermiPoints, tolerance; loadData=loadData), kondoJVals)
    for (i, (PGStart, PGStop)) in enumerate(criticalBathIntResults)
        phaseDiagram[i, bathIntVals .≥ PGStart] .= phaseMaps["L-FL"]
        phaseDiagram[i, PGStart .≥ bathIntVals .≥ PGStop] .= phaseMaps["L-PG"]
        phaseDiagram[i, PGStop .≥ bathIntVals] .= phaseMaps["LM"]
    end

    mkpath(SAVEDIR)
    jldopen(savePath, "w"; compress = true) do f
        for (kondoJ, r) in zip(kondoJVals, criticalBathIntResults)
            f[keyFunc(kondoJ)] = (r[1], r[2])
        end
    end
    return phaseDiagram
end
