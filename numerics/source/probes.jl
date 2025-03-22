##### Functions for calculating various probes          #####
##### (correlation functions, Greens functions, etc)    #####

@everywhere using ProgressMeter, Combinatorics, Serialization, Fermions

"""
Function to calculate the total Kondo scattering probability Γ(k) = ∑_q J(k,q)^2
at the RG fixed point.
"""

@everywhere function ScattProb(
        size_BZ::Int64,
        kondoJArray::Array{Float64,3},
        dispersion::Vector{Float64},
    )

    # allocate zero arrays to store Γ at fixed point and for the bare Hamiltonian.
    results_scaled = zeros(size_BZ^2)

    # loop over all points k for which we want to calculate Γ(k).
    for point in 1:size_BZ^2
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
function KondoCoupMap(
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
    reference = sum(abs.(results_bare)) / length(results_bare)
    @assert reference > RG_RELEVANCE_TOL
    results_bool = [abs(r / reference) ≤ RG_RELEVANCE_TOL ? -1 : 1 for (r, r_b) in zip(results, results_bare)]
    return results, results_bare, results_bool
end


@everywhere function IterDiagResults(
        hamiltDetails::Dict,
        maxSize::Int64,
        pivotPoints::Vector{Int64},
        sortedPoints::Vector{Int64},
        correlationFuncDict::Dict{String, Tuple{Union{Nothing, Int64}, Function}},
        vneFuncDict::Dict,
        mutInfoFuncDict::Dict,
        bathIntLegs::Int64,
        noSelfCorr::Vector{String},
        addPerStep::Int64,
    )
    allKeys = vcat(keys(correlationFuncDict)..., keys(vneFuncDict)..., keys(mutInfoFuncDict)...)
    corrResults = Dict{String, Vector{Float64}}(k => repeat([NaN], hamiltDetails["size_BZ"]^2) for k in allKeys)

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
            if partyA ≠ partyB
                mutInfoDefDict[name * join(pivotIndex)] = (partyA, partyB)
                mapCorrNameToIndex[name * join(pivotIndex)] = (name, pointsSequence[pivotIndex])
            end
        end
    end

    bathIntFunc = points -> hamiltDetails["bathIntForm"](hamiltDetails["W_val"], 
                                                         hamiltDetails["orbitals"][2],
                                                         hamiltDetails["size_BZ"],
                                                         points)
    if "chemPot" in keys(hamiltDetails)
        hamiltDetails["dispersion"][pointsSequence] .+= hamiltDetails["chemPot"]
    end    
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
    @assert all(!isempty, hamiltonianFamily)
    for hamiltonian in hamiltonianFamily
        @assert all(!isempty, hamiltonian)
    end
    iterDiagResults = nothing
    id = nothing
    while true
        output = IterDiag(
                          hamiltonianFamily, 
                          maxSize;
                          symmetries=Char['N', 'S'],
                          #=magzReq=(m, N) -> -1 ≤ m ≤ 2,=#
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


@everywhere function IterDiagSpecFunc(
        hamiltDetails::Dict,
        maxSize::Int64,
        sortedPoints::Vector{Int64},
        specFuncDict::Dict{String, Vector{Tuple{String, Vector{Int64}, Float64}}},
        bathIntLegs::Int64,
        freqValues::Vector{Float64},
        standDev::Dict{String, Union{Float64, Vector{Float64}}};
        addPerStep::Int64=2,
        silent::Bool=false,
        broadFuncType::Union{String, Dict{String, String}}="gauss",
        normEveryStep::Bool=true,
        onlyCoeffs::Vector{String}=String[],
    )
    if typeof(broadFuncType) == String
        broadFuncType = Dict(name => broadFuncType for name in specFuncDict)
    end

    bathIntFunc = points -> hamiltDetails["bathIntForm"](0.,
                                                         # hamiltDetails["W_val"],
                                                         hamiltDetails["orbitals"][2],
                                                         hamiltDetails["size_BZ"],
                                                         points)
                            
    hamiltonian = KondoModel(
                             hamiltDetails["dispersion"][sortedPoints],
                             hamiltDetails["kondoJArray"][sortedPoints, sortedPoints] ./ length(sortedPoints),
                             sortedPoints, bathIntFunc;
                             bathIntLegs=bathIntLegs,
                             globalField=hamiltDetails["globalField"],
                             couplingTolerance=1e-10,
                            )

    kondoTemp = (sum(hamiltDetails["kondoJArray"][sortedPoints[1], sortedPoints] .|> abs) / sum(hamiltDetails["kondoJArray_bare"][sortedPoints[1], sortedPoints] .|> abs))^0.6

    # impurity local terms
    push!(hamiltonian, ("n",  [1], -hamiltDetails["imp_corr"]/2)) # Ed nup
    push!(hamiltonian, ("n",  [2], -hamiltDetails["imp_corr"]/2)) # Ed ndown
    push!(hamiltonian, ("nn",  [1, 2], hamiltDetails["imp_corr"])) # U nup ndown
    excludeRange = (1.0 * maximum(hamiltDetails["kondoJArray_bare"]), hamiltDetails["imp_corr"]/4)

    indexPartitions = [4]
    while indexPartitions[end] < 2 + 2 * length(sortedPoints)
        push!(indexPartitions, indexPartitions[end] + 2 * addPerStep)
    end
    hamiltonianFamily = MinceHamiltonian(hamiltonian, indexPartitions)
    
    savePaths, resultsDict, specFuncOperators = IterDiag(
                                                         hamiltonianFamily, 
                                                         maxSize;
                                                         symmetries=Char['N', 'S'],
                                                         magzReq=(m, N) -> -3 ≤ m ≤ 4,
                                                         occReq=(x, N) -> div(N, 2) - 4 ≤ x ≤ div(N, 2) + 4,
                                                         silent=silent,
                                                         maxMaxSize=maxSize,
                                                         specFuncDefDict=specFuncDict,
                                                        ) 

    specFuncResults = Dict()
    for (name, operator) in specFuncOperators
        if name ∈ onlyCoeffs
            specCoeffs = IterSpectralCoeffs(savePaths, operator;
                                            degenTol=1e-10, silent=silent,
                                            #=excludeLevels= E -> excludeRange[1] < E < excludeRange[2],=#
                                           )
            scaledSpecCoeffs = NTuple{2, Float64}[(weight * kondoTemp, pole) for (weight, pole) in vcat(specCoeffs...)]
            specFuncResults[name] = scaledSpecCoeffs
        else
            specFunc = IterSpecFunc(savePaths, operator, freqValues, 
                                                    standDev[name]; normEveryStep=normEveryStep, 
                                                    degenTol=1e-10, silent=false, 
                                                    broadFuncType=broadFuncType[name],
                                                    returnEach=false, normalise=false,
                                                   )
            specFuncResults[name] = specFunc
        end
    end
    return specFuncResults

end

@everywhere function AuxiliaryCorrelations(
        hamiltDetails::Dict,
        numShells::Int64,
        correlationFuncDict::Dict{String, Tuple{Union{Nothing, Int64}, Function}},
        maxSize::Int64;
        savePath::Union{Nothing, Function}=nothing,
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

    if !isnothing(savePath) && loadData
        corrResults = Dict{String, Vector{Float64}}()
        allCorrelationKeys = vcat(([correlationFuncDict, vneFuncDict, mutInfoFuncDict] .|> keys .|> collect)...)
        for corrName in allCorrelationKeys
            savePathCorr = savePath(corrName)
            if isfile(savePathCorr)
                corrResults[corrName] = load(savePathCorr)[corrName]
            end
        end
        if length(corrResults) == length(allCorrelationKeys)
            corrResultsBool = Dict()
            for (name, results) in corrResults
                corrResultsBool[name] = [ifelse(isnan(r), r, abs(r) ≤ 1e-6 ? -1 : 1) for r in results]
            end
            println("Collected from saved data.")
            return corrResults, corrResultsBool
        end
    end

    distancesFromNode = [sum((map1DTo2D(p, size_BZ) .- (-π/2, -π/2)) .^ 2)^0.5 for p in SWIndices]
    symmetricPairsNode = SWIndices[sortperm(distancesFromNode)]
    distancesFromAntiNode = [minimum([sum((map1DTo2D(p, size_BZ) .- (-π, 0.)) .^ 2)^0.5,
                                      sum((map1DTo2D(p, size_BZ) .- (0., -π)) .^ 2)^0.5])
                                     for p in SWIndices
                                    ]
    symmetricPairsAntiNode = SWIndices[sortperm(distancesFromAntiNode)]
    
    desc = "W=$(round(hamiltDetails["W_val"], digits=3))"

    corrNode = @showprogress pmap(pivotPoint -> IterDiagResults(hamiltDetails, maxSize, [pivotPoint], symmetricPairsNode, 
                                                                correlationFuncDict, vneFuncDict, mutInfoFuncDict, 
                                                                bathIntLegs, noSelfCorr, addPerStep),
                                  WorkerPool(1:numProcs), calculatePoints)
    corrAntiNode = @showprogress pmap(pivotPoint -> IterDiagResults(hamiltDetails, maxSize, [pivotPoint], symmetricPairsAntiNode, 
                                                                    correlationFuncDict, vneFuncDict, mutInfoFuncDict, bathIntLegs, 
                                                                    noSelfCorr, addPerStep), 
                                      WorkerPool(1:numProcs), calculatePoints)

    corrResults = mergewith((V1, V2) -> [(isnan(v1) && isnan(v2)) ? NaN : ((isnan(v1) || isnan(v2)) ? filter(!isnan, [v1, v2])[1] : (v1 + v2))
                                         for (v1, v2) in zip(V1, V2)], 
                            corrNode..., corrAntiNode...
                           )
    map!(v -> v ./ 2, values(corrResults))

    corrResults = PropagateIndices(calculatePoints, corrResults, size_BZ, oppositePoints)

    if !isnothing(savePath)
        mkpath(SAVEDIR)
        for (corrName, corrValue) in corrResults
            jldopen(savePath(corrName), "w"; compress = true) do file
                file[corrName] = corrValue
            end
        end
    end

    corrResultsBool = Dict()
    for (name, results) in corrResults
        corrResultsBool[name] = [ifelse(isnan(r), r, abs(r) ≤ 1e-6 ? -1 : 1) for r in results]
    end
    return corrResults, corrResultsBool
end


function AuxiliaryLocalSpecfunc(
        hamiltDetails::Dict,
        numShells::Int64,
        specFuncDictFunc::Function,
        freqValues::Vector{Float64},
        standDevInner::Union{Vector{Float64}, Float64},
        standDevOuter::Union{Vector{Float64}, Float64},
        maxSize::Int64;
        targetHeight::Float64=0.,
        standDevGuess::Float64=0.1,
        heightTolerance::Float64=1e-4,
        bathIntLegs::Int64=2,
        addPerStep::Int64=1,
        maxIter::Int64=20,
        broadFuncType::String="gauss",
    )
    if targetHeight < 0
        targetHeight = 0
    end

    size_BZ = hamiltDetails["size_BZ"]
    cutoffEnergy = hamiltDetails["dispersion"][div(size_BZ - 1, 2) + 2 - numShells]

    # pick out k-states from the southwest quadrant that have positive energies 
    # (hole states can be reconstructed from them (p-h symmetry))
    SWIndicesAll = [p for p in 1:size_BZ^2 if 
                 map1DTo2D(p, size_BZ)[1] ≤ 0 &&
                 map1DTo2D(p, size_BZ)[2] ≤ 0 &&
                 #=map1DTo2D(p, size_BZ)[1] ≤ map1DTo2D(p, size_BZ)[2] &&=#
                 abs(cutoffEnergy) ≥ abs(hamiltDetails["dispersion"][p])
                ]

    SWIndices = filter(p -> maximum(abs.(hamiltDetails["kondoJArray"][p, SWIndicesAll])) > 0, SWIndicesAll)
    if length(SWIndices) == 1
        push!(SWIndices, SWIndicesAll[findfirst(∉(SWIndices), SWIndicesAll)])
    end
    println(length(SWIndices))
    distancesFromNode = [minimum([sum((map1DTo2D(p, size_BZ) .- node) .^ 2)^0.5 
                                  for node in NODAL_POINTS])
                        for p in SWIndices
                       ]
    distancesFromAntiNode = [minimum([sum((map1DTo2D(p, size_BZ) .- antinode) .^ 2)^0.5
                                     for antinode in ANTINODAL_POINTS])
                            for p in SWIndices
                           ]
    @assert distancesFromNode |> length == distancesFromAntiNode |> length

    sortedPointsNode = SWIndices[sortperm(distancesFromNode)]
    sortedPointsAntiNode = SWIndices[sortperm(distancesFromAntiNode)]
    
    onlyCoeffs = ["Sd+", "Sd-"]
    specDictSet = ImpurityExcitationOperators(length(SWIndices))
    standDev = Dict{String, Union{Float64, Vector{Float64}}}(name => ifelse(name ∈ ("Sd+", "Sd-"), standDevInner, standDevOuter) for name in keys(specDictSet))
    broadType = Dict{String, String}(name => ifelse(name ∈ ("Sd+", "Sd-"), "lorentz", "gauss")
                                     for name in keys(specDictSet)
                                    )
    sortedPointsSets = [sortedPointsNode, sortedPointsAntiNode]
    #=sortedPointsSets = [sortedPointsNode]=#
    #=sortedPointsSets = [sortedPointsAntiNode]=#
    #=energySigns = [1]=#
    #=energySigns = [-1]=#
    energySigns = [1, -1]
    argsPermutations = [(p, s) for p in sortedPointsSets for s in energySigns]
    specFuncResultsGathered = repeat(Any[nothing], length(argsPermutations))

    @sync for (threadCounter, (sortedPoints, energySign)) in enumerate(argsPermutations)
        Threads.@spawn begin
            hamiltDetailsModified = deepcopy(hamiltDetails)
            hamiltDetailsModified["imp_corr"] = hamiltDetails["imp_corr"] * energySign
            hamiltDetailsModified["W_val"] = hamiltDetails["W_val"] * energySign
            specFuncResultsGathered[threadCounter] = IterDiagSpecFunc(hamiltDetailsModified, maxSize, deepcopy(sortedPoints),
                                                  specDictSet, bathIntLegs, freqValues, 
                                                  standDev; addPerStep=addPerStep,
                                                  silent=false, broadFuncType=broadType,
                                                  normEveryStep=false, onlyCoeffs=onlyCoeffs,
                                              )
        end
    end
    fixedContrib = Vector{Float64}[]
    specCoeffs = Vector{Tuple{Float64, Float64}}[]
    for specFuncResults in specFuncResultsGathered
        for (key, val) in specFuncResults
            if key ∈ onlyCoeffs
                push!(specCoeffs, val)
            else
                push!(fixedContrib, val)
            end
        end
    end

    insideArea = sum(sum([SpecFunc(coeffs, freqValues, standDevInner; normalise=false) for coeffs in specCoeffs])) * (maximum(freqValues) - minimum(freqValues)) / (length(freqValues) - 1)
    centerSpecFuncArr, localSpecFunc, standDevInner = SpecFuncVariational(specCoeffs, freqValues, targetHeight, 1e-3;
                                 degenTol=1e-10, silent=false, 
                                 broadFuncType="lorentz", 
                                 fixedContrib=fixedContrib,
                                 standDevGuess=standDevGuess,
                                )

    outsideArea = 0
    for specFunc in fixedContrib
        outsideArea += sum(specFunc) * (maximum(freqValues) - minimum(freqValues)) / (length(freqValues) - 1)
    end
    println("Areas = ", (insideArea, outsideArea))
    quasipResidue = insideArea / (insideArea + outsideArea)

    return localSpecFunc, standDevInner, quasipResidue, centerSpecFuncArr
end


function LatticeKspaceDOS(
        hamiltDetails::Dict,
        numShells::Int64,
        specFuncDictFunc::Function,
        freqValues::Vector{Float64},
        standDevInner::Union{Vector{Float64}, Float64},
        standDevOuter::Union{Vector{Float64}, Float64},
        maxSize::Int64;
        onlyAt::Union{Nothing,NTuple{2, Float64}}=nothing,
        bathIntLegs::Int64=2,
        addPerStep::Int64=1,
        maxIter::Int64=20,
        broadFuncType::String="gauss",
        targetHeight::Float64=0.,
        standDevGuess::Float64=0.1,
    )

    size_BZ = hamiltDetails["size_BZ"]
    cutoffEnergy = hamiltDetails["dispersion"][div(size_BZ - 1, 2) + 2 - numShells]

    # pick out k-states from the southwest quadrant that have positive energies 
    # (hole states can be reconstructed from them (p-h symmetry))
    SWIndices = [p for p in 1:size_BZ^2 if 
                 #=map1DTo2D(p, size_BZ)[1] ≤ 0 &&=#
                 #=map1DTo2D(p, size_BZ)[2] ≤ 0 &&=#
                 #=map1DTo2D(p, size_BZ)[1] ≤ -map1DTo2D(p, size_BZ)[2] &&=#
                 abs(cutoffEnergy) ≥ abs(hamiltDetails["dispersion"][p])
                ]
    calculatePoints = [p for p in SWIndices
                       if 0 ≥ map1DTo2D(p, size_BZ)[1] ≥ map1DTo2D(p, size_BZ)[2]
                      ]

    oppositePoints = Dict{Int64, Vector{Int64}}()
    for point in calculatePoints
        reflectDiagonal = map2DTo1D(reverse(map1DTo2D(point, size_BZ))..., size_BZ)
        reflectFS = map2DTo1D((-1 .* reverse(map1DTo2D(point, size_BZ)) .+ [-π, -π])..., size_BZ)
        reflectBoth = map2DTo1D((-1 .* map1DTo2D(point, size_BZ) .+ [-π, -π])..., size_BZ)
        oppositePoints[point] = [reflectDiagonal, reflectFS, reflectBoth]
    end
    rotationMatrix(rotateAngle) = [cos(rotateAngle) -sin(rotateAngle); sin(rotateAngle) cos(rotateAngle)]

    specCoeffsBZone = [NTuple{2, Float64}[] for _ in calculatePoints]
    fixedContrib = [freqValues |> length |> zeros for _ in calculatePoints]

    onlyCoeffs = ["Sd+", "Sd-"]

    function SpectralCoeffsAtKpoint(
            kspacePoint::Int64,
            onlyCoeffs::Vector{String},
        )
        equivalentPoints = [rotationMatrix(rotateAngle) * map1DTo2D(kspacePoint, size_BZ) for rotateAngle in (0, π/2, π, 3π/2)]
        distancesFromPivot = [minimum([sum((map1DTo2D(p, size_BZ) .- pivot) .^ 2)^0.5 for pivot in equivalentPoints])
                            for p in SWIndices
                           ]
        sortedPoints = SWIndices[sortperm(distancesFromPivot)]
        
        specDictSet = specFuncDictFunc(length(SWIndices), (3, 4))
        if hamiltDetails["kondoJArray"][sortedPoints[1], sortedPoints] .|> abs |> maximum == 0 
            delete!(specDictSet, "Sd+")
            delete!(specDictSet, "Sd-")
        end

        standDev = Dict{String, Union{Float64, Vector{Float64}}}(name => ifelse(name ∈ ("Sd+", "Sd-"), standDevInner, standDevOuter) for name in keys(specDictSet))
        broadType = Dict{String, String}(name => ifelse(name ∈ ("Sd+", "Sd-"), "lorentz", "gauss")
                                         for name in keys(specDictSet)
                                        )
        specFuncResultsPoint = IterDiagSpecFunc(hamiltDetails, maxSize, sortedPoints,
                                           specDictSet, bathIntLegs, freqValues, 
                                           standDev; addPerStep=addPerStep,
                                           silent=true, broadFuncType=broadType,
                                           normEveryStep=false, onlyCoeffs=onlyCoeffs,
                                          )
        return specFuncResultsPoint
    end

    if !isnothing(onlyAt)
        pointIndex = map2DTo1D(onlyAt..., size_BZ)
        specFuncResults = SpectralCoeffsAtKpoint(pointIndex, String[])
        specFunc = Normalise(sum(values(specFuncResults)), freqValues, true)
        return specFunc
    end

    specFuncResults = @showprogress pmap(k -> SpectralCoeffsAtKpoint(k, onlyCoeffs), calculatePoints)
    for (index, specFuncResultsPoint) in enumerate(specFuncResults)
        for (name, val) in specFuncResultsPoint 
            if name ∉ onlyCoeffs
                fixedContrib[index] .+= val
            else
                append!(specCoeffsBZone[index], val)
            end
        end
    end

    centerSpecFuncArr, localSpecFunc, standDevFinal = SpecFuncVariational(specCoeffsBZone, freqValues, targetHeight, 1e-3; 
                                   degenTol=1e-10, silent=false, 
                                   broadFuncType="lorentz", fixedContrib=fixedContrib,
                                   standDevGuess=standDevGuess,
                                  )

    specFuncKSpace = [freqValues |> length |> zeros for _ in calculatePoints]

    results = Dict("kspaceDOS" => zeros(size_BZ^2), "quasipRes" => zeros(size_BZ^2))

    results["kspaceDOS"][:, :] .= NaN
    results["quasipRes"][:, :] .= NaN

    for (index, centerSpecFunc) in enumerate(centerSpecFuncArr)
            centerArea = sum(centerSpecFunc) * (maximum(freqValues) - minimum(freqValues)) / (length(freqValues) - 1)
            sideArea = sum(fixedContrib[index]) * (maximum(freqValues) - minimum(freqValues)) / (length(freqValues) - 1)
            results["quasipRes"][calculatePoints[index]] = centerArea / (centerArea + sideArea)
            specFuncKSpace[index] = Normalise(centerSpecFunc .+ fixedContrib[index], freqValues, true)
            results["kspaceDOS"][calculatePoints[index]] = specFuncKSpace[index][freqValues .≥ 0][1]
    end

    results = PropagateIndices(calculatePoints, results, size_BZ, 
                                 oppositePoints)

    return specFuncKSpace, localSpecFunc, standDevFinal, results
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
    scattProbBool = ScattProb(size_BZ, kondoJArray, dispersion)[2]
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
