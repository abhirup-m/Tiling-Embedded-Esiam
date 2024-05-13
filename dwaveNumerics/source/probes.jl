##### Functions for calculating various probes          #####
##### (correlation functions, Greens functions, etc)    #####

using LinearAlgebra
using Combinatorics
include("../../../fermionise/source/fermionise.jl")
include("../../../fermionise/source/correlations.jl")
include("../../../fermionise/source/models.jl")

"""
Function to calculate the average Kondo scattering probability Γ(k) = (1/N^*)∑_q J(k,q)^2
at the RG fixed point, where N^* is the number of points within the fixed point window.
"""
function scattProb(kondoJArray::Array{Float64,3}, stepIndex::Int64, size_BZ::Int64, dispersion::Vector{Float64}, fixedpointEnergy::Float64)

    # allocate zero arrays to store Γ at fixed point and for the bare Hamiltonian.
    results = zeros(size_BZ^2)
    results_bare = zeros(size_BZ^2)

    # get all allowed energies starting from the fixed point energy
    # up to the Fermi energy. Only the k-states lying in this range
    # will be considering for the summation over q.
    E_cloud = dispersion[-fixedpointEnergy.<=dispersion.<=fixedpointEnergy]

    # set of points q that lie within the allowed window. This is
    # obtained by calculating the isoenergetic points that lie on 
    # each energy level E within the set of levels E_cloud = [E1, E2, ...].
    point2_arr = unique(getIsoEngCont(dispersion, E_cloud))

    # loop over all points k for which we want to calculate Γ(k).
    Threads.@threads for point1 in 1:size_BZ^2

        # check if the point is one of the four corners of the 
        # Brillouin zone. If it is, then set that to zero, because
        # it is not affected by the RG and is therefore not of interest.
        kx, ky = map1DTo2D(point1, size_BZ)
        if (abs(kx) ≈ K_MAX && abs(ky) ≈ K_MAX) || (kx ≈ 0 && ky ≈ 0)
            continue
        end

        # calculate the average over q, both for the fixed point and the bare Hamiltonian.
        results[point1] = sum(kondoJArray[point1, point2_arr, stepIndex] .^ 2) / length(point2_arr)
        results_bare[point1] = sum(kondoJArray[point1, point2_arr, 1] .^ 2) / length(point2_arr)
    end

    # get a boolean representation of results for visualisation, using the mapping
    # results_bool = -1 if results/results_bare < TOLERANCE and +1 otherwise.
    results_bool = tolerantSign.(abs.(results), abs.(results_bare) .* RG_RELEVANCE_TOL)

    # set the -1 to NaN, in order to make them stand out on the plots.
    results[results_bool.<0] .= NaN

    return results ./ results_bare, results_bare, results_bool
end


"""
Return the map of Kondo couplings M_k(q) = J^2_{k,q} given the state k.
Useful for visualising how a single state k interacts with all other states q.
"""
function kondoCoupMap(k_vals::Tuple{Float64,Float64}, size_BZ::Int64, kondoJArrayFull::Array{Float64,3})
    kspacePoint = map2DTo1D(k_vals..., size_BZ)
    results = kondoJArrayFull[kspacePoint, :, end] .^ 2
    results_bare = kondoJArrayFull[kspacePoint, :, 1] .^ 2
    results_bool = tolerantSign.(abs.(results), RG_RELEVANCE_TOL / size_BZ)
    results[results_bool.<0] .= NaN
    return results, results_bare, results_bool
end


"""
Create two dictionaries kondoDict and bathIntDict for use in another function.
The dictionary kondoDict has tuples (k1, k2) as keys and the corresponding 
Kondo coupling J_{k1, k2} as the values, while bathIntDict has 4-tuples (k1,k2,k3,k4)
as keys and the corresponding bath W-interaction W_{k1,k2,k3,k4} as the values.
Useful if I want to look at the interactions only between a certain set of k-states.
Used for constructing lattice eSIAM Hamiltonians for samples of 2/3/4 k-states, in
order to diagonalise and obtain correlations.
"""
function sampleKondoIntAndBathInt(sequence::Vector{Int64}, dispersion::Vector{Float64}, kondoJArray::Array{Float64,3}, bathIntArgs::Tuple{Float64,String,Int64})
    k_indices = 1:length(sequence)

    # create the twice repeated and four times repeated vectors, for nested iterations.
    indices_two_repeated = ntuple(x -> k_indices, 2)
    indices_four_repeated = ntuple(x -> k_indices, 4)

    # create the dictionaries by iterating over all combinations of the momentum states.
    dispersionDict = Dict{Int64,Float64}(index => dispersion[state] for (index, state) in enumerate(sequence))
    kondoDict = Dict{Tuple{Int64,Int64},Float64}(pair => kondoJArray[sequence[collect(pair)]..., end] for pair in Iterators.product(indices_two_repeated...))
    kondoDictBare = Dict{Tuple{Int64,Int64},Float64}(pair => kondoJArray[sequence[collect(pair)]..., 1] for pair in Iterators.product(indices_two_repeated...))
    bathIntDict = Dict{Tuple{Int64,Int64,Int64,Int64},Float64}(fourSet => bathIntForm(bathIntArgs..., sequence[collect(fourSet)]) for fourSet in Iterators.product(indices_four_repeated...))
    return dispersionDict, kondoDict, kondoDictBare, bathIntDict
end

function sampleKondoIntAndBathInt(sequenceSet::Vector{Vector{Int64}}, dispersion::Vector{Float64}, kondoJArray::Array{Float64,3}, bathIntArgs::Tuple{Float64,String,Int64})
    dispersionDictSet = Dict{Int64,Float64}[]
    kondoDictSet = Dict{Tuple{Int64,Int64},Float64}[]
    kondoDictBareSet = Dict{Tuple{Int64,Int64},Float64}[]
    bathIntDictSet = Dict{Tuple{Int64,Int64,Int64,Int64},Float64}[]
    for sequence in sequenceSet
        dispersionDict, kondoDict, kondoDictBare, bathIntDict = sampleKondoIntAndBathInt(sequence, dispersion, kondoJArray, bathIntArgs)
        push!(dispersionDictSet, dispersionDict)
        push!(kondoDictSet, kondoDict)
        push!(kondoDictBareSet, kondoDictBare)
        push!(bathIntDictSet, bathIntDict)
    end
    return dispersionDictSet, kondoDictSet, kondoDictBareSet, bathIntDictSet
end


function correlationMap(size_BZ::Int64, dispersion::Vector{Float64}, kondoJArray::Array{Float64,3}, W_val::Float64, orbitals::Tuple{String,String}, correlationDefinition; trunc_dim::Int64=3)

    # get all possible indices of momentum states within the Brillouin zone
    k_indices = collect(1:size_BZ^2)

    # initialise zero array for storing correlations
    results = zeros(size_BZ^2)
    results_bare = zeros(size_BZ^2)

    # initialise zero array to count the number of times a particular k-state
    # appears in the computation. Needed to finally average over all combinations.
    contributorCounter = zeros(size_BZ^2)

    # generate basis states for constructing prototype Hamiltonians which
    # will be diagonalised to obtain correlations
    basis = BasisStates(trunc_dim * 2 + 2)

    # define correlation operators for all k-states within Brillouin zone.
    correlationOperatorList = [correlationDefinition(i) for i in 1:trunc_dim]

    # loop over energy scales and calculate correlation values at each stage.
    for energy in [0]

        # extract the k-states which lie within the energy window of the present iteration and are in the lower bottom quadrant.
        # the values of the other quadrants will be equal to these (C_4 symmetry), so we just calculate one quadrant.
        suitableIndices = [index for index in k_indices if abs(dispersion[index] - energy) < TOLERANCE && map1DTo2D(index, size_BZ)[1] >= 0 && map1DTo2D(index, size_BZ)[2] <= 0]

        # generate all possible configurations of k-states from among these k-states. These include combinations as well as permutations.
        # All combinations are needed in order to account for interactions of a particular k-state with all other k-states.
        # All permutations are needed in order to remove basis independence ([k1, k2, k3] & [k1, k3, k2] are not same, because
        # interchanges lead to fermion signs.
        allSequences = [perm for chosenIndices in collect(combinations(suitableIndices, trunc_dim)) for perm in permutations(chosenIndices)]
        @assert !isempty(allSequences)

        # get Kondo interaction terms and bath interaction terms involving only the indices
        # appearing in the present sequence.
        dispersionDictSet, kondoDictSet, _, bathIntDictSet = sampleKondoIntAndBathInt(allSequences, dispersion, kondoJArray, (W_val, orbitals[2], size_BZ))
        operatorList, couplingMatrix = kondoKSpace(dispersionDictSet, kondoDictSet, bathIntDictSet)
        @time matrixSet = generalOperatorMatrix(basis, operatorList, couplingMatrix)
        @time eigenSet = fetch.([Threads.@spawn getSpectrum(matrix) for matrix in matrixSet])
        @time correlationResults = fetch.([Threads.@spawn gstateCorrelation(basis, eigenInfo..., correlationOperatorList) 
                                           for (sequence, matrix, eigenInfo) in zip(allSequences, matrixSet, eigenSet)])
        # calculate the correlation for all such configurations.
        @time for (sequence, correlationResult) in zip(allSequences, correlationResults)

            # calculate the correlation for all points in the present sequence
            results[sequence] .+= correlationResult
            contributorCounter[sequence] .+= 1
        end

        # average over all sequences
        results[suitableIndices] ./= contributorCounter[suitableIndices]
    end

    # propagate results from lower bottom quadrant to all quadrants.
    for index in [index for index in k_indices if map1DTo2D(index, size_BZ)[1] >= 0 && map1DTo2D(index, size_BZ)[2] <= 0]
        k_val = map1DTo2D(index, size_BZ)

        # points in other quadrants are obtained by multiplying kx,ky with 
        # ±1 factors. 
        for signs in [(-1, 1), (-1, -1), (1, -1)]
            index_prime = map2DTo1D((k_val .* signs)..., size_BZ)
            results[index_prime] = results[index]
        end
    end

    # get a boolean representation of results for visualisation, using the mapping
    # results_bool = -1 if results/results_bare < TOLERANCE and +1 otherwise.
    results_bool = Dict(k => tolerantSign.(abs(results[k]), abs.(results_bare[k]) .* RG_RELEVANCE_TOL) for k in keys(results))
    return results, results_bare, results_bool
end


"""
Maps the given probename string (such as "kondoCoupNodeMap") to its appropriate function 
which can calculate and return the value of that probe.
"""
function mapProbeNameToProbe(probeName::String, size_BZ::Int64, kondoJArrayFull::Array{Float64,3}, W_val::Float64, dispersion::Vector{Float64}, orbitals::Tuple{String,String}, fixedpointEnergy::Float64)
    if probeName == "scattProb"
        results, results_bare, results_bool = scattProb(kondoJArrayFull, size(kondoJArrayFull)[3], size_BZ, dispersion, fixedpointEnergy)
    elseif probeName == "kondoCoupNodeMap"
        results, results_bare, results_bool = kondoCoupMap(node, size_BZ, kondoJArrayFull)
    elseif probeName == "kondoCoupAntinodeMap"
        results, results_bare, results_bool = kondoCoupMap(antinode, size_BZ, kondoJArrayFull)
    elseif probeName == "kondoCoupOffNodeMap"
        results, results_bare, results_bool = kondoCoupMap(offnode, size_BZ, kondoJArrayFull)
    elseif probeName == "kondoCoupOffAntinodeMap"
        results, results_bare, results_bool = kondoCoupMap(offantinode, size_BZ, kondoJArrayFull)
    elseif probeName == "spinFlipCorrMap"
        results, results_bool = spinFlipCorrMapCoarse(size_BZ, dispersion, kondoJArrayFull, W_val, orbitals, Dict(("+-+-", [2, 1, 2 * i + 1, 2 * i + 2]) => 1.0, ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1]) => 1.0))
    end
    return results, results_bool
end
