##### Functions for calculating various probes          #####
##### (correlation functions, Greens functions, etc)    #####

using LinearAlgebra
using Combinatorics
using fermions

"""
Function to calculate the average Kondo scattering probability Γ(k) = (1/N^*)∑_q J(k,q)^2
at the RG fixed point, where N^* is the number of points within the fixed point window.
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
        if point ∈ [1, size_BZ, size_BZ^2 - size_BZ + 1, size_BZ^2, trunc(Int, 0.5 * (size_BZ^2 + 1))]
            continue
        end

        targetStatesForPoint = collect(1:size_BZ^2)[abs.(dispersion).<=abs(dispersion[point])]

        # calculate the average over q, both for the fixed point and the bare Hamiltonian.
        results[point] = sum(kondoJArray[point, targetStatesForPoint, end] .^ 2) / length(targetStatesForPoint)
        results_bare[point] = sum(kondoJArray[point, targetStatesForPoint, 1] .^ 2) / length(targetStatesForPoint)
    end

    # get a boolean representation of results for visualisation, using the mapping
    # results_bool = -1 if results/results_bare < TOLERANCE and +1 otherwise.
    results_bool = [abs(r / r_b) < RG_RELEVANCE_TOL ? -1 : 1 for (r, r_b) in zip(results, results_bare)]

    # set the -1 to NaN, in order to make them stand out on the plots.
    # results[results_bool.==-1] .= NaN
    results_scaled = [r_b == 0 ? r : r / r_b for (r, r_b) in zip(results, results_bare)]

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


function correlationMap(size_BZ::Int64, dispersion::Vector{Float64}, kondoJArray::Array{Float64,3}, W_val::Float64, orbitals::Tuple{String,String}, correlationDefinition; trunc_dim::Int64=2)

    _, resultsScattProbBool = scattProb(kondoJArray, size_BZ, dispersion)

    # get all possible indices of momentum states within the Brillouin zone
    k_indices = collect(1:size_BZ^2)

    for index in k_indices
        if resultsScattProbBool[index] == -1
            kondoJArray[index, :, end] .= 0
            kondoJArray[:, index, end] .= 0
        end
    end

    # initialise zero array for storing correlations
    results = zeros(size_BZ^2)
    results_bare = zeros(size_BZ^2)

    # initialise zero array to count the number of times a particular k-state
    # appears in the computation. Needed to finally average over all combinations.
    contributorCounter = zeros(size_BZ^2)

    # generate basis states for constructing prototype Hamiltonians which
    # will be diagonalised to obtain correlations
    basis = fermions.BasisStates(trunc_dim * 2 + 2; totOccupancy=[trunc_dim + 1])

    # define correlation operators for all k-states within Brillouin zone.
    correlationOperatorList = [correlationDefinition(i) for i in 1:trunc_dim]

    # loop over energy scales and calculate correlation values at each stage.
    # Threads.@threads 
    for energy in dispersion[abs.(dispersion).<maximum(dispersion)/2] .|> (x -> round(x, digits=trunc(Int, -log10(TOLERANCE)))) .|> abs |> unique |> (x -> sort(x, rev=true))
        # extract the k-states which lie within the energy window of the present iteration and are in the lower bottom quadrant.
        # the values of the other quadrants will be equal to these (C_4 symmetry), so we just calculate one quadrant.
        suitableIndices = [index for index in k_indices if abs(dispersion[index]) <= (energy + TOLERANCE) && map1DTo2D(index, size_BZ)[1] >= 0 && map1DTo2D(index, size_BZ)[2] <= 0]

        # generate all possible configurations of k-states from among these k-states. These include combinations as well as permutations.
        # All combinations are needed in order to account for interactions of a particular k-state with all other k-states.
        # All permutations are needed in order to remove basis independence ([k1, k2, k3] & [k1, k3, k2] are not same, because
        # interchanges lead to fermion signs.
        onShellStates = [index for index in suitableIndices if abs(abs(dispersion[index]) - energy) < TOLERANCE]
        allCombinations = [comb for comb in combinations(suitableIndices, trunc_dim) if !isempty(intersect(onShellStates, comb))]
        allSequences = [perm for chosenIndices in allCombinations for perm in permutations(chosenIndices)]
        @assert !isempty(allSequences)

        # get Kondo interaction terms and bath interaction terms involving only the indices
        # appearing in the present sequence.
        dispersionDictSet, kondoDictSet, _, bathIntDictSet = sampleKondoIntAndBathInt(allSequences, dispersion, kondoJArray, (0.0, orbitals[2], size_BZ))
        operatorList, couplingMatrix = fermions.kondoKSpace(dispersionDictSet, kondoDictSet, bathIntDictSet; tolerance=TOLERANCE)
        uniqueHamiltonians = Dict{Vector{Float64}, Vector{Vector{Int64}}}()
        for (sequence, couplingSet) in zip(allSequences, couplingMatrix)
            if couplingSet ∉ keys(uniqueHamiltonians)
                uniqueHamiltonians[couplingSet] = Vector{Int64}[]
            end
            push!(uniqueHamiltonians[couplingSet], sequence)
        end
        matrixSet = fermions.generalOperatorMatrix(basis, operatorList, collect(keys(uniqueHamiltonians)))
        eigenSet = fetch.([Threads.@spawn fermions.getSpectrum(matrix) for matrix in matrixSet])
        correlationResults = fetch.([Threads.@spawn fermions.gstateCorrelation(basis, eigenvals, eigenstates, correlationOperatorList) for (sequence, (eigenvals, eigenstates)) in zip(allSequences, eigenSet)])

        # calculate the correlation for all such configurations.
        for (sequences, correlationResult) in zip(values(uniqueHamiltonians), correlationResults)

            for sequence in sequences
                # calculate the correlation for all points in the present sequence
                for (i, index) in enumerate(sequence)
                    if index ∉ onShellStates
                        continue
                    end
                    results[index] += correlationResult[i]
                    contributorCounter[index] += 1
                end
            end
        end

        # average over all sequences
        results[onShellStates] ./= contributorCounter[onShellStates]

    end

    # propagate results from lower bottom quadrant to all quadrants.
    Threads.@threads for index in [index for index in k_indices if map1DTo2D(index, size_BZ)[1] >= 0 && map1DTo2D(index, size_BZ)[2] <= 0]
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
    # println(results[13])
    # results = [r <= 1e-3 ? NaN : r for r in results]
    # println(results[13])
    # results[results .< 0.1] .= NaN
    results_bool = [r <= 1e-3 ? -1 : 1 for (r, r_b) in zip(results, results_bare)]
    return results, results_bool
end


"""
Maps the given probename string (such as "kondoCoupNodeMap") to its appropriate function 
which can calculate and return the value of that probe.
"""
function mapProbeNameToProbe(probeName::String, size_BZ::Int64, kondoJArrayFull::Array{Float64,3}, W_val::Float64, dispersion::Vector{Float64}, orbitals::Tuple{String,String})
    if probeName == "scattProb"
        results, results_bool = scattProb(kondoJArrayFull, size_BZ, dispersion)
    elseif probeName == "kondoCoupNodeMap"
        results, results_bare, results_bool = kondoCoupMap(node, size_BZ, kondoJArrayFull)
    elseif probeName == "kondoCoupAntinodeMap"
        results, results_bare, results_bool = kondoCoupMap(antinode, size_BZ, kondoJArrayFull)
    elseif probeName == "kondoCoupOffNodeMap"
        results, results_bare, results_bool = kondoCoupMap(offnode, size_BZ, kondoJArrayFull)
    elseif probeName == "kondoCoupOffAntinodeMap"
        results, results_bare, results_bool = kondoCoupMap(offantinode, size_BZ, kondoJArrayFull)
    elseif probeName == "spinFlipCorrMap"
        @time results, results_bool = correlationMap(size_BZ, dispersion, kondoJArrayFull, W_val, orbitals, i -> Dict(("+-+-", [2, 1, 2 * i + 1, 2 * i + 2]) => 1.0, ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1]) => 1.0))
    end
    return results, results_bool
end
