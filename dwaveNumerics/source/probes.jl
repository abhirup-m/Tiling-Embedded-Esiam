##### Functions for calculating various probes          #####
##### (correlation functions, Greens functions, etc)    #####

using LinearAlgebra
using Combinatorics
include("../../../../juliaProjects/fermions/src/base.jl")
include("../../../../juliaProjects/fermions/src/eigen.jl")
include("../../../../juliaProjects/fermions/src/correlations.jl")
include("../../../../juliaProjects/fermions/src/models.jl")
using ProgressMeter

"""
Function to calculate the total Kondo scattering probability Γ(k) = ∑_q J(k,q)^2
at the RG fixed point.
"""
function scattProb(kondoJArray::Array{Float64,3}, size_BZ::Int64, dispersion::Vector{Float64})

    # allocate zero arrays to store Γ at fixed point and for the bare Hamiltonian.
    results = zeros(size_BZ^2)

    # loop over all points k for which we want to calculate Γ(k).
    Threads.@threads for point in 1:size_BZ^2

        # check if the point is one of the four corners or the 
        # center. If it is, then don't bother. These points are
        # not affected by the RG and therefore not of interest.
        if point ∉ [1, size_BZ, size_BZ^2 - size_BZ + 1, size_BZ^2, trunc(Int, 0.5 * (size_BZ^2 + 1))]
            targetStatesForPoint = collect(1:size_BZ^2)[abs.(dispersion).<=abs(dispersion[point])]

            # calculate the sum over q
            results[point] = sum(kondoJArray[point, targetStatesForPoint, end] .^ 2)
        end

    end

    # get a boolean representation of results for visualisation, using the mapping
    results_bool = ifelse.(abs.(results) .> TOLERANCE, 1, 0)

    return results, results_bool
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

    for results in fetch.([Threads.@spawn sampleKondoIntAndBathInt(sequence, dispersion, kondoJArray, bathIntArgs) for sequence in sequenceSet])
        dispersionDict, kondoDict, kondoDictBare, bathIntDict = results
        push!(dispersionDictSet, dispersionDict)
        push!(kondoDictSet, kondoDict)
        push!(kondoDictBareSet, kondoDictBare)
        push!(bathIntDictSet, bathIntDict)
    end
    return dispersionDictSet, kondoDictSet, kondoDictBareSet, bathIntDictSet
end


function correlationMap(size_BZ::Int64, dispersion::Vector{Float64}, kondoJArray::Array{Float64,3}, W_val::Float64, orbitals::Tuple{String,String}, correlationDefinition; trunc_dim::Int64=4)

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

    # initialise zero array to count the number of times a particular k-state
    # appears in the computation. Needed to finally average over all combinations.
    contributorCounter = zeros(size_BZ^2)

    # generate basis states for constructing prototype Hamiltonians which
    # will be diagonalised to obtain correlations
    basis = BasisStates(trunc_dim * 2 + 2; totOccupancy=[trunc_dim + 1])

    # define correlation operators for all k-states within Brillouin zone.
    correlationOperatorList = [correlationDefinition(i) for i in 1:trunc_dim]

    # loop over energy scales and calculate correlation values at each stage.
    @showprogress Threads.@threads for energy in dispersion[abs.(dispersion).<maximum(dispersion)/8] .|> (x -> round(x, digits=trunc(Int, -log10(TOLERANCE)))) .|> abs |> unique |> (x -> sort(x, rev=true))
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
        @time dispersionDictSet, kondoDictSet, _, bathIntDictSet = sampleKondoIntAndBathInt(allSequences, dispersion, kondoJArray, (0.0, orbitals[2], size_BZ))
        @time operatorList, couplingMatrix = kondoKSpace(dispersionDictSet, kondoDictSet, bathIntDictSet; tolerance=TOLERANCE)
        uniqueHamiltonians = Dict{Vector{Float64},Vector{Vector{Int64}}}()
        @time for (sequence, couplingSet) in zip(allSequences, couplingMatrix)
            if couplingSet ∉ keys(uniqueHamiltonians)
                uniqueHamiltonians[couplingSet] = Vector{Int64}[]
            end
            push!(uniqueHamiltonians[couplingSet], sequence)
        end
        @time matrixSet = generalOperatorMatrix(basis, operatorList, collect(keys(uniqueHamiltonians)))
        @time eigenSet = fetch.([Threads.@spawn getSpectrum(matrix) for matrix in matrixSet])
        @time correlationResults = fetch.([Threads.@spawn gstateCorrelation(basis, eigenvals, eigenstates, correlationOperatorList) for (sequence, (eigenvals, eigenstates)) in zip(allSequences, eigenSet)])
        println("-------------------")

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

    results_bool = [r <= 1e-3 ? -1 : 1 for r in results]
    return results, results_bool
end


function tiledCorrelationMap(size_BZ::Int64, dispersion::Vector{Float64}, kondoJArray::Array{Float64,3}, W_val::Float64, orbitals::Tuple{String, String})
    results = zeros(size_BZ^2, size_BZ^2)
    correlationDefinition = i -> Dict(("+-+-", [2, 1, 2 * i + 1, 2 * i + 2]) => 1.0, ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1]) => 1.0)
    correlationmap, _ = correlationMap(size_BZ, dispersion, kondoJArray, W_val, orbitals, correlationDefinition)
    results = 0.5 .* sqrt.(correlationmap * correlationmap')
    results[abs.(results) .< TOLERANCE ^ 0.5] .= TOLERANCE ^ 0.5
    results_bool = [r <= 1e-3 ? -1 : 1 for r in results]
    return log10.(results), results_bool
end
