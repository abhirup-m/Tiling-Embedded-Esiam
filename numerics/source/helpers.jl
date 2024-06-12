# helper functions for switching back and forth between the 1D flattened representation (1 → N^2) 
# and the 2D representation ((1 → N)×(1 → N))
function map1DTo2D(point::Int64, size_BZ::Int64)
    # Convert overall point to row and column values.
    # These serve as indices of kx and ky
    kx_index = (point - 1) % size_BZ + 1
    ky_index = div((point - 1), size_BZ) + 1

    # construct a 1D array of possible k values, and convert
    # kx_index and ky_index into values
    k_values = range(K_MIN, stop=K_MAX, length=size_BZ)
    return [k_values[kx_index], k_values[ky_index]]
end
function map1DTo2D(point::Vector{Int64}, size_BZ::Int64)
    # same as above, but for multiple points. In this case,
    # two tuples are returned, for kx values and ky values.
    kx_index = (point .- 1) .% size_BZ .+ 1
    ky_index = div.((point .- 1), size_BZ) .+ 1
    k_values = range(K_MIN, stop=K_MAX, length=size_BZ)
    return [k_values[kx_index], k_values[ky_index]]
end

function map2DTo1D(kx_val::Float64, ky_val::Float64, size_BZ::Int64)
    # obtain the indices, given the values of kx and ky
    k_values = range(K_MIN, stop=K_MAX, length=size_BZ)
    kx_index, ky_index = argmin(abs.(k_values .- kx_val)), argmin(abs.(k_values .- ky_val))

    # covert the row and column indices into an overall flattened index
    return kx_index + (ky_index - 1) * size_BZ
end
function map2DTo1D(kx_val::Vector{Float64}, ky_val::Vector{Float64}, size_BZ::Int64)
    # same but for multiple (kx,ky) pairs.
    k_values = range(K_MIN, stop=K_MAX, length=size_BZ)
    points = Int64[]
    for (kx, ky) in zip(kx_val, ky_val)
        kx_index, ky_index = argmin(abs.(k_values .- kx)), argmin(abs.(k_values .- ky))
        push!(points, kx_index + (ky_index - 1) * size_BZ)
    end
    return points
end


# create flattened tight-binding dispersion
function tightBindDisp(kx_vals::Vector{Float64}, ky_vals::Vector{Float64})
    dispersion = -2 * HOP_T .* (cos.(kx_vals) + cos.(ky_vals))
    dispersion[abs.(dispersion) .< TOLERANCE] .= 0
    return dispersion
end
function tightBindDisp(kx_val::Float64, ky_val::Float64)
    dispersion = -2 * HOP_T .* (cos.(kx_val) + cos.(ky_val))
    dispersion[abs.(dispersion) .< TOLERANCE] .= 0
    return dispersion
end


function getDensityOfStates(dispersionFunc, size_BZ)
    kx_vals = repeat(range(K_MIN, stop=K_MAX, length=size_BZ), outer=size_BZ)
    ky_vals = repeat(range(K_MIN, stop=K_MAX, length=size_BZ), inner=size_BZ)

    dispersion = dispersionFunc(kx_vals, ky_vals)
    dispersion_xplus1 = dispersionFunc(circshift(kx_vals, 1), ky_vals)
    dispersion_xminus1 = dispersionFunc(circshift(kx_vals, -1), ky_vals)
    dispersion_yplus1 = dispersionFunc(ky_vals, circshift(ky_vals, size_BZ))
    dispersion_yminus1 = dispersionFunc(ky_vals, circshift(ky_vals, -size_BZ))
    dOfStates =
        4 ./
        sqrt.(
            (dispersion_xplus1 - dispersion_xminus1) .^ 2 +
            (dispersion_yplus1 - dispersion_yminus1) .^ 2
        )

    # van-Hoves might result in Nan, replace them with the largest finite value
    replace!(dOfStates, Inf => maximum(dOfStates[dOfStates.≠Inf]))

    # normalise the DOS according to the convention \int dE \rho(E) = N (where N is the total number of k-states)
    dOfStates /=
        sum([
            dos * abs(dispersion[i%size_BZ+1] - dispersion[i]) for
            (i, dos) in enumerate(dOfStates)
        ]) / size_BZ^2
    return dOfStates, dispersion
end


function getIsoEngCont(dispersion::Vector{Float64}, probeEnergy::Float64)
    # obtain the k-space points that have the specified energy. We count one point per row.

    contourPoints = Int64[]
    for (point, energy) in enumerate(dispersion)
        if abs(energy - probeEnergy) < TOLERANCE
            push!(contourPoints, point)
        end
    end
    return contourPoints
end


function getIsoEngCont(dispersion::Vector{Float64}, probeEnergies::Vector{Float64})
    # obtain the k-space points that have the specified energy. We count one point per row.

    contourPoints = Int64[]
    for (point, energy) in enumerate(dispersion)
        for probeEnergy in probeEnergies
            if abs(energy - probeEnergy) < TOLERANCE
                push!(contourPoints, point)
            end
        end
    end
    return contourPoints
end


function particleHoleTransf(point::Int64, size_BZ::Int64)
    # obtain the particle-hole transformed point k --> (k + π) % π.
    kx_val, ky_val = map1DTo2D(point, size_BZ)
    kx_new = kx_val <= 0.5 * (K_MAX + K_MIN) ? kx_val + 0.5 * (K_MAX - K_MIN) : kx_val - 0.5 * (K_MAX - K_MIN)
    ky_new = ky_val <= 0.5 * (K_MAX + K_MIN) ? ky_val + 0.5 * (K_MAX - K_MIN) : ky_val - 0.5 * (K_MAX - K_MIN)
    return map2DTo1D(kx_new, ky_new, size_BZ)
end

function particleHoleTransf(points::Vector{Int64}, size_BZ::Int64)
    # obtain the particle-hole transformed point k --> (k + π) % π.
    kx_vals, ky_vals = map1DTo2D(points, size_BZ)
    kx_new = [kx <= 0.5 * (K_MAX + K_MIN) ? kx + 0.5 * (K_MAX - K_MIN) : kx - 0.5 * (K_MAX - K_MIN) for kx in kx_vals]
    ky_new = [ky <= 0.5 * (K_MAX + K_MIN) ? ky + 0.5 * (K_MAX - K_MIN) : ky - 0.5 * (K_MAX - K_MIN) for ky in ky_vals]
    return map2DTo1D(kx_new, ky_new, size_BZ)
end


function tolerantSign(quant, boundary)
    if boundary - abs(TOLERANCE) <= quant <= boundary + abs(TOLERANCE)
        return 0
    elseif quant > boundary + abs(TOLERANCE)
        return 1
    else
        return -1
    end
end


function getUpperQuadrantLowerIndices(size_BZ)

    k_indices = collect(1:size_BZ^2)

    # extract the k-states which lie within the energy window of the present iteration and are in the lower bottom quadrant.
    # the values of the other quadrants will be equal to these (C_4 symmetry), so we just calculate one quadrant.
    kx_vals, ky_vals = map1DTo2D(k_indices, size_BZ)
    suitableIndices = k_indices[(kx_vals .>= ky_vals) .& (kx_vals .>= 0) .& (ky_vals .>= 0)]
    return suitableIndices
end


function propagateIndices(suitableIndices::Vector{Int64}, size_BZ::Int64, results::Vector{Float64})
    # propagate results from lower octant to upper octant
    quadrantIndices = copy(suitableIndices)
    for index in suitableIndices
        k_val = map1DTo2D(index, size_BZ)
        index_prime = map2DTo1D(reverse(k_val)..., size_BZ)
        @assert index_prime ∉ suitableIndices || k_val[1] == k_val[2]
        results[index_prime] = results[index]
        push!(quadrantIndices, index_prime)
    end

    # propagate results from lower bottom quadrant to all quadrants.
    for index in quadrantIndices
        k_val = map1DTo2D(index, size_BZ)

        # points in other quadrants are obtained by multiplying kx,ky with 
        # ±1 factors. 
        for signs in [(-1, 1), (-1, -1), (1, -1)]
            index_prime = map2DTo1D((k_val .* signs)..., size_BZ)
            @assert index_prime ∉ suitableIndices || prod(k_val) == 0
            results[index_prime] = results[index]
        end
    end
    return results
end


function getBlockSpectrum(size_BZ::Int64, dispersion::Vector{Float64}, kondoJArray::Array{Float64,3}, W_val::Float64, orbitals::Tuple{String,String})

    # generate basis states for constructing prototype Hamiltonians which
    # will be diagonalised to obtain correlations
    basis = fermions.BasisStates(TRUNC_DIM * 2 + 2)
    suitableIndices = getUpperQuadrantLowerIndices(size_BZ)
    allCombs = collect(combinations(suitableIndices, TRUNC_DIM))
    allSequences = NTuple{TRUNC_DIM, Int64}[]
    for energy in dispersion .|> (x -> round(x, digits=trunc(Int, -log10(TOLERANCE)))) .|> abs |> unique |> (x -> sort(x, rev=true))
        onshellPoints = filter(x -> abs.((dispersion[x]) .- energy) .< TOLERANCE, suitableIndices)
        onshellCombs = filter(x -> !isempty(intersect(x, onshellPoints)), allCombs)
        for comb in onshellCombs
            push!(allSequences, Tuple.(collect(permutations(comb)))...)
        end
    end

    # generate all possible permutations of size TRUNC_DIM from among these k-states.
    # All combinations are needed in order to account for interactions of a particular k-state with all other k-states.
    # All permutations are needed in order to remove basis independence ([k1, k2, k3] & [k1, k3, k2] are not same, because
    # interchanges lead to fermion signs.
    # allSequences = [Tuple(perm) for perm in permutations(suitableIndices, TRUNC_DIM)]
    @assert !isempty(allSequences)

    # get Kondo interaction terms and bath interaction terms involving only the indices
    # appearing in the present sequence.
    dispersionDictSet, kondoDictSet, _, bathIntDictSet = sampleKondoIntAndBathInt(allSequences, dispersion, kondoJArray, (W_val, orbitals[2], size_BZ))

    operatorList, couplingMatrix = fermions.kondoKSpace(dispersionDictSet, kondoDictSet, bathIntDictSet; bathField=0.0, tolerance=TOLERANCE)
    uniqueCouplingSets = Vector{Float64}[]
    uniqueSequences = Vector{NTuple{TRUNC_DIM, Int64}}[]
    for (sequence, couplingSet) in zip(allSequences, couplingMatrix)
        if couplingSet ∉ uniqueCouplingSets
            push!(uniqueCouplingSets, couplingSet)
            push!(uniqueSequences, [sequence])
        else
            location = findall(g -> g == couplingSet, uniqueCouplingSets)
            @assert length(location) == 1
            push!(uniqueSequences[location[1]], sequence)
        end
    end
    matrixSet = fermions.generalOperatorMatrix(basis, operatorList, uniqueCouplingSets)
    eigenSet = fetch.([Threads.@spawn fermions.getSpectrum(matrix) for matrix in matrixSet])

    return basis, uniqueSequences, eigenSet
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
function sampleKondoIntAndBathInt(sequence::NTuple{TRUNC_DIM, Int64}, dispersion::Vector{Float64}, kondoJArray::Array{Float64,3}, bathIntArgs::Tuple{Float64,String,Int64})
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

function sampleKondoIntAndBathInt(sequenceSet::Vector{NTuple{TRUNC_DIM, Int64}}, dispersion::Vector{Float64}, kondoJArray::Array{Float64,3}, bathIntArgs::Tuple{Float64,String,Int64})
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
