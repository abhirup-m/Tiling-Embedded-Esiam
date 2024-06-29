include("models.jl")
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


function propagateIndices(index::Int64, size_BZ::Int64)
    # propagate results from lower octant to upper octant
    k_val = map1DTo2D(index, size_BZ)
    newIndices = []
    push!(newIndices, map2DTo1D(reverse(k_val)..., size_BZ))
    for signs in [(-1, 1), (-1, -1), (1, -1)]
        index_prime = map2DTo1D((k_val .* signs)..., size_BZ)
        push!(newIndices, index_prime)
        index_prime = map2DTo1D((reverse(k_val) .* signs)..., size_BZ)
        push!(newIndices, index_prime)
    end
    return newIndices
end


function getIterativeSpectrum(size_BZ::Int64, dispersion::Vector{Float64}, kondoJArray::Array{Float64,3}, W_val::Float64, orbitals::Tuple{String,String}, fractionBZ::Float64)
    retainSize = 500
    energyContours = dispersion[1:trunc(Int, (size_BZ + 1)/2)] |> sort
    energyShells = energyContours[abs.(energyContours) ./ maximum(energyContours) .< fractionBZ]
    println("Working with ", length(energyShells), " shells.")
    southWestQuadrantIndices = [p for p in 1:size_BZ^2 if map1DTo2D(p, size_BZ)[1] < 0 && map1DTo2D(p, size_BZ)[2] < 0]
    activeStates = Int64[]
    initBasis = Dict{Tuple{Int64, Int64}, Vector{Dict{BitVector, Float64}}}()
    numStatesFamily = Int64[]
    activeStatesArr = []
    newStatesArr = []
    @showprogress desc="Hamiltonian family" for energy in energyShells
        onShellStates = filter(x -> abs(dispersion[x] - energy) < TOLERANCE, southWestQuadrantIndices)
        kxvals, _ = map1DTo2D(onShellStates, size_BZ)
        leftToRightSequence = onShellStates[sortperm(kxvals)]

        for point in leftToRightSequence
            push!(newStatesArr, [point])
            activeStates = [activeStates; [point]]
            push!(activeStatesArr, activeStates)
            push!(numStatesFamily, length(activeStates))
            if length(initBasis) == 0
                initBasis = fermions.BasisStates(2 + 2 * length(activeStates); 
                                                 totOccupancy=1 + length(activeStates), localOccupancy=([1,2], 1))
            end
        end
    end
    hamiltonianFamily = fetch.([Threads.@spawn kondoKSpace(activeStates, dispersion, kondoJArray, states -> bathIntForm(W_val, orbitals[2], size_BZ, states); specialIndices=newStates) for (activeStates, newStates) in zip(activeStatesArr, newStatesArr)])
    spectrum = fermions.iterativeDiagonaliser(hamiltonianFamily, initBasis, numStatesFamily, retainSize; tolerance=TOLERANCE)
    return spectrum
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
function sampleKondoIntAndBathInt(sequence::Union{Vector{Int64}, NTuple{TRUNC_DIM, Int64}}, dispersion::Vector{Float64}, kondoJArray::Array{Float64,3}, bathIntArgs::Tuple{Float64,String,Int64}; specialIndices=Int64[])
    k_indices = 1:length(sequence)
    if isempty(specialIndices)
        specialIndices = k_indices
    end

    # create the twice repeated and four times repeated vectors, for nested iterations.
    indices_two_repeated = [pair for pair in Iterators.product(repeat([k_indices], 2)...) if !isempty(intersect(specialIndices, pair))]
    indices_four_repeated = [set for set in Iterators.product(repeat([k_indices], 4)...) if !isempty(intersect(specialIndices, set))]

    # create the dictionaries by iterating over all combinations of the momentum states.
    dispersionDict = Dict{Int64,Float64}(index => dispersion[state] for (index, state) in enumerate(sequence))
    kondoDict = Dict{Tuple{Int64,Int64},Float64}(pair => kondoJArray[sequence[collect(pair)]..., end] for pair in indices_two_repeated) 
    bathIntDict = Dict{Tuple{Int64,Int64,Int64,Int64},Float64}(zip(indices_four_repeated, 
                                                                   fetch.([Threads.@spawn bathIntForm(bathIntArgs..., sequence[collect(fourSet)]) for fourSet in indices_four_repeated])))
    return dispersionDict, kondoDict, Dict(), bathIntDict
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
