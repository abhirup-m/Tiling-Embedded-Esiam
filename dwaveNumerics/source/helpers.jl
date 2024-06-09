include("./constants.jl")

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
    return -2 * HOP_T .* (cos.(kx_vals) + cos.(ky_vals))
end
function tightBindDisp(kx_val::Float64, ky_val::Float64)
    return -2 * HOP_T .* (cos.(kx_val) + cos.(ky_val))
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


function blockSpectrum(size_BZ::Int64, dispersion::Vector{Float64}, kondoJArray::Array{Float64,3}, W_val::Float64, orbitals::Tuple{String,String}; trunc_dim::Int64=2)

    # generate basis states for constructing prototype Hamiltonians which
    # will be diagonalised to obtain correlations
    basis = fermions.BasisStates(trunc_dim * 2 + 2; totOccupancy=[trunc_dim + 1])

    energyContours = dispersion[1:Int(round((size_BZ+1)/2))]
    spectrumSet = []
    sequenceSets = []

    # loop over energy scales and calculate correlation values at each stage.
    @showprogress for energy in energyContours
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
        uniqueHamiltonianSequences = Dict{Vector{Float64},Vector{Vector{Int64}}}()
        for (sequence, couplingSet) in zip(allSequences, couplingMatrix)
            if couplingSet ∉ keys(uniqueHamiltonianSequences)
                uniqueHamiltonianSequences[couplingSet] = Vector{Int64}[]
            end
            push!(uniqueHamiltonianSequences[couplingSet], sequence)
        end
        matrixSet = fermions.generalOperatorMatrix(basis, operatorList, collect(keys(uniqueHamiltonianSequences)))
        eigenSet = fetch.([Threads.@spawn fermions.getSpectrum(matrix) for matrix in matrixSet])
        push!(spectrumSet, eigenInfo)
        push!(sequenceSets, uniqueHamiltonianSequences)
    end
    return energyContour, spectrumSet, sequenceSets
end
