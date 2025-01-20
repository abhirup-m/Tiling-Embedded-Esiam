# helper functions for switching back and forth between the 1D flattened representation (1 → N^2) 
# and the 2D representation ((1 → N)×(1 → N))
@everywhere function map1DTo2D(point::Int64, size_BZ::Int64)
    # Convert overall point to row and column values.
    # These serve as indices of kx and ky
    kx_index = (point - 1) % size_BZ + 1
    ky_index = div((point - 1), size_BZ) + 1

    # construct a 1D array of possible k values, and convert
    # kx_index and ky_index into values
    k_values = range(K_MIN, stop=K_MAX, length=size_BZ)
    return [k_values[kx_index], k_values[ky_index]]
end
@everywhere function map1DTo2D(point::Vector{Int64}, size_BZ::Int64)
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


# propagate results from lower octant to upper octant
function PropagateIndices(
        innerPoints::Vector{Int64}, 
        corrResults::Dict{String, Vector{Float64}}, 
        size_BZ::Int64, 
        oppositePoints::Dict{Int64, Vector{Int64}}
    )
    for point in innerPoints
        for (name, correlation) in corrResults
            corrResults[name][oppositePoints[point]] .= correlation[point]
        end
    end
    
    for pivot in innerPoints
        newPoints = [map2DTo1D((map1DTo2D(point, size_BZ) .* signs)..., size_BZ) 
                     for point in [[pivot]; oppositePoints[pivot]] 
                     for signs in [(-1, 1), (-1, -1), (1, -1)]
                    ]
        for (name, correlation) in corrResults
            corrResults[name][newPoints] .= correlation[pivot]
        end
    end
    return corrResults
end


# propagate results from lower octant to upper octant
function PropagateIndices(
        innerPoints::Vector{Int64}, 
        size_BZ::Int64, 
        oppositePoints::Dict{Int64, Vector{Int64}}
    )
    propagators = Dict{Int64, Vector{Int64}}(p => Int64[] for p in innerPoints)
    for pivot in innerPoints
        append!(propagators[pivot], oppositePoints[pivot])
    end
    
    for pivot in innerPoints
        newPoints = [map2DTo1D((map1DTo2D(point, size_BZ) .* signs)..., size_BZ) 
                     for point in [[pivot]; oppositePoints[pivot]] 
                     for signs in [(-1, 1), (-1, -1), (1, -1)]
                    ]
        append!(propagators[pivot], newPoints)
    end
    return propagators
end


@everywhere function bathIntForm(
    bathIntStr::Float64,
    orbital::String,
    size_BZ::Int64,
    points,
)
    # bath interaction does not renormalise, so we don't need to make it into a matrix. A function
    # is enough to invoke the W(k1,k2,k3,k4) value whenever we need it. To obtain it, we call the p-wave
    # function for each momentum k_i, then multiply them to get W_1234 = W × p(k1) * p(k2) * p(k3) * p(k4)
    k2_vals = map1DTo2D(points[2], size_BZ)
    k3_vals = map1DTo2D(points[3], size_BZ)
    k4_vals = map1DTo2D(points[4], size_BZ)
    k1_vals = map1DTo2D(points[1], size_BZ)
    if orbital == "d"
        bathInt = bathIntStr
        return 0.5 .* bathIntStr .* (
            cos.(k1_vals[1] .- k2_vals[1] .+ k3_vals[1] .- k4_vals[1]) .-
            cos.(k1_vals[2] .- k2_vals[2] .+ k3_vals[2] .- k4_vals[2])
        )
    elseif orbital == "p"
        return 0.5 .* bathIntStr .* (
            cos.(k1_vals[1] .- k2_vals[1] .+ k3_vals[1] .- k4_vals[1]) .+
            cos.(k1_vals[2] .- k2_vals[2] .+ k3_vals[2] .- k4_vals[2])
        )
    elseif orbital == "poff"
        bathInt = bathIntStr
        for (kx, ky) in [k1_vals, k2_vals, k3_vals, k4_vals]
            bathInt = bathInt .* (cos.(kx) + cos.(ky))
        end
        return bathInt
    else
        bathInt = bathIntStr
        for (kx, ky) in [k1_vals, k2_vals, k3_vals, k4_vals]
            bathInt = bathInt .* (cos.(kx) - cos.(ky))
        end
        return bathInt
    end
end


function impSpecFunc(
        numBathPoints::Int64,
    )
    siamSpecDict = Dict{String, Vector{Tuple{String,Vector{Int64}, Float64}}}("create" => [], "destroy" => [])
    append!(siamSpecDict["create"], [("+", [1], 1.), ("+", [2], 1.)])
    append!(siamSpecDict["destroy"], [("-", [1], 1.), ("-", [2], 1.)])
    kondoSpecDict = Dict{String, Vector{Tuple{String,Vector{Int64}, Float64}}}("create" => [], "destroy" => [])
    for index in 1:numBathPoints
        append!(kondoSpecDict["create"], [("+-+", [2, 1, 2 * index + 1], 1.), ("+-+", [1, 2, 2 * index + 2], 1.),])
        append!(kondoSpecDict["destroy"], [("+--", [1, 2, 2 * index + 1], 1.), ("+--", [2, 1, 2 * index + 2], 1.),])
    end
    return siamSpecDict, kondoSpecDict
end


function kspaceSpecFunc(
        numBathPoints::Int64,
        kpointIndices::NTuple{2, Int64},
    )
    specDictSet = Dict{String, Vector{Tuple{String,Vector{Int64}, Float64}}}()

    # composite excitation where a bath k-state is created (|n_{k↑} = 0⟩ → |n_{k↑} = 1⟩) 
    # along with the destruction of the impurity down state (|n_{d↓} = 0⟩ → |n_{d↓} = 1⟩).
    # Overall spin is conserved, in line with the local Fermi liquid excitations
    specDictSet["cd_down"] = [("+-", [kpointIndices[1], 2], 1.)]
    specDictSet["cd_up"] = [("+-", [kpointIndices[2], 1], 1.)]
    specDictSet["cdagd_down"] = [("++", [kpointIndices[1], 2], 1.)]
    specDictSet["cdagd_up"] = [("++", [kpointIndices[2], 1], 1.)]

    # composite excitation where a bath k-state is created (|n_{k↑} = 0⟩ → |n_{k↑} = 1⟩) 
    # along with a down-flip of the impurity spin (|↑⟩ → |↓⟩).
    # Overall spin is conserved, in line with the local Fermi liquid excitations
    specDictSet["Sd-"] = [("++-", [kpointIndices[1], 2, 1], 1.)]
    specDictSet["Sd+"] = [("++-", [kpointIndices[2], 1, 2], 1.)]
    return specDictSet
end
