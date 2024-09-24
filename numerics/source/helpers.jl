using fermions

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


# propagate results from lower octant to upper octant
function propagateIndices(
        innerPoints::Vector{Int64}, 
        corrResults::Dict{String, Vector{Float64}}, 
        size_BZ::Int64, 
        oppositePoints::Dict{Int64, Vector{Int64}}
    )
    
    for pivot in innerPoints
        newIndices = Int64[oppositePoints[pivot]...]
        for index in [[pivot]; oppositePoints[pivot]]
            k_val = map1DTo2D(index, size_BZ)
            append!(newIndices, [map2DTo1D((k_val .* signs)..., size_BZ) for signs in [(-1, 1), (-1, -1), (1, -1)]])
        end
        for (name, correlation) in corrResults
            corrResults[name][newIndices] .= correlation[pivot]
        end
    end
    return corrResults
end


function PhaseDiagram(J_vals_arr::Vector{Float64}, W_val_arr::Vector{Float64}, numPoints::Int64, phaseMaps::Dict{String, Int64})
    function PhaseStrip(J_val::Float64, W_val_arr::Vector{Float64}, phaseMaps::Dict{String, Int64})
        phaseFlags = 0 .* collect(W_val_arr)
        trackPoints = Dict(
                           "N" => map2DTo1D(π/2, π/2, size_BZ),
                           "AN" => map2DTo1D(π/1, 0.0, size_BZ),
                           "M" => map2DTo1D(0.75 * π, 0.25 * π, size_BZ),
                          )

        gapTrackers = Dict("N" => NaN, "AN" => NaN, "M" => NaN)
        for (i, W_val) in enumerate(W_val_arr)
            kondoJArray, dispersion = momentumSpaceRG(size_BZ, omega_by_t, J_val, W_val, orbitals)
            averageKondoScale = sum(abs.(kondoJArray[:, :, 1])) / length(kondoJArray[:, :, 1])
            @assert averageKondoScale > RG_RELEVANCE_TOL
            kondoJArray[:, :, end] .= ifelse.(abs.(kondoJArray[:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, kondoJArray[:, :, end], 0)
            scattProbBool = scattProb(kondoJArray, size_BZ, dispersion, fractionBZ)[2]
            if all(>(0), scattProbBool[fermiPoints])
                phaseFlags[i] = phaseMaps["FL"]
            elseif !all(==(0), scattProbBool[fermiPoints])
                phaseFlags[i] = phaseMaps["PG"]
            else
                phaseFlags[i] = phaseMaps["MI"]
            end
            for (point, kpoint) in trackPoints
                if isnan(gapTrackers[point]) && scattProbBool[kpoint] == 0
                    gapTrackers[point] = W_val
                end
            end
        end
        return phaseFlags, gapTrackers
    end

    densityOfStates, dispersionArray = getDensityOfStates(tightBindDisp, size_BZ)
    fermiPoints = unique(getIsoEngCont(dispersionArray, 0.0))
    @assert length(fermiPoints) == 2 * size_BZ - 2
    @assert all(==(0), dispersionArray[fermiPoints])

    phaseDiagram = zeros(numPoints, numPoints)
    nodeGap = zeros(numPoints)
    antiNodeGap = zeros(numPoints)
    midPointGap = zeros(numPoints)
    @time results = fetch.(@showprogress [Threads.@spawn PhaseStrip(J_val, W_val_arr, phaseMaps) for J_val in J_val_arr])
    for (i, (phaseResult, gapTrackers)) in enumerate(results)
        phaseDiagram[i, :] = phaseResult
        nodeGap[i] = gapTrackers["N"]
        antiNodeGap[i] = gapTrackers["AN"]
        midPointGap[i] = gapTrackers["M"]
    end
    return phaseDiagram, nodeGap, antiNodeGap, midPointGap
end

