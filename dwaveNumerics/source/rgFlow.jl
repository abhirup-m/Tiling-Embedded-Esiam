using LinearAlgebra
include("./helpers.jl")


function getCutOffEnergy(size_BZ)
    kx_pos_arr = collect(range(K_MIN, K_MAX, length=size_BZ))
    kx_holesector = kx_pos_arr[kx_pos_arr.>=0]
    cutOffEnergies = -tightBindDisp(kx_holesector, 0 .* kx_holesector)
    cutOffEnergies = cutOffEnergies[cutOffEnergies.>=-TOLERANCE]
    return sort(cutOffEnergies, rev=true)
end


function highLowSeparation(dispersionArray::Vector{Float64}, energyCutoff::Float64, proceed_flags::Matrix{Int64}, size_BZ::Int64)

    # get the k-points that will be decoupled at this step, by getting the isoenergetic contour at the cutoff energy.
    cutoffPoints = unique(getIsoEngCont(dispersionArray, energyCutoff))
    cutoffHolePoints = particleHoleTransf(cutoffPoints, size_BZ)

    # these cutoff points will no longer participate in the RG flow, so disable their flags
    proceed_flags[cutoffPoints, :] .= 0
    proceed_flags[:, cutoffPoints] .= 0
    proceed_flags[cutoffHolePoints, :] .= 0
    proceed_flags[:, cutoffHolePoints] .= 0

    # get the k-space points that need to be tracked for renormalisation, by getting the states 
    # below the cutoff energy. We only take points within the lower left quadrant, because the
    # other quadrant is obtained through symmetry relations.
    innerIndicesArr = [
        point for (point, energy) in enumerate(dispersionArray) if
        abs(energy) < (abs(energyCutoff) - TOLERANCE) &&
        map1DTo2D(point, size_BZ)[1] <= 0.5 * (K_MAX + K_MIN)
    ]
    excludedIndicesArr = [
        point for (point, energy) in enumerate(dispersionArray) if
        abs(energy) < (abs(energyCutoff) - TOLERANCE) &&
        map1DTo2D(point, size_BZ)[1] > 0.5 * (K_MAX + K_MIN)
    ]
    excludedVertexPairs = [
        (p1, p2) for p1 in sort(excludedIndicesArr) for
        p2 in sort(excludedIndicesArr)[sort(excludedIndicesArr).>=p1]
    ]
    mixedVertexPairs = [(p1, p2) for p1 in innerIndicesArr for p2 in excludedIndicesArr]
    return innerIndicesArr, excludedVertexPairs, mixedVertexPairs, cutoffPoints, cutoffHolePoints, proceed_flags
end


function initialiseKondoJ(size_BZ::Int64, orbital::String, num_steps::Int64, J_val::Float64)
    @assert orbital in ["d", "p", "poff"]
    # Kondo coupling must be stored in a 3D matrix. Two of the dimensions store the 
    # incoming and outgoing momentum indices, while the third dimension stores the 
    # behaviour along the RG flow. For example, J[i][j][k] reveals the value of J 
    # for the momentum pair (i,j) at the k^th Rg step.
    kondoJArray = Array{Float64}(undef, size_BZ^2, size_BZ^2, num_steps)
    Threads.@threads for p1 = 1:size_BZ^2
        k1x, k1y = map1DTo2D(p1, size_BZ)
        p2_arr = collect(p1:size_BZ^2)
        k2x, k2y = map1DTo2D(p2_arr, size_BZ)
        if orbital == "d"
            kondoJArray[p1, p2_arr, 1] =
                J_val * (cos(k1x) - cos(k1y)) .* (cos.(k2x) - cos.(k2y))
        elseif orbital == "p"
            kondoJArray[p1, p2_arr, 1] = 0.5 * J_val .* (cos.(k1x .- k2x) .+ cos.(k1y .- k2y))
        else
            kondoJArray[p1, p2_arr, 1] =
                J_val * (cos(k1x) + cos(k1y)) .* (cos.(k2x) + cos.(k2y))
        end
        kondoJArray[p2_arr, p1, 1] = kondoJArray[p1, p2_arr, 1]
    end
    return kondoJArray
end


function bathIntForm(
    bathIntStr::Float64,
    orbital::String,
    size_BZ::Int64,
    points,
)
    # bath interaction does not renormalise, so we don't need to make it into a matrix. A function
    # is enough to invoke the W(k1,k2,k3,k4) value whenever we need it. To obtain it, we call the p-wave
    # function for each momentum k_i, then multiply them to get W_1234 = W × p(k1) * p(k2) * p(k3) * p(k4)
    @assert orbital in ["d", "p", "poff"]
    @assert length(points) == 4
    k2_vals = map1DTo2D(points[2], size_BZ)
    k3_vals = map1DTo2D(points[3], size_BZ)
    k4_vals = map1DTo2D(points[4], size_BZ)
    k1_vals = map1DTo2D(points[1], size_BZ)
    if orbital == "d"
        bathInt = bathIntStr
        for (kx, ky) in [k1_vals, k2_vals, k3_vals, k4_vals]
            bathInt = bathInt .* (cos.(kx) - cos.(ky))
        end
        return bathInt
    elseif orbital == "p"
        return 0.5 .* bathIntStr .* (
            cos.(k1_vals[1] .- k2_vals[1] .+ k3_vals[1] .- k4_vals[1]) .+
            cos.(k1_vals[2] .- k2_vals[2] .+ k3_vals[2] .- k4_vals[2])
        )
    else
        bathInt = bathIntStr
        for (kx, ky) in [k1_vals, k2_vals, k3_vals, k4_vals]
            bathInt = bathInt .* (cos.(kx) + cos.(ky))
        end
        return bathInt
    end
end


function deltaJk1k2(
    denominators::Vector{Float64},
    proceed_flagk1k2::Int64,
    kondoJArrayPrev_k1k2::Float64,
    kondoJ_k2q_qk1::Vector{Float64},
    kondoJ_qqbar::Vector{Float64},
    deltaEnergy::Float64,
    bathIntArgs,
    densityOfStates_q::Vector{Float64},
)
    # if the flag is disabled for this momentum pair, don't bother to do the rest
    if proceed_flagk1k2 == 0 || length(denominators) == 0
        return 0, 0
    end

    # the renormalisation itself is given by the expression
    # ΔJ(k1, k2) = ∑_q [J(k2,q)J(q,k1) + 4 * J(q,qbar) * W(qbar, k2, k1, q)]/[ω - E/2 + J(q)/4 + W(q)/2]
    renormalisation =
        -deltaEnergy * sum(
            densityOfStates_q .*
            (kondoJ_k2q_qk1 .+ 4 .* kondoJ_qqbar .* bathIntForm(bathIntArgs...)) ./
            denominators,
        )

    # if a non-zero coupling goes through a zero, we set it to zero, and disable its flag.
    if (kondoJArrayPrev_k1k2 + renormalisation) * kondoJArrayPrev_k1k2 <= 0
        kondoJArrayNext_k1k2 = 0
        proceed_flagk1k2 = 0
    else
        kondoJArrayNext_k1k2 = kondoJArrayPrev_k1k2 + renormalisation
    end
    return kondoJArrayNext_k1k2, proceed_flagk1k2
end


function symmetriseRGFlow(innerIndicesArr, excludedVertexPairs, mixedVertexPairs, size_BZ, kondoJArrayNext, kondoJArrayPrev, proceed_flags)
    Threads.@threads for (innerIndex1, excludedIndex) in mixedVertexPairs
        innerIndex2 = particleHoleTransf(excludedIndex, size_BZ)
        @assert innerIndex1 in innerIndicesArr
        @assert innerIndex2 in innerIndicesArr
        kondoJArrayNext[innerIndex1, excludedIndex] = kondoJArrayNext[excludedIndex, innerIndex1] = -kondoJArrayNext[innerIndex1, innerIndex2]
        proceed_flags[innerIndex1, excludedIndex] = proceed_flags[innerIndex1, innerIndex2]
        proceed_flags[excludedIndex, innerIndex1] = proceed_flags[innerIndex1, innerIndex2]
        if kondoJArrayPrev[innerIndex1, excludedIndex] * kondoJArrayNext[innerIndex1, excludedIndex] < -abs(TOLERANCE)
            kondoJArrayNext[innerIndex1, excludedIndex] = kondoJArrayNext[excludedIndex, innerIndex1] = 0
            proceed_flags[innerIndex1, excludedIndex] = proceed_flags[excludedIndex, innerIndex1] = 0
        end
    end
    Threads.@threads for (index1, index2) in excludedVertexPairs
        sourcePoint1, sourcePoint2 = particleHoleTransf([index1, index2], size_BZ)
        @assert sourcePoint1 in innerIndicesArr
        @assert sourcePoint2 in innerIndicesArr
        kondoJArrayNext[index1, index2] = kondoJArrayNext[index2, index1] = kondoJArrayNext[sourcePoint1, sourcePoint2]
        proceed_flags[index1, index2] = proceed_flags[sourcePoint1, sourcePoint2]
        proceed_flags[index2, index1] = proceed_flags[sourcePoint1, sourcePoint2]
        if kondoJArrayPrev[index1, index2] * kondoJArrayNext[index1, index2] < -abs(TOLERANCE)
            kondoJArrayNext[index1, index2] = 0
            kondoJArrayNext[index2, index1] = 0
            proceed_flags[index1, index2] = proceed_flags[index2, index1] = 0
        end
    end
    return kondoJArrayNext, proceed_flags
end


function stepwiseRenormalisation(
    innerIndicesArr::Vector{Int64},
    excludedVertexPairs::Vector{Tuple{Int64,Int64}},
    mixedVertexPairs::Vector{Tuple{Int64,Int64}},
    energyCutoff::Float64,
    cutoffPoints::Vector{Int64},
    cutoffHolePoints::Vector{Int64},
    proceed_flags::Matrix{Int64},
    kondoJArrayPrev::Array{Float64,2},
    kondoJArrayNext::Array{Float64,2},
    bathIntStr::Float64,
    size_BZ::Int64,
    deltaEnergy::Float64,
    orbital::String,
    densityOfStates::Vector{Float64},
)

    # construct denominators for the RG equation, given by
    # d = ω - E/2 + J(q)/4 + W(q)/2

    denominators =
        OMEGA .- abs(energyCutoff) / 2 .+ diag(
            kondoJArrayPrev[cutoffPoints, cutoffPoints] / 4 .+
            bathIntForm(
                bathIntStr,
                orbital,
                size_BZ,
                [cutoffPoints, cutoffPoints, cutoffPoints, cutoffPoints],
            ) / 2,
        )

    # println(denominators[1])
    # only consider those terms whose denominator haven't gone through zeros
    cutoffPoints = cutoffPoints[denominators.<0]
    cutoffHolePoints = cutoffHolePoints[denominators.<0]
    denominators = denominators[denominators.<0]

    if length(cutoffPoints) == 0
        proceed_flags[:, :] .= 0
        return kondoJArrayNext, proceed_flags
    end

    # loop over (k1, k2) pairs that represent the momentum states within the emergent window,
    # so that we can calculate the renormalisation of J(k1, k2), for all k1, k2.
    externalVertexPairs = [
        (p1, p2) for p1 in sort(innerIndicesArr) for
        p2 in sort(innerIndicesArr)[sort(innerIndicesArr).>=p1]
    ]
    kondoJ_qq_bar = diag(kondoJArrayPrev[cutoffPoints, cutoffHolePoints])
    dOfStates_cutoff = densityOfStates[cutoffPoints]
    Threads.@threads for (innerIndex1, innerIndex2) in externalVertexPairs
        kondoJ_k2_q_times_kondoJ_q_k1 = diag(kondoJArrayPrev[innerIndex2, cutoffPoints] * kondoJArrayPrev[innerIndex1, cutoffPoints]')
        kondoJArrayNext_k1k2, proceed_flag_k1k2 = deltaJk1k2(
            denominators,
            proceed_flags[innerIndex1, innerIndex2],
            kondoJArrayPrev[innerIndex1, innerIndex2],
            kondoJ_k2_q_times_kondoJ_q_k1,
            kondoJ_qq_bar,
            deltaEnergy,
            [
                bathIntStr,
                orbital,
                size_BZ,
                [cutoffHolePoints, innerIndex2, innerIndex1, cutoffPoints],
            ],
            dOfStates_cutoff,
        )
        kondoJArrayNext[innerIndex1, innerIndex2] = kondoJArrayNext_k1k2
        kondoJArrayNext[innerIndex2, innerIndex1] = kondoJArrayNext_k1k2
        proceed_flags[innerIndex1, innerIndex2] = proceed_flag_k1k2
        proceed_flags[innerIndex2, innerIndex1] = proceed_flag_k1k2
    end
    kondoJArrayNext, proceed_flags = symmetriseRGFlow(innerIndicesArr, excludedVertexPairs, mixedVertexPairs, size_BZ, kondoJArrayNext, kondoJArrayPrev, proceed_flags)
    return kondoJArrayNext, proceed_flags
end
