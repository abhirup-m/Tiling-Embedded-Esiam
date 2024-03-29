using LinearAlgebra
using ProgressMeter
include("./helpers.jl")
include("./constants.jl")



function bathIntForm(
    bathIntStr::Float64,
    orbital::String,
    num_kspace::Int64,
    points::Vector,
)
    # bath interaction does not renormalise, so we don't need to make it into a matrix. A function
    # is enough to invoke the W(k1,k2,k3,k4) value whenever we need it. To obtain it, we call the p-wave
    # function for each momentum k_i, then multiply them to get W_1234 = W × p(k1) * p(k2) * p(k3) * p(k4)
    @assert orbital == "d" || orbital == "p"
    @assert length(points) == 4
    k2_vals = map1DTo2D(points[2], num_kspace)
    k3_vals = map1DTo2D(points[3], num_kspace)
    k4_vals = map1DTo2D(points[4], num_kspace)
    k1_vals = map1DTo2D(points[1], num_kspace)
    if orbital == "d"
        bathInt = bathIntStr
        for (kx, ky) in [k1_vals, k2_vals, k3_vals, k4_vals]
            bathInt = bathInt .* (cos.(kx) - cos.(ky))
        end
        return bathInt
    else
        return 0.5 .* bathIntStr .* (
            cos.(k1_vals[1] .- k2_vals[1] .+ k3_vals[1] .- k4_vals[1]) .+
            cos.(k1_vals[2] .- k2_vals[2] .+ k3_vals[2] .- k4_vals[2])
        )
    end
end


function stepwiseRenormalisation(
    innerIndicesArr::Vector{Int64},
    energyCutoff::Float64,
    cutoffPoints::Vector{Int64},
    proceed_flags::Matrix{Int64},
    kondoJArrayPrev::Array{Float64,2},
    kondoJArrayNext::Array{Float64,2},
    bathIntStr::Float64,
    num_kspace::Int64,
    deltaEnergy::Float64,
    orbital::String,
    densityOfStates::Vector{Float64},
)

    # construct denominators for the RG equation, given by
    # d = ω - E/2 + J(q)/4 + W(q)/2
    omega = -abs(energyCutoff) / 2
    denominators = [
        omega - abs(energyCutoff) / 2 +
        kondoJArrayPrev[qpoint, qpoint] / 4 +
        bathIntForm(bathIntStr, orbital, num_kspace, [qpoint, qpoint, qpoint, qpoint]) / 2 for qpoint in cutoffPoints
    ]

    # only consider those terms whose denominator haven't gone through zeros
    cutoffPoints = cutoffPoints[denominators.<0]
    denominators = denominators[denominators.<0]

    # obtain the hole counterparts of the UV states
    cutoffHolePoints = particleHoleTransf(cutoffPoints, num_kspace)

    # loop over (k1, k2) pairs that represent the momentum states within the emergent window,
    # so that we can calculate the renormalisation of J(k1, k2), for all k1, k2.
    externalVertexPairs = collect(Iterators.product(innerIndicesArr, innerIndicesArr))
    Threads.@threads for (innerIndex1, innerIndex2) in externalVertexPairs

        # if the flag is disabled for this momentum pair, don't bother to do the rest
        if proceed_flags[innerIndex1, innerIndex2] == 0
            continue
        end

        # the renormalisation itself is given by the expression
        # ΔJ(k1, k2) = ∑_q [J(k2,q)J(q,k1) + 4 * J(q,qbar) * W(qbar, k2, k1, q)]/[ω - E/2 + J(q)/4 + W(q)/2]
        kondoJArrayNext[innerIndex1, innerIndex2] +=
            -deltaEnergy * sum(
                densityOfStates[cutoffPoints] .* (
                    kondoJArrayPrev[innerIndex1, cutoffPoints] .*
                    kondoJArrayPrev[innerIndex2, cutoffPoints] .+
                    4 .* diag(kondoJArrayPrev[cutoffPoints, cutoffHolePoints]) .*
                    bathIntForm(
                        bathIntStr,
                        orbital,
                        num_kspace,
                        [cutoffHolePoints, innerIndex2, innerIndex1, cutoffPoints],
                    )
                ) ./ denominators,
            )

        # if a non-zero coupling goes through a zero, we set it to zero, and disable its flag.
        if kondoJArrayNext[innerIndex1, innerIndex2] *
           kondoJArrayPrev[innerIndex1, innerIndex2] <= 0
            kondoJArrayNext[innerIndex1, innerIndex2] = 0
            proceed_flags[innerIndex1, innerIndex2] = 0
        end

    end
    return kondoJArrayNext, proceed_flags
end


function main(num_kspace_half::Int64, J_init::Float64, bathIntStr::Float64, orbital::String)
    # ensure that [0, \pi] has odd number of states, so 
    # that the nodal point is well-defined.
    @assert num_kspace_half % 2 == 0
    @assert orbital in ["p", "d"]

    # num_kspace_half is the number of points from 0 until pi,
    # so that the total number of k-points along an axis is 
    # num_kspace_half(for [antinode, pi)) + 1(for node) + num_kspace_half(for (node, antinode])
    num_kspace = 2 * num_kspace_half + 1

    # create flattened array of momenta, of the form
    # (kx1, ky1), (kx2, ky1), ..., (kxN, ky1), (kx1, ky2), ..., (kxN, kyN)
    kx_arr_flat =
        repeat(range(K_MIN, stop = K_MAX, length = num_kspace), inner = num_kspace)
    ky_arr_flat =
        repeat(range(K_MIN, stop = K_MAX, length = num_kspace), outer = num_kspace)
    kx_pos_arr = collect(range(0, K_MAX, length = num_kspace_half + 1))

    densityOfStates, dispersionArray = getDensityOfStates(tightBindDisp, num_kspace)
    cutOffEnergies = -tightBindDisp(kx_pos_arr, 0 .* kx_pos_arr)
    cutOffEnergies = cutOffEnergies[cutOffEnergies.>=-TOLERANCE]

    # Kondo coupling must be stored in a 3D matrix. Two of the dimensions store the 
    # incoming and outgoing momentum indices, while the third dimension stores the 
    # behaviour along the RG flow. For example, J[i][j][k] reveals the value of J 
    # for the momentum pair (i,j) at the k^th Rg step.
    kondoJArray = Array{Float64}(undef, num_kspace^2, num_kspace^2, length(cutOffEnergies))

    # bare value of J is given by J(k1, k2) = J_init × [cos(k1x) - cos(k1y)] × [cos(k2x) - cos(k2y)] 
    # The full matrix can therefore be seen as the multiplication [d(k1) d(k2) ... d(kN)]^T × [d(k1) ... d(kN)]
    k1x = repeat(kx_arr_flat', inner = (num_kspace^2, 1))
    k2x = repeat(kx_arr_flat, inner = (1, num_kspace^2))
    k1y = repeat(ky_arr_flat', inner = (num_kspace^2, 1))
    k2y = repeat(ky_arr_flat, inner = (1, num_kspace^2))
    if orbital == "p"
        kondoJArray[:, :, 1] .= J_init .* (cos.(k1x - k2x) + cos.(k1y - k2y))
    else
        kondoJArray[:, :, 1] .= J_init .* (cos.(k1x) + cos.(k2x) - cos.(k1y) - cos.(k2y))
    end

    # define flags to track whether the RG flow for a particular J_{k1, k2} needs to be stopped 
    # (perhaps because it has gone to zero, or its denominator has gone to zero). These flags are
    # initialised to one, which means that by default, the RG can proceed for all the momenta.
    proceed_flags = fill(1, num_kspace^2, num_kspace^2)

    # Run the RG flow starting from the maximum energy, down to the penultimate energy (ΔE), in steps of ΔE
    @showprogress for (stepIndex, energyCutoff) in enumerate(cutOffEnergies[1:end-1])
        deltaEnergy = abs(cutOffEnergies[stepIndex+1] - cutOffEnergies[stepIndex])

        # set the Kondo coupling of all subsequent steps equal to that of the present step 
        # for now, so that we can just add the renormalisation to it later
        kondoJArray[:, :, stepIndex+1:end] .= kondoJArray[:, :, stepIndex]

        # get the k-points that will be decoupled at this step, by getting the isoenergetic contour at the cutoff energy.
        cutoffPoints = getIsoEngCont(dispersionArray, energyCutoff)

        # these cutoff points will no longer participate in the RG flow, so disable their flags
        proceed_flags[cutoffPoints, :] .= 0
        proceed_flags[:, cutoffPoints] .= 0

        # if there are no enabled flags (i.e., all are zero), stop the RG flow
        if all(==(0), proceed_flags)
            break
        end

        # get the k-space points that need to be tracked for renormalisation, by getting the states 
        # below the cutoff energy as well within the lower left quadrant
        innerIndicesArr = [
            point for (point, energy) in enumerate(dispersionArray) if
            abs(energy) < abs(energyCutoff)
        ]

        # calculate the renormalisation for this step and for all k1,k2 pairs
        kondoJArrayNext, proceed_flags_updated = stepwiseRenormalisation(
            innerIndicesArr,
            energyCutoff,
            cutoffPoints,
            proceed_flags,
            kondoJArray[:, :, stepIndex],
            kondoJArray[:, :, stepIndex+1],
            bathIntStr,
            num_kspace,
            deltaEnergy,
            orbital,
            densityOfStates,
        )
        kondoJArray[:, :, stepIndex+1] = kondoJArrayNext
        proceed_flags = proceed_flags_updated
    end
    return kondoJArray, dispersionArray
end
