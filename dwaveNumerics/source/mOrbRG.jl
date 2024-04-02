using ProgressMeter
include("./rgFlow.jl")

function main(num_kspace_half::Int64, J_init::Float64, bathIntStr::Float64, orbital::String)
    # ensure that [0, \pi] has odd number of states, so 
    # that the nodal point is well-defined.
    @assert num_kspace_half % 2 == 0
    @assert orbital in ["p", "d"]

    # num_kspace_half is the number of points from 0 until pi,
    # so that the total number of k-points along an axis is 
    # num_kspace_half(for [antinode, pi)) + 1(for node) + num_kspace_half(for (node, antinode])
    num_kspace = 2 * num_kspace_half + 1

    densityOfStates, dispersionArray = getDensityOfStates(tightBindDisp, num_kspace)

    cutOffEnergies = getCutOffEnergy(num_kspace)

    # Kondo coupling must be stored in a 3D matrix. Two of the dimensions store the 
    # incoming and outgoing momentum indices, while the third dimension stores the 
    # behaviour along the RG flow. For example, J[i][j][k] reveals the value of J 
    # for the momentum pair (i,j) at the k^th Rg step.
    kondoJArray = initialiseKondoJ(num_kspace, orbital, length(cutOffEnergies), J_init)

    # define flags to track whether the RG flow for a particular J_{k1, k2} needs to be stopped 
    # (perhaps because it has gone to zero, or its denominator has gone to zero). These flags are
    # initialised to one, which means that by default, the RG can proceed for all the momenta.
    proceed_flags = fill(1, num_kspace^2, num_kspace^2)

    # Run the RG flow starting from the maximum energy, down to the penultimate energy (ΔE), in steps of ΔE
    @showprogress for (stepIndex, energyCutoff) in enumerate(cutOffEnergies[1:end-1])
        deltaEnergy = abs(cutOffEnergies[stepIndex+1] - cutOffEnergies[stepIndex])

        # set the Kondo coupling of all subsequent steps equal to that of the present step 
        # for now, so that we can just add the renormalisation to it later
        kondoJArray[:, :, stepIndex+1] = kondoJArray[:, :, stepIndex]

        # if there are no enabled flags (i.e., all are zero), stop the RG flow
        if all(==(0), proceed_flags)
            kondoJArray[:, :, stepIndex+2:end] .= kondoJArray[:, :, stepIndex]
            break
        end

        innerIndicesArr, excludedVertexPairs, mixedVertexPairs, cutoffPoints, cutoffHolePoints, proceed_flags = highLowSeparation(dispersionArray, energyCutoff, proceed_flags, num_kspace)

        # calculate the renormalisation for this step and for all k1,k2 pairs
        kondoJArrayNext, proceed_flags_updated = stepwiseRenormalisation(
            innerIndicesArr,
            excludedVertexPairs,
            mixedVertexPairs,
            energyCutoff,
            cutoffPoints,
            cutoffHolePoints,
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
