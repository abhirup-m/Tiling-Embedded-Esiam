using LinearAlgebra

function getForwardScattFlow(kondoJArray::Array{Float64,3}, num_kspace_half::Int64)
    # given the entire matrix of the RG flow of Kondo couplings, extract the RG flow
    # of the three sets of couplings {J(kFN, k2) for all k2}, {J(kFAN, k2) for all k2}
    # and {J(kFM, k2) for all k2}, where kFN, kFAN and kFM are the nodal point, the
    # antinodal point and the point on the Fermi surface midway between these two points.
    # In each set, k2 runs over the set of k-states in the lower left quadrant that lie on
    # the isoneergetic shell immediately above the Fermi surface

    num_kspace = 2 * num_kspace_half + 1

    # get the points along the arm of the Fermi surface in the south west quadrant, as well those
    # on the isoenergetic shell just above it, in the same quadrant.
    FermiArmSouthWest =
        [num_kspace_half + 1 + j * (num_kspace - 1) for j = 0:(num_kspace_half)]
    FermiArmAbove = [
        point - 1 for
        point in FermiArmSouthWest if 1 <= (point - 1) % num_kspace <= num_kspace_half
    ]
    kx_arr_flat = repeat(range(-pi, stop = pi, length = num_kspace), inner = num_kspace)
    ky_arr_flat = repeat(range(-pi, stop = pi, length = num_kspace), outer = num_kspace)

    # define coordinates of the antinodal, midway and nodal points
    antinodeD = FermiArmSouthWest[1]
    midway = FermiArmSouthWest[div(1 + num_kspace_half, 4)+1]
    node = FermiArmSouthWest[div(1 + num_kspace_half, 2)] # this point is actually just beside the nodal point (the exact nodal point gives zero contribution)

    results = zeros((3, length(FermiArmAbove), num_kspace_half + 1))
    # loop over the initial and final momentum states and store the RG flow of the corresponding couplings
    for (j, point) in enumerate([antinodeD, midway, node])
        for (k, abovepoint) in enumerate(FermiArmAbove)
            results[j, k, :] = kondoJArray[point, abovepoint, :]
        end
    end

    # we also return the momentum values of the initial states: AN=(0, pi), N=(pi/2, pi/2), 
    # etc, for purposes of displaying in figures
    return results,
    [
        round.((kx_arr_flat[point], ky_arr_flat[point]) ./ pi, digits = 2) for
        point in [antinodeD, midway, node]
    ]
end


function getBackScattFlow(kondoJArray::Array{Float64,3}, num_kspace_half::Int64)
    # extract the RG flow of backscattering couplings (for eg, node to node, antinode to antinode, etc)
    num_kspace = 2 * num_kspace_half + 1

    # extract coordinates of the south west and north east arms of the Fermi surface, since backscattering
    # is just the coupling between the i^th point of the SW arm and the i^th point of the NE arm.
    FermiArmSouthWest =
        [num_kspace_half + 1 + j * (num_kspace - 1) for j = 0:(num_kspace_half)]
    FermiArmNorthEast = [
        num_kspace * (num_kspace_half + 1) + j * (num_kspace - 1) for
        j = 0:(num_kspace_half)
    ]

    results = zeros((length(FermiArmSouthWest), num_kspace_half + 1))
    k = 1
    for (p1, p2) in zip(FermiArmSouthWest, FermiArmNorthEast)
        results[k, :] = kondoJArray[p1, p2, :] ./ kondoJArray[p1, p2, 1]
        k += 1
    end
    return results
end


function getBackScattFlowBool(kondoJArray::Array{Float64,3}, num_kspace_half::Int64)
    # extract the RG flow of backscattering couplings (for eg, node to node, antinode to antinode, etc)
    num_kspace = 2 * num_kspace_half + 1

    # extract coordinates of the south west and north east arms of the Fermi surface, since backscattering
    # is just the coupling between the i^th point of the SW arm and the i^th point of the NE arm.
    FermiArmSouthWest =
        [num_kspace_half + 1 + j * (num_kspace - 1) for j = 0:(num_kspace_half)]
    FermiArmNorthEast = [
        num_kspace * (num_kspace_half + 1) + j * (num_kspace - 1) for
        j = 0:(num_kspace_half)
    ]

    results = zeros((length(FermiArmSouthWest), num_kspace_half + 1))
    k = 1
    for (p1, p2) in zip(FermiArmSouthWest, FermiArmNorthEast)
        # sign.(kondoJArray[point,point,:]) .* (abs.(kondoJArray[point,point,:] ./ kondoJArray[point,point,1]) .>= 1)
        results[k, :] =
            sign.(kondoJArray[p1, p2, :]) .*
            (abs.(kondoJArray[p1, p2, :] ./ kondoJArray[p1, p2, 1]) .>= 1)
        k += 1
    end
    return results
end


function getInStateScattFlow(kondoJArray::Array{Float64,3}, num_kspace_half::Int64)
    # extract the in-place scattering Rg flow (essentially, 
    # the RG flow of the couplings {J(k,k), k in FS}.

    num_kspace = 2 * num_kspace_half + 1
    FermiArmSouthWest =
        [num_kspace_half + 1 + j * (num_kspace - 1) for j = 0:(num_kspace_half)]
    results = zeros((length(FermiArmSouthWest), num_kspace_half + 1))
    for (k, point) in enumerate(FermiArmSouthWest)
        results[k, :] = kondoJArray[point, point, :] ./ kondoJArray[point, point, 1]
    end
    return results
end


function getInStateScattFlowBool(kondoJArray::Array{Float64,3}, num_kspace_half::Int64)
    # extract the in-place scattering Rg flow (essentially, 
    # the RG flow of the couplings {J(k,k), k in FS}.

    num_kspace = 2 * num_kspace_half + 1
    FermiArmSouthWest =
        [num_kspace_half + 1 + j * (num_kspace - 1) for j = 0:(num_kspace_half)]
    results = zeros((length(FermiArmSouthWest), num_kspace_half + 1))
    for (k, point) in enumerate(FermiArmSouthWest)
        results[k, :] .=
            sign.(kondoJArray[point, point, :]) .*
            (sign.(kondoJArray[point, point, :] .- kondoJArray[point, point, 1]) .+ 1)
    end
    return results
end


function getGlobalFlowBool(kondoJArray::Array{Float64,3}, num_kspace_half::Int64, stepIndex::Int64)
    num_kspace = 2 * num_kspace_half + 1
    bare_J_squared = reshape(
        diag(kondoJArray[:, :, 1] * kondoJArray[:, :, 1]'),
        (num_kspace, num_kspace),
    )
    results =
        sign.(
            reshape(
                diag(kondoJArray[:, :, stepIndex] * kondoJArray[:, :, stepIndex]'),
                (num_kspace, num_kspace),
            ) .- bare_J_squared
        )
    return results
end


function getGlobalFlow(kondoJArray::Array{Float64,3}, num_kspace_half::Int64, stepIndex::Int64)
    num_kspace = 2 * num_kspace_half + 1
    bare_J_squared = reshape(
        diag(kondoJArray[:, :, 1] * kondoJArray[:, :, 1]'),
        (num_kspace, num_kspace),
    )
    results =
        reshape(
            diag(kondoJArray[:, :, stepIndex] * kondoJArray[:, :, stepIndex]'),
            (num_kspace, num_kspace),
        ) ./ bare_J_squared
    return results
end
