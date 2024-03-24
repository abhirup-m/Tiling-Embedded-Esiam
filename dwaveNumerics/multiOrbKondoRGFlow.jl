using LinearAlgebra

# helper functions for switching back and forth between the 1D flattened representation (1 → N^2) 
# and the 2D representation ((1 → N)×(1 → N))
map1DTo2D(point::Int64, num_kspace::Int64) = [(point - 1) % num_kspace + 1, div((point - 1), num_kspace) + 1]
map2DTo1D(kx_index::Int64, ky_index::Int64, num_kspace::Int64) = kx_index + (ky_index - 1) * num_kspace

# create tight-binding dispersion
tightBindDispersion(t::Float64, kx_arr_flat::Array{Float64}, ky_arr_flat::Array{Float64}) = -2 * t .* (cos.(kx_arr_flat) + cos.(ky_arr_flat))

# d-wave and p-wave generation functions
dwaveKdep(kx::Float64, ky::Float64) = cos.(kx) - cos.(ky)
pwaveKdep(kx_arr::Vector{Float64}, ky_arr::Vector{Float64}) = cos(kx_arr[1] - kx_arr[2] + kx_arr[3] - kx_arr[4]) + cos(ky_arr[1] - ky_arr[2] + ky_arr[3] - ky_arr[4])

function getdensityOfStates(num_kspace::Int64, dispersionArray::Vector{Float64})
	# momentum space interval is 2pi/N
    delta_k = 2 * pi / num_kspace

	# since each momentum interval has one state, k-space DOS 
	# is Δ N/Δ k = 1/Δ K
    kspaceDos = 1 / delta_k
    densityOfStates = zeros(num_kspace * num_kspace)
    Threads.@threads for j in 0:num_kspace-1
        for i in 1:num_kspace
            E_xy = dispersionArray[j * num_kspace + i]
            E_xpy = dispersionArray[j * num_kspace + i % num_kspace + 1]
            E_xyp = dispersionArray[(j * num_kspace + num_kspace) % num_kspace^2 + i]

			# E-space DOS is Δ N / Δ E = DOS_k * Δ K / Δ E.
			# In 2D, Δ E = sqrt(ΔE^2_x + ΔE^2_y) (just the gradient of E(kx, ky))
            dE_xydk = sqrt((E_xpy - E_xy)^2 / delta_k^2
                           + (E_xyp - E_xy)^2 / delta_k^2
                           )
            densityOfStates[j * num_kspace + i] = kspaceDos / dE_xydk
        end
    end

	# van-Hoves might result in Nan, replace them with the largest finite value
    replace!(densityOfStates, Inf=>maximum(densityOfStates[densityOfStates .≠ Inf]))

	# normalise the DOS according to the convention \int dE \rho(E) = N (where N is the total number of k-states)
	normalisation = sum([dos * abs(dispersionArray[i % num_kspace + 1] - dispersionArray[i]) for (i, dos) in enumerate(densityOfStates)]) / num_kspace^2
    return densityOfStates / normalisation
end


function getIsoEnergeticContour(dispersionArray::Vector{Float64}, num_kspace::Int64, energy::Float64)
	# obtain the k-space points that have the specified energy. We count one point per row.
    contourPoints = [[] for j in 1:num_kspace]
    Threads.@threads for j in 0:num_kspace-1 # loop over ky
		# get array of differences E(k_x, k_y) - E for all k_x and fixed k_y
        energyDiffArr = dispersionArray[j * num_kspace + 1: (j + 1) * num_kspace] .- energy
        for i in 1:num_kspace
			# check of E(k_x - k_y) - E changes sign at some k_x, 
			# this should be the k_X where E(k_x, k_y) matches E
            if energyDiffArr[i] * energyDiffArr[i % num_kspace + 1] <= 0
                push!(contourPoints[j+1], j * num_kspace + i + 1)
				break
            end
        end
    end
    return collect(Iterators.flatten(contourPoints))
end


function stepwiseRenormalisation(innerIndicesArr::Vector{Int64}, energyCutoff::Float64, cutoffPoints::Vector{Int64}, proceed_flags::Matrix{Int64}, kondoJArrayPrev::Array{Float64, 2}, kondoJArrayNext::Array{Float64, 2}, bathInt, num_kspace::Int64, deltaEnergy::Float64, densityOfStates::Vector{Float64})
    omega = -abs(energyCutoff) / 2
    Threads.@threads for (innerIndex1, innerIndex2) in collect(Iterators.product(innerIndicesArr, innerIndicesArr))
        denominators = [omega - abs(energyCutoff) / 2 + kondoJArrayPrev[qpoint, qpoint] / 4 + 
            bathInt([qpoint, qpoint, qpoint, qpoint]) / 2 for qpoint in cutoffPoints]
        
        # if the flag is disabled for this momentum pair, don't bother to do the rest
        if proceed_flags[innerIndex1,innerIndex2] == 0
            continue
        end

        # loop over the cutoff momentum states
        for (qpoint, denominator) in zip(cutoffPoints, denominators)

            # only consider those terms whose denominator haven't gone through zeros
            if  denominator < 0

                # the expression for renormalisation is -∑ ΔE × ρ(q) [J(k1,q) J(q,k2) + 4J(q,-q) W(-q,q,k1,k2)]/denominator(q). 
                # since we are summing q only over one quadrant, we multiply a factor of 4 in front of ΔE. We also replace -q with q,
                # because each -q brings a minus sign, so J(q,-q) W(-q,q,k1,k2) = (-J(q,q))(-W(q,q,...))
                kondoJArrayNext[innerIndex1,innerIndex2] += -4 * deltaEnergy * densityOfStates[qpoint] * (kondoJArrayPrev[innerIndex1, qpoint] * kondoJArrayPrev[innerIndex2, qpoint] + 4 * kondoJArrayPrev[qpoint, qpoint] * bathInt([innerIndex1,innerIndex2,qpoint, qpoint])) / denominator
            end
        end

        # if a non-zero coupling goes through a zero, we set it to zero, and disable its flag.
        if kondoJArrayNext[innerIndex1,innerIndex2] * kondoJArrayPrev[innerIndex1,innerIndex2] <= 0 && kondoJArrayPrev[innerIndex1,innerIndex2] != 0
            kondoJArrayNext[innerIndex1,innerIndex2] = kondoJArrayPrev[innerIndex1,innerIndex2]
            proceed_flags[innerIndex1,innerIndex2] = 0
        end

        ## having obtained the renormalisation for one quadrant, we now obtain the reflected 
        # points in the other three quadrants, and set their renormalisation equal to the one we have obtained.
        telePoints1 = teleportToAllQuads(innerIndex1, num_kspace)
        telePoints2 = teleportToAllQuads(innerIndex2, num_kspace)
        for (point1, point2) in Iterators.product(telePoints1, telePoints2)
            kondoJArrayNext[point1,point2] = kondoJArrayNext[innerIndex1,innerIndex2]
        end
    end
    return kondoJArrayNext, proceed_flags
end


function multiOrbKondoRGFlow(num_kspace_half::Int64, t::Float64, J_init::Float64, W_val::Float64, orbs::String)
	# ensure that [0, \pi] has odd number of states, so 
	# that the nodal point is well-defined.
    @assert num_kspace_half % 2 == 0
    @assert orbs in ["pp", "pd", "dp", "dd"]

	# num_kspace_half is the number of points from 0 until pi,
	# so that the total number of k-points along an axis is 
	# num_kspace_half(for [antinode, pi)) + 1(for node) + num_kspace_half(for (node, antinode])
    num_kspace = 2 * num_kspace_half + 1

	# create flattened array of momenta, of the form
	# (kx1, ky1), (kx2, ky1), ..., (kxN, ky1), (kx1, ky2), ..., (kxN, kyN)
    kx_arr_flat = repeat(range(-pi, stop=pi, length=num_kspace), inner=num_kspace)
    ky_arr_flat = repeat(range(-pi, stop=pi, length=num_kspace), outer=num_kspace)

    dispersionArray = tightBindDispersion(t, kx_arr_flat, ky_arr_flat)
    densityOfStates = getdensityOfStates(num_kspace, dispersionArray)

	# we divide the positive half of the spectrum into num_kspace chunks
    deltaEnergy = maximum(dispersionArray) / num_kspace_half

	# Kondo coupling must be stored in a 3D matrix. Two of the dimensions store the 
    # incoming and outgoing momentum indices, while the third dimension stores the 
    # behaviour along the RG flow. For example, J[i][j][k] reveals the value of J 
    # for the momentum pair (i,j) at the k^th Rg step.
    kondoJArray = Array{Float64}(undef, num_kspace^2, num_kspace^2, num_kspace_half+1)

    # bare value of J is given by J(k1, k2) = J_init × [cos(k1x) - cos(k1y)] × [cos(k2x) - cos(k2y)] 
    # The full matrix can therefore be seen as the multiplication [d(k1) d(k2) ... d(kN)]^T × [d(k1) ... d(kN)]
    if orbs[1] == 'p'
        kondoJArray[:,:,1] .= J_init .* (pwaveKdep.(kx_arr_flat, ky_arr_flat) * pwaveKdep.(kx_arr_flat, ky_arr_flat)')
    else
        kondoJArray[:,:,1] .= J_init .* (dwaveKdep.(kx_arr_flat, ky_arr_flat) * dwaveKdep.(kx_arr_flat, ky_arr_flat)')
    end 
	# bath interaction does not renormalise, so we don't need to make it into a matrix. A function
	# is enough to invoke the W(k1,k2,k3,k4) value whenever we need it. To obtain it, we call the p-wave
    # function for each momentum k_i, then multiply them to get W_1234 = W × p(k1) * p(k2) * p(k3) * p(k4)
    bathInt(momenta) = orbs[2] == 'd' ? W_val * prod([dwaveKdep(kx_arr_flat[momentum], ky_arr_flat[momentum]) for momentum in momenta]) : W_val * pwaveKdep(kx_arr_flat[momenta], ky_arr_flat[momenta])

    
	# indices of the points within the lower left quadrant. We need these so that we can track the RG 
	# of just these points, since the other points can then be reconstructed from these points using symmetries.
	quadrantIndices = [kx_ind + ky_ind * num_kspace for (ky_ind, kx_ind) in Iterators.product(0:num_kspace-1, 1:num_kspace) if kx_ind <= num_kspace_half + 1 && ky_ind <= num_kspace_half]

    # define flags to track whether the RG flow for a particular J_{k1, k2} needs to be stopped 
    # (perhaps because it has gone to zero, or its denominator has gone to zero). These flags are
    # initialised to one, which means that by default, the RG can proceed for all the momenta.
    proceed_flags = fill(1, num_kspace^2, num_kspace^2)

    # Run the RG flow starting from the maximum energy, down to the penultimate energy (ΔE), in steps of ΔE
    @showprogress for (stepIndex, energyCutoff) in enumerate(maximum(dispersionArray):-deltaEnergy:deltaEnergy)
        # set the Kondo coupling of the next step equal to that of the present step 
        # for now, so that we can just add the renormalisation to it later
		kondoJArray[:,:,stepIndex+1:end] .= kondoJArray[:,:,stepIndex]

        # get the k-points that will be decoupled at this step, by getting the isoenergetic contour at the cutoff energy.
        # then filter them by keeping only the points in the lower left quadrant, because those in the other quadrants give the same contribution
		cutoffPoints = [qpoint for qpoint in getIsoEnergeticContour(dispersionArray, num_kspace, energyCutoff) if qpoint in quadrantIndices]

        # these cutoff points will no longer participate in the RG flow, so disable their flags
        proceed_flags[cutoffPoints,:] .= 0
        proceed_flags[:,cutoffPoints] .= 0

        # if there are no enabled flags (i.e., all are zero), stop the RG flow
        all(==(0), proceed_flags) ? break : nothing

        # get the k-space points that need to be tracked for renormalisation, by getting the states 
        # below the cutoff energy as well within the lower left quadrant
		innerIndicesArr = [point for (point, energy) in enumerate(dispersionArray) if abs(energy) < abs(energyCutoff) && point in quadrantIndices]

        kondoJArrayNext, proceed_flags_updated = stepwiseRenormalisation(innerIndicesArr, energyCutoff, cutoffPoints,
                                                                         proceed_flags, kondoJArray[:,:,stepIndex], 
                                                                         kondoJArray[:,:,stepIndex+1], bathInt, num_kspace,
                                                                         deltaEnergy, densityOfStates)
        kondoJArray[:,:,stepIndex+1] = kondoJArrayNext
        proceed_flags = proceed_flags_updated
    end
    return kondoJArray, dispersionArray
end

function teleportToAllQuads(point::Int64, num_kspace::Int64)
    # given a point in the lower left quadrant, find other point symmetric to this,
    # through the tranformation kx -> 2\pi - kx, ky -> 2pi - ky.
    # -------------------
    # | x             x |
    # |                 |
    # |                 |
    # | o             x |
    # -------------------
    kx_index_L, ky_index_D = map1DTo2D(point, num_kspace)
    kx_index_R = (num_kspace + 1) - kx_index_L # gives kx' = 2pi - kx
    ky_index_U = (num_kspace + 1) - ky_index_D # gives ky' = 2pi - ky

    # get all four points by combining the transformed coordinates:
    # (for eg., (kx, ky), (kx', ky), (kx, ky'), (kx', ky')
	telepPoints = [map2DTo1D(kx_index, ky_index, num_kspace) for (kx_index, ky_index) in 
				   [[kx_index_L, ky_index_D], [kx_index_L, ky_index_U], [kx_index_R, ky_index_D], [kx_index_R, ky_index_U]]]
	return telepPoints
end


function getForwardScattFlow(kondoJArray::Array{Float64, 3}, num_kspace_half::Int64)
    # given the entire matrix of the RG flow of Kondo couplings, extract the RG flow
    # of the three sets of couplings {J(kFN, k2) for all k2}, {J(kFAN, k2) for all k2}
    # and {J(kFM, k2) for all k2}, where kFN, kFAN and kFM are the nodal point, the
    # antinodal point and the point on the Fermi surface midway between these two points.
    # In each set, k2 runs over the set of k-states in the lower left quadrant that lie on
    # the isoneergetic shell immediately above the Fermi surface
    
    num_kspace = 2 * num_kspace_half + 1

    # get the points along the arm of the Fermi surface in the south west quadrant, as well those
    # on the isoenergetic shell just above it, in the same quadrant.
    FermiArmSouthWest = [num_kspace_half + 1 + j * (num_kspace - 1) for j in 0:(num_kspace_half)]
    FermiArmAbove = [point - 1 for point in FermiArmSouthWest if 1 <= (point - 1) % num_kspace <= num_kspace_half]
    kx_arr_flat = repeat(range(-pi, stop=pi, length=num_kspace), inner=num_kspace)
    ky_arr_flat = repeat(range(-pi, stop=pi, length=num_kspace), outer=num_kspace)

    # define coordinates of the antinodal, midway and nodal points
    antinodeD = FermiArmSouthWest[1]
    midway = FermiArmSouthWest[div(1 + num_kspace_half, 4) + 1]
    node = FermiArmSouthWest[div(1 + num_kspace_half, 2)] # this point is actually just beside the nodal point (the exact nodal point gives zero contribution)
    
    results = zeros((3, length(FermiArmAbove), num_kspace_half + 1))
    # loop over the initial and final momentum states and store the RG flow of the corresponding couplings
    for (j, point) in enumerate([antinodeD, midway, node])
        for (k, abovepoint) in enumerate(FermiArmAbove)
            results[j,k,:] = kondoJArray[point,abovepoint,:]
        end
    end

    # we also return the momentum values of the initial states: AN=(0, pi), N=(pi/2, pi/2), 
    # etc, for purposes of displaying in figures
    return results, [round.((kx_arr_flat[point], ky_arr_flat[point]) ./ pi, digits=2) for point in [antinodeD, midway, node]]
end


function getBackScattFlow(kondoJArray::Array{Float64, 3}, num_kspace_half::Int64)
    # extract the RG flow of backscattering couplings (for eg, node to node, antinode to antinode, etc)
    num_kspace = 2 * num_kspace_half + 1

    # extract coordinates of the south west and north east arms of the Fermi surface, since backscattering
    # is just the coupling between the i^th point of the SW arm and the i^th point of the NE arm.
    FermiArmSouthWest = [num_kspace_half + 1 + j * (num_kspace - 1) for j in 0:(num_kspace_half)]
    FermiArmNorthEast = [num_kspace * (num_kspace_half + 1) + j * (num_kspace - 1) for j in 0:(num_kspace_half)]
    
    results = zeros((length(FermiArmSouthWest), num_kspace_half + 1))
    k = 1
    for (p1, p2) in zip(FermiArmSouthWest, FermiArmNorthEast)
        results[k,:] = kondoJArray[p1,p2,:]
        k += 1
    end
    return results
end


function getInStateScattFlow(kondoJArray::Array{Float64, 3}, num_kspace_half::Int64)
    # extract the in-place scattering Rg flow (essentially, 
    # the RG flow of the couplings {J(k,k), k in FS}.
    
    num_kspace = 2 * num_kspace_half + 1
    FermiArmSouthWest = [num_kspace_half + 1 + j * (num_kspace - 1) for j in 0:(num_kspace_half)]    
    results = zeros((length(FermiArmSouthWest), num_kspace_half + 1))
    for (k, point) in enumerate(FermiArmSouthWest)
        results[k,:] = kondoJArray[point,point,:]
    end
    return results
end