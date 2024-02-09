using LinearAlgebra


function isogeometricContour(kx_start, num_kspace)
    if kx_start > div(num_kspace, 2)
       kx_start =  num_kspace - kx_start
    end
    contourPoints = []
    kx_points_left = 1:kx_start
    kx_points_right = num_kspace + 1 .- kx_points_left
    
    ky_points_lowerleft = kx_start + 1 .- kx_points_left
    ky_points_upperleft = num_kspace - kx_start .+ kx_points_left
    ky_points_lowerright = kx_start - num_kspace .+ kx_points_right
    ky_points_upperright = 2 * num_kspace - kx_start + 1 .- kx_points_right
    
    push!(contourPoints, (kx_points_left .+ num_kspace .* (ky_points_lowerleft .- 1))...)
    push!(contourPoints, (kx_points_left .+ num_kspace .* (ky_points_upperleft .- 1))...)
    push!(contourPoints, (kx_points_right .+ num_kspace .* (ky_points_lowerright .- 1))...)
    push!(contourPoints, (kx_points_right .+ num_kspace .* (ky_points_upperright .- 1))...)

    return contourPoints
end


function dispersionFlattened(t, kx_arr, ky_arr)
    kx_arr_flattened = repeat(kx_arr, inner=length(ky_arr))
    ky_arr_flattened = repeat(ky_arr, outer=length(kx_arr))
    return -2*t.*(cos.(kx_arr_flattened) + cos.(ky_arr_flattened))
end


function getDOS(num_kspace, dispersionArray)
    delta_k = 2 * pi / num_kspace
    kspaceDos = 1 / delta_k
    densityOfStates = zeros(num_kspace * num_kspace)
    Threads.@threads for j in 0:num_kspace-1
        for i in 1:num_kspace
            E_xy = dispersionArray[j * num_kspace + i]
            E_xpy = dispersionArray[j * num_kspace + i % num_kspace + 1]
            E_xyp = dispersionArray[(j * num_kspace + num_kspace) % num_kspace^2 + i]
            dE_xydk = sqrt((E_xpy - E_xy)^2 / delta_k^2
                           + (E_xyp - E_xy)^2 / delta_k^2
                           )
            densityOfStates[j * num_kspace + i] = kspaceDos / dE_xydk
        end
    end
    replace!(densityOfStates, Inf=>maximum(densityOfStates[densityOfStates .≠ Inf]))
	normalisation = sum([dos * abs(dispersionArray[i % num_kspace + 1] - dispersionArray[i]) for (i, dos) in enumerate(densityOfStates)]) / num_kspace^2
    return densityOfStates / normalisation
end


function getIsoEnergeticContour(dispersionArray, num_kspace, energy)
    contourPoints = [[] for j in 1:num_kspace]
    Threads.@threads for j in 0:num_kspace-1
        energyDiffArr = dispersionArray[j * num_kspace + 1: (j + 1) * num_kspace] .- energy
        for i in 1:num_kspace
            if energyDiffArr[i] * energyDiffArr[i % num_kspace + 1] <= 0
                push!(contourPoints[j+1], j * num_kspace + i + 1)
				break
            end
        end
    end
    return collect(Iterators.flatten(contourPoints))
end


function main(num_kspace_half, t, J_init, W_val)
    @assert num_kspace_half % 2 == 0
    num_kspace = 2 * num_kspace_half + 1
    kx_arr = range(-pi, stop=pi, length=num_kspace)
    ky_arr = copy(kx_arr)
    dispersionArray = dispersionFlattened(t, kx_arr, ky_arr)
    densityOfStates = getDOS(num_kspace, dispersionArray)
    deltaD = maximum(dispersionArray) / (num_kspace - 1)
    dwave_karr = - repeat(cos.(ky_arr), inner=num_kspace) + repeat(cos.(kx_arr), num_kspace)
    pwave_karr = repeat(cos.(ky_arr), inner=num_kspace) + repeat(cos.(kx_arr), num_kspace)
    kondoJArray = Array{Float64}(undef, num_kspace^2, num_kspace^2, num_kspace_half+1)
    kondoJArray[:,:,1] .= J_init .* (dwave_karr * dwave_karr')
	bathInt(i,j,k) = W_val * (-0.5/t) * dispersionArray[i] * dispersionArray[j] * dispersionArray[k]^2 
	quadrantIndices = [kx_ind + ky_ind * num_kspace for (ky_ind, kx_ind) in Iterators.product(0:num_kspace-1, 1:num_kspace) if kx_ind <= num_kspace_half + 1 && ky_ind <= num_kspace_half]

    flags = fill(1, num_kspace^2, num_kspace^2)
    energyCutoff = maximum(dispersionArray) + deltaD
    denominatorSign = -1
    @showprogress for stepIndex in 1:num_kspace_half
        energyCutoff -= deltaD
		kondoJArray[:,:,stepIndex+1:end] .= kondoJArray[:,:,stepIndex]
		cutoffPoints = [qpoint for qpoint in getIsoEnergeticContour(dispersionArray, num_kspace, energyCutoff) if qpoint in quadrantIndices]

        flags[cutoffPoints,:] .= 0
        flags[:,cutoffPoints] .= 0
        all(==(0), flags) ? break : nothing

		innerIndicesArr = [point for (point, energy) in enumerate(dispersionArray) if abs(energy) < abs(energyCutoff) && point in quadrantIndices]
        omega = -abs(energyCutoff) / 2
		Threads.@threads for innerIndex1 in innerIndicesArr
		    for innerIndex2 in innerIndicesArr
				denominators = [omega - abs(energyCutoff) / 2 + kondoJArray[qpoint, qpoint, stepIndex] / 4 + bathInt(qpoint, qpoint, qpoint) / 2 for qpoint in cutoffPoints]
				# if ! all(<(0), denominators)
                #     flags[innerIndex1,innerIndex2] = 0
			    # end
				if flags[innerIndex1,innerIndex2] == 0
				    continue
				end
				for (qpoint, denominator) in zip(cutoffPoints, denominators)
				    if  denominator < 0
							kondoJArray[innerIndex1,innerIndex2,stepIndex+1] += -4 * deltaD * densityOfStates[qpoint] * (kondoJArray[innerIndex1, qpoint, stepIndex] * kondoJArray[innerIndex2, qpoint, stepIndex] + 4 * kondoJArray[qpoint, qpoint, stepIndex] * bathInt(innerIndex1,innerIndex2,qpoint)) / denominator
					end
                end
                if kondoJArray[innerIndex1,innerIndex2,stepIndex+1] * kondoJArray[innerIndex1,innerIndex2,stepIndex] <= 0 && stepIndex > 1
                    kondoJArray[innerIndex1,innerIndex2,stepIndex+1] = kondoJArray[innerIndex1,innerIndex2,stepIndex]
                    flags[innerIndex1,innerIndex2] = 0
                end
				telePoints1 = teleportToAllQuads(innerIndex1, num_kspace)
				telePoints2 = teleportToAllQuads(innerIndex2, num_kspace)
				for (point1, point2) in Iterators.product(telePoints1, telePoints2)
					kondoJArray[point1,point2,stepIndex+1] = kondoJArray[innerIndex1,innerIndex2,stepIndex+1]
				end
		    end
        end
    end
    return kondoJArray, dispersionArray
end

function teleportToAllQuads(point, num_kspace)
    map1DTo2D(points, num_kspace) = [(points .- 1) .% num_kspace .+ 1, div.((points .- 1), num_kspace) + 1]
    map2DTo1D(kx_index, ky_index, num_kspace) = kx_index + (ky_index - 1) * num_kspace
    kx_index_L, ky_index_D = map1DTo2D(point, num_kspace)
    kx_index_R = (num_kspace + 1) - kx_index_L
    ky_index_U = (num_kspace + 1) - ky_index_D
	telepPoints = [map2DTo1D(kx_index, ky_index, num_kspace) for (kx_index, ky_index) in 
				   [[kx_index_L, ky_index_D], [kx_index_L, ky_index_U], [kx_index_R, ky_index_D], [kx_index_R, ky_index_U]]]
	return telepPoints
end
