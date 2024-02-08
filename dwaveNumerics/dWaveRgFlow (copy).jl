using LinearAlgebra


function mapTo2D(points, kx_arr, ky_arr)
    kx = (points .- 1) .% num_kspace .+ 1
    ky = div.((points .- 1), num_kspace) .+ 1
    return kx_arr[kx], ky_arr[ky]
end


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
    return-2*t.*(cos.(kx_arr_flattened) + cos.(ky_arr_flattened))
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
    return densityOfStates
end


function getIsoEnergeticContour(dispersionArray, num_kspace, energy)
    contourPoints = [[] for j in 1:num_kspace]
    Threads.@threads for j in 0:num_kspace-1
        energyDiffArr = dispersionArray[j * num_kspace + 1: (j + 1) * num_kspace] .- energy
        for i in 1:num_kspace
            if energyDiffArr[i] * energyDiffArr[i % num_kspace + 1] <= 0
                push!(contourPoints[j+1], j * num_kspace + i)
            end
        end
    end
    return collect(Iterators.flatten(contourPoints))
end



function main(num_kspace_half, t, J_init, W_val)
    num_kspace = 2 * num_kspace_half + 1
    kx_arr = range(-pi, stop=pi, length=num_kspace)
    ky_arr = copy(kx_arr)
    k_arr_pairs = [collect(pair) for pair in Iterators.product(kx_arr, ky_arr)]
    dispersionArray = dispersionFlattened(t, kx_arr, ky_arr)
    densityOfStates = getDOS(num_kspace, dispersionArray)

    deltaD = maximum(dispersionArray) / (num_kspace - 1)
    dwave_karr = - repeat(cos.(ky_arr), inner=num_kspace) + repeat(cos.(kx_arr), num_kspace)
    pwave_karr = repeat(cos.(ky_arr), inner=num_kspace) + repeat(cos.(kx_arr), num_kspace)
    kondoJArray = Array{Float64}(undef, num_kspace^2, num_kspace^2, num_kspace_half+1)
    kondoJArray[:,:,1] .= J_init .* (dwave_karr * dwave_karr')
    bathInt = W_val * (pwave_karr * pwave_karr') .* reshape(pwave_karr, 1, 1, :)

    flags = fill(1, num_kspace^2, num_kspace^2)
    energyCutoff = maximum(dispersionArray) + deltaD
    for stepIndex in 1:num_kspace_half
        energyCutoff -= deltaD
        kondoJArray[:,:,stepIndex+1] = kondoJArray[:,:,stepIndex]
        cutoffPoints = getIsoEnergeticContour(dispersionArray, num_kspace, energyCutoff)

        flags[cutoffPoints,:] .= 0
        flags[:,cutoffPoints] .= 0
        if all(==(0), flags)
            kondoJArray[:,:,stepIndex+1:end] .= kondoJArray[:,:,stepIndex]
            break
        end
        innerIndicesArr = collect(1:length(dispersionArray))[abs.(dispersionArray) .< abs(energyCutoff)]
        omega = -abs(energyCutoff) / 2
        innerEnergiesArr = dispersionArray[abs.(dispersionArray) .< abs(energyCutoff)]
        denominators = Diagonal([omega - energyCutoff / 2 + kondoJArray[qpoint, qpoint, stepIndex] / 4 + bathInt[qpoint, qpoint, qpoint] / 2 for qpoint in cutoffPoints])
        Threads.@threads for innerIndex1 in innerIndicesArr
            for innerIndex2 in innerIndicesArr
                for qpoint in cutoffPoints
                    denominator = omega - energyCutoff / 2 + kondoJArray[qpoint, qpoint, stepIndex] / 4 + bathInt[qpoint, qpoint, qpoint] / 2
                    kondoJArray[innerIndex1,innerIndex2,stepIndex+1] += -deltaD * densityOfStates[qpoint] * (kondoJArray[innerIndex1,qpoint,stepIndex] * kondoJArray[qpoint,innerIndex2,stepIndex]
                    - 4 * kondoJArray[qpoint, qpoint, stepIndex] * bathInt[innerIndex1,innerIndex2,qpoint]) / denominator
                end
                if kondoJArray[innerIndex1,innerIndex2,stepIndex+1] * kondoJArray[innerIndex1,innerIndex2,stepIndex] < 0
                    kondoJArray[innerIndex1,innerIndex2,stepIndex+1] = kondoJArray[innerIndex1,innerIndex2,stepIndex]
                    flags[innerIndex1,innerIndex2] = 0
                end
            end
        end
    end
    return kondoJArray, dispersionArray
end
