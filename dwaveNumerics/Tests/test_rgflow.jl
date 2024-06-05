using Test
include("../source/rgFlow.jl")

@testset "High-energy low-energy separation" begin
    dos, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    innerIndicesArr, excludedVertexPairs, mixedVertexPairs, cutoffPoints, cutoffHolePoints, proceed_flags_updated = highLowSeparation(dispersion, abs(PROBE_ENERGIES[3]), PROCEED_FLAGS, SIZE_BZ[1])
    @test sort(cutoffPoints) == sort(OUTSIDE_POINTS)
    @test unique(sort(cutoffHolePoints)) == sort(INSIDE_POINTS)
    @test sort(innerIndicesArr) == sort(FS_POINTS_LEFT)
    @test sort(excludedVertexPairs) == [(p1, p2) for p1 in [9, 15, 19] for p2 in [9, 15, 19] if p2 >= p1]
    @test proceed_flags_updated[OUTSIDE_POINTS] == 0 .* OUTSIDE_POINTS

    innerIndicesArr, excludedVertexPairs, mixedVertexPairs, cutoffPoints, cutoffHolePoints, proceed_flags_updated = highLowSeparation(dispersion, abs(PROBE_ENERGIES[5]), PROCEED_FLAGS, SIZE_BZ[1])
    @test sort(cutoffPoints) == sort(CORNER_POINTS)
    @test unique(sort(cutoffHolePoints)) == sort(CENTER_POINTS)
    @test sort(innerIndicesArr) == sort(INNER_INDICES_ONE)
    @test sort(unique([p1 for (p1, p2) in excludedVertexPairs])) == sort(EXCLUDED_INDICES_ONE)
    @test proceed_flags_updated[OUTSIDE_POINTS] == 0 .* OUTSIDE_POINTS
end

@testset "Getting cutoff energies" begin
    cutoffEnergies = getCutOffEnergy(SIZE_BZ[1])
    for (i, E) in enumerate(cutoffEnergies)
        @test E ≈ [4, 2, 0][i] atol = TOLERANCE
    end
end


function kondoArraySymmetriesCheck(kondoJArray, orbital, size_BZ)
    if orbital == "p"
        # check that the diagonal elements J(k,k) are all unity for the p-wave case
        @test all(diag(kondoJArray[:, :]) .== 1)
    else
        # check that J(k_1,k_2) is zero if k1 and k2 are on the 
        # kx=ky lines, for the d-wave case
        kx_equals_ky = [1:size_BZ+1:size_BZ^2;
            size_BZ^2-size_BZ+1:size_BZ-1:size_BZ]
        @testset for (p1, p2) in Iterators.product(kx_equals_ky, kx_equals_ky)
            @test kondoJArray[p1, p2] ≈ 0 atol = TOLERANCE
        end
    end

    # J(k,q_x=0,q_y) == J(k,q_x=2π,q_y)
    @test kondoJArray[1:size_BZ:size_BZ^2-size_BZ+1, :] ≈ kondoJArray[size_BZ:size_BZ:size_BZ^2, :] atol = TOLERANCE

    # J(k,q_y=0,q_x) == J(k,q_y=2π,q_x)
    @test kondoJArray[1:1:size_BZ, :] ≈ kondoJArray[size_BZ^2-size_BZ+1:1:size_BZ^2, :] atol = TOLERANCE

    # J(k,q) == -J(k,q+π)
    @testset for p1 in 1:size_BZ^2
        for p2 in 1:size_BZ^2
            @test kondoJArray[p1, p2] ≈ -kondoJArray[p1, particleHoleTransf(p2, size_BZ)] atol = TOLERANCE
        end
    end

    # J(k,q) == J(q,k)
    @testset for p1 in eachindex(kondoJArray[1, :])
        for p2 in eachindex(kondoJArray[:, 1])
            @test kondoJArray[p1, p2] == kondoJArray[p2, p1]
        end
    end

    # J(k,k+π) == -1 for p-wave and J(k,k+π) == -J(k,k)
    @testset for p1 in rand(1:size_BZ^2, 10)
        p2 = particleHoleTransf(p1, size_BZ)
        if orbital == "p"
            @test kondoJArray[p1, p2] ≈ -1 atol = TOLERANCE
        else
            @test kondoJArray[p1, p2] ≈ -kondoJArray[p1, p1] atol = TOLERANCE
        end
    end

    # quantitative checks for the values of J(k,q)
    @testset for p1 in eachindex(kondoJArray[:, 1])
        kx, ky = map1DTo2D(p1, size_BZ)
        for p2 in 1:size_BZ^2
            qx, qy = map1DTo2D(p2, size_BZ)
            if orbital == "p"
                @test kondoJArray[p1, p2] ≈ 0.5 * (cos(kx - qx) + cos(ky - qy)) atol = TOLERANCE
            else
                @test kondoJArray[p1, p2] ≈ 0.5 * (cos(kx - qx) - cos(ky - qy)) atol = TOLERANCE
            end
        end
    end
end


@testset "Initialise Kondo coupling matrix" begin
    size_BZ = 9
    J_val = 0.1
    @testset for orbital in ["p", "d", "poff", "doff"]
        kondoJArray = initialiseKondoJ(size_BZ, orbital, 1, J_val)[:, :, 1]
        for (k_index1, k_index2) in Iterators.product(1:size_BZ^2, 1:size_BZ^2)
            kx1, ky1 = map1DTo2D(k_index1, size_BZ)
            kx2, ky2 = map1DTo2D(k_index2, size_BZ)
            if orbital == "p"
                @test kondoJArray[k_index1, k_index2] ≈ 0.5 * J_val * (cos(kx1 - kx2) + cos(ky1 - ky2)) atol = TOLERANCE
            elseif orbital == "d"
                @test kondoJArray[k_index1, k_index2] ≈ 0.5 * J_val * (cos(kx1 - kx2) - cos(ky1 - ky2)) atol = TOLERANCE
            elseif orbital == "poff"
                @test kondoJArray[k_index1, k_index2] ≈ J_val * (cos(kx1) + cos(ky1)) * (cos(kx2) + cos(ky2)) atol = TOLERANCE
            elseif orbital == "doff"
                @test kondoJArray[k_index1, k_index2] ≈ J_val * (cos(kx1) - cos(ky1)) * (cos(kx2) - cos(ky2)) atol = TOLERANCE
            end
        end
    end
end


begin
    size_BZ = 5
    W_val = 0.1
    Threads.@threads for orbital in ["p", "d", "poff", "doff"]
        @testset "Bath interaction function: $orbital" begin
        for points in collect(Iterators.product(1:size_BZ^2, 1:size_BZ^2, 1:size_BZ^2, 1:size_BZ^2))
            kx_arr, ky_arr = map1DTo2D(collect(points), size_BZ)
            bathIntValue = bathIntForm(W_val, orbital, size_BZ, points)
            if orbital == "p"
                @test bathIntValue ≈ 0.5 * W_val * (cos(kx_arr[1] - kx_arr[2] + kx_arr[3] - kx_arr[4]) + cos(ky_arr[1] - ky_arr[2] + ky_arr[3] - ky_arr[4])) atol = TOLERANCE
            elseif orbital == "d"
                @test bathIntValue ≈ 0.5 * W_val * (cos(kx_arr[1] - kx_arr[2] + kx_arr[3] - kx_arr[4]) - cos(ky_arr[1] - ky_arr[2] + ky_arr[3] - ky_arr[4])) atol = TOLERANCE
            elseif orbital == "poff"
                @test bathIntValue ≈ W_val * prod([cos(kx) + cos(ky) for (kx, ky) in zip(kx_arr, ky_arr)]) atol = TOLERANCE
            elseif orbital == "doff"
                @test bathIntValue ≈ W_val * prod([cos(kx) - cos(ky) for (kx, ky) in zip(kx_arr, ky_arr)]) atol = TOLERANCE
            end
        end
        end
    end
end


@testset "RG flow" begin
    J_val = 0.1
    size_BZ = 5
    centerPoint = trunc(Int, 0.5 * (size_BZ^2 + 1))
    omega_by_t = -2.0
    cutoffPoints1 = [1, size_BZ, size_BZ^2 - size_BZ + 1, size_BZ^2]
    cutoffPoints1Particle = [centerPoint]
    cutoffPoints1Bar = particleHoleTransf(cutoffPoints1, size_BZ)
    cutoffPoints2 = [2, size_BZ-1, size_BZ+1, 2 * size_BZ, size_BZ * (size_BZ - 2) + 1, size_BZ^2-size_BZ, size_BZ^2-size_BZ+2, size_BZ^2-1]
    cutoffPoints2Particle = [centerPoint-1, centerPoint+1, centerPoint-size_BZ, centerPoint+size_BZ]
    cutoffPoints2Bar = particleHoleTransf(cutoffPoints2, size_BZ)
    innerPoints1 = [p for p ∈ 1:size_BZ^2 if p ∉ cutoffPoints1 && p ∉ cutoffPoints1Particle]
    innerPoints2 = [p for p ∈ innerPoints1 if p ∉ cutoffPoints2 && p ∉ cutoffPoints2Particle]
    dos, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
    deltaEnergy1 = dispersion[1] - dispersion[2]
    deltaEnergy2 = dispersion[2] - dispersion[3]
    for orbitals in [("p", "p"), ("d", "d")]
        for W_val in [0, -0.1 * J_val, -J_val]
            kondoJArray, _ = momentumSpaceRG(size_BZ, omega_by_t, J_val, W_val, orbitals)
            denominators1 = [omega_by_t * HOP_T - abs(dispersion[q]) / 2 + kondoJArray[q,q,1] / 4 + bathIntForm(W_val, orbitals[1], size_BZ, (q,q,q,q)) / 2 for q in cutoffPoints1]
            for (p1, p2) in Iterators.product(innerPoints1, innerPoints1)
                numerators_JJ = [kondoJArray[p2,q,1] .* kondoJArray[q,p1,1] for q in cutoffPoints1]
                numerators_JW = [4 * kondoJArray[q,qBar,1] * bathIntForm(W_val, orbitals[1], size_BZ, (qBar, p2 , p1 , q)) for (q, qBar) in zip(cutoffPoints1, cutoffPoints1Bar)]
                @test kondoJArray[p1, p2, 2] ≈ kondoJArray[p1, p2, 1] - deltaEnergy1 * sum((numerators_JJ .+ numerators_JW) .* dos[cutoffPoints1] ./ denominators1) atol=TOLERANCE
            end
            for (q1, q2) in Iterators.product(cutoffPoints1, cutoffPoints1)
                @test kondoJArray[q1, q2, 2] == kondoJArray[q1, q2, 1]
            end
            continue
            denominators2 = [omega_by_t * HOP_T - abs(dispersion[q]) / 2 + kondoJArray[q,q,2] / 4 + bathIntForm(W_val, orbitals[1], size_BZ, (q,q,q,q)) / 2 for q in cutoffPoints2]
            for (p1, p2) in Iterators.product(innerPoints2, innerPoints2)
                numerators_JJ = [kondoJArray[p2,q,2] .* kondoJArray[q,p1,2] for q in cutoffPoints2]
                numerators_JW = [4 * kondoJArray[q,qBar,2] * bathIntForm(W_val, orbitals[1], size_BZ, (qBar, p2 , p1 , q)) for (q, qBar) in zip(cutoffPoints2, cutoffPoints2Bar)]
                @test kondoJArray[p1, p2, 3] ≈ kondoJArray[p1, p2, 2] - deltaEnergy2 * sum((numerators_JJ .+ numerators_JW) .* dos[cutoffPoints2] ./ denominators2) atol=TOLERANCE
            end
            for (q1, q2) in Iterators.product(cutoffPoints2, cutoffPoints2)
                @test kondoJArray[q1, q2, 3] == kondoJArray[q1, q2, 2]
            end
        end
    end
end
