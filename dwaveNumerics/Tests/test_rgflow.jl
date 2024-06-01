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
    cutOffEnergies = getCutOffEnergy(SIZE_BZ[1])
    for (i, E) in enumerate(cutOffEnergies)
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

    @testset for orbital in ["p", "d"]
        kondoJArray = initialiseKondoJ(SIZE_BZ[1], orbital, 1, 1.0)[:, :, 1]
        kondoArraySymmetriesCheck(kondoJArray, orbital, SIZE_BZ[1])
    end
end


@testset "Bath interaction function" begin
    p1_range = rand(1:SIZE_BZ[1]^2, 10)
    p2_range = rand(1:SIZE_BZ[1]^2, 10)
    @testset for (p1, p2) in zip(p1_range, p2_range)
        @test bathIntForm(1.0, "p", SIZE_BZ[1], [p1, p1, p2, p2]) == 1
    end
    dos, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    _, _, _, cutoffPoints, cutoffHolePoints, _ = highLowSeparation(dispersion, PROBE_ENERGIES[3], PROCEED_FLAGS, SIZE_BZ[1])
    p1_range = rand(1:SIZE_BZ[1]^2, length(cutoffPoints))
    p2_range = rand(1:SIZE_BZ[1]^2, length(cutoffPoints))
    @testset for (p1, p2, q, q_bar) in zip(p1_range, p2_range, cutoffPoints, cutoffHolePoints)
        @test bathIntForm(1.0, "p", SIZE_BZ[1], [q_bar, p1, p2, q]) ≈ -bathIntForm(1.0, "p", SIZE_BZ[1], [q, p1, p2, q]) atol = TOLERANCE
    end
    bathIntMatrix = zeros((SIZE_BZ[1]^2, SIZE_BZ[1]^2))
    for p1 in 1:SIZE_BZ[1]^2
        for p2 in 1:SIZE_BZ[1]^2
            bathIntMatrix[p1, p2] = bathIntForm(1.0, "p", SIZE_BZ[1], [1, p1, p2, 1])
        end
    end
    @test all(diag(bathIntMatrix[:, :]) .== 1)
    @test bathIntMatrix[[1, 6, 11, 16, 21], :] ≈ bathIntMatrix[[5, 10, 15, 20, 25], :] atol = TOLERANCE
    @test bathIntMatrix[[1, 2, 3, 4, 5], :] ≈ bathIntMatrix[[21, 22, 23, 24, 25], :] atol = TOLERANCE
    @test bathIntMatrix[[2, 7, 12, 17, 22], :] ≈ -bathIntMatrix[[14, 19, 24, 9, 14], :] atol = TOLERANCE
    @test bathIntMatrix[[6, 7, 8, 9, 10], :] ≈ -bathIntMatrix[[18, 19, 20, 17, 18], :] atol = TOLERANCE
    @test bathIntMatrix[[1, 2, 3, 4, 5], :] ≈ bathIntMatrix[[21, 22, 23, 24, 25], :] atol = TOLERANCE
    @testset for p1 in eachindex(bathIntMatrix[1, :])
        for p2 in eachindex(bathIntMatrix[:, 1])
            @test bathIntMatrix[p1, p2] == bathIntMatrix[p2, p1]
        end
    end
    @testset for p2 in eachindex(bathIntMatrix[:, 1])
        kx, ky = map1DTo2D(p2, SIZE_BZ[1])
        @test bathIntMatrix[1, p2] ≈ -0.5 * (cos(kx) + cos(ky)) atol = TOLERANCE
        @test bathIntMatrix[2, p2] ≈ -0.5 * (sin(kx) + cos(ky)) atol = TOLERANCE
        @test bathIntMatrix[3, p2] ≈ 0.5 * (cos(kx) - cos(ky)) atol = TOLERANCE
        @test bathIntMatrix[6, p2] ≈ -0.5 * (cos(kx) + sin(ky)) atol = TOLERANCE
        @test bathIntMatrix[7, p2] ≈ -0.5 * (sin(kx) + sin(ky)) atol = TOLERANCE
        @test bathIntMatrix[8, p2] ≈ 0.5 * (cos(kx) - sin(ky)) atol = TOLERANCE
        @test bathIntMatrix[11, p2] ≈ 0.5 * (-cos(kx) + cos(ky)) atol = TOLERANCE
        @test bathIntMatrix[12, p2] ≈ 0.5 * (-sin(kx) + cos(ky)) atol = TOLERANCE
        @test bathIntMatrix[13, p2] ≈ 0.5 * (cos(kx) + cos(ky)) atol = TOLERANCE
    end
end


@testset "DeltaJ(k1,k2)" begin
    dos, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    _, _, _, cutoffPoints, cutoffHolePoints, _ = highLowSeparation(dispersion, PROBE_ENERGIES[3], PROCEED_FLAGS, SIZE_BZ[1])
    denominators = -1 .* [1.5, 3, 1, 1, 2, 4, 1, 1]
    proceed_flag_k1k2 = 1
    Jk1k2Prev = 0.1
    Jk2q_qk1 = [1, 1.5, 2, 0, -2, 1, 2, 0]
    J_qqbar = -1 * ones(size(cutoffPoints))
    dos_qq = 0.5 * ones(size(cutoffPoints))
    args = [-0.25, "p", SIZE_BZ[1], [cutoffPoints, 1, 2, cutoffHolePoints]]
    kondoJArrayNext_k1k2, proceed_flags_updated = deltaJk1k2(denominators, proceed_flag_k1k2, Jk1k2Prev, Jk2q_qk1, J_qqbar, DELTA_ENERGY, args, dos_qq)
    @test kondoJArrayNext_k1k2 ≈ 0.1 + DELTA_ENERGY * (1 / 3 + 3.75 / 2 - 23 / 16) atol = TOLERANCE
    @test proceed_flags_updated == 1
    args = [-1.0, "p", SIZE_BZ[1], [cutoffPoints, 1, 2, cutoffHolePoints]]
    kondoJArrayNext_k1k2, proceed_flags_updated = deltaJk1k2(denominators, proceed_flag_k1k2, Jk1k2Prev, Jk2q_qk1, J_qqbar, DELTA_ENERGY, args, dos_qq)
    @test kondoJArrayNext_k1k2 ≈ 0 atol = TOLERANCE
    @test proceed_flags_updated == 0
end


@testset "Symmetrise RG flow" begin
    dos, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    kondoJArrayPrev = initialiseKondoJ(SIZE_BZ[1], "p", 3, 0.1)[:, :, 1]
    @testset for E in [PROBE_ENERGIES[1], PROBE_ENERGIES[3]]
        innerIndicesArr, excludedVertexPairs, mixedVertexPairs, cutoffPoints, cutoffHolePoints, proceed_flags = highLowSeparation(dispersion, E, PROCEED_FLAGS, SIZE_BZ[1])
        kondoJArrayNext = copy(kondoJArrayPrev)
        kondoJArrayNext[innerIndicesArr] = kondoJArrayNext[innerIndicesArr] .* 2

        externalVertexPairs = [
            (p1, p2) for p1 in sort(innerIndicesArr) for
            p2 in sort(innerIndicesArr)[sort(innerIndicesArr).>=p1]
        ]
        kondoJArrayNext_updated, proceed_flags = symmetriseRGFlow(innerIndicesArr, excludedVertexPairs, mixedVertexPairs, SIZE_BZ[1], kondoJArrayNext, kondoJArrayPrev, proceed_flags)
        @testset for p1 in 1:SIZE_BZ[1]^2
            for p2 in 1:SIZE_BZ[1]^2
                @test kondoJArrayNext_updated[p1, p2] ≈ kondoJArrayNext_updated[p2, p1] atol = TOLERANCE
            end
        end
        holePoints = [4, 5, 9, 10, 14, 15, 19, 20]
        particlePoints = [12, 13, 17, 18, 22, 23, 7, 8, 12, 13]

        @testset for p1 in innerIndicesArr
            for (p2, p3) in zip(holePoints, particlePoints)
                @test kondoJArrayNext_updated[p1, p2] ≈ -kondoJArrayNext_updated[p1, p3] atol = TOLERANCE
                @test proceed_flags[p1, p2] ≈ proceed_flags[p1, p3] atol = TOLERANCE
            end
        end
        @testset for ((p1, p3), (p2, p4)) in Iterators.product(zip(holePoints, particlePoints), zip(holePoints, particlePoints))
            @test kondoJArrayNext_updated[p1, p2] ≈ kondoJArrayNext_updated[p3, p4] atol = TOLERANCE
            @test proceed_flags[p1, p2] ≈ proceed_flags[p3, p4] atol = TOLERANCE
        end
    end
end


@testset "Symmetrise RG flow" begin
    dos, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    kondoJArrayPrev = initialiseKondoJ(SIZE_BZ[1], "p", 3, 0.1)[:, :, 1]
    @testset for E in [PROBE_ENERGIES[1], PROBE_ENERGIES[3]]
        innerIndicesArr, excludedVertexPairs, mixedVertexPairs, cutoffPoints, cutoffHolePoints, proceed_flags = highLowSeparation(dispersion, E, PROCEED_FLAGS, SIZE_BZ[1])
        kondoJArrayNext = copy(kondoJArrayPrev)
        kondoJArrayNext[innerIndicesArr] = kondoJArrayNext[innerIndicesArr] .* 2

        externalVertexPairs = [
            (p1, p2) for p1 in sort(innerIndicesArr) for
            p2 in sort(innerIndicesArr)[sort(innerIndicesArr).>=p1]
        ]
        kondoJArrayNext_updated, proceed_flags = symmetriseRGFlow(innerIndicesArr, excludedVertexPairs, mixedVertexPairs, SIZE_BZ[1], kondoJArrayNext, kondoJArrayPrev, proceed_flags)
        @testset for p1 in 1:SIZE_BZ[1]^2
            for p2 in 1:SIZE_BZ[1]^2
                @test kondoJArrayNext_updated[p1, p2] ≈ kondoJArrayNext_updated[p2, p1] atol = TOLERANCE
            end
        end
        holePoints = [4, 5, 9, 10, 14, 15, 19, 20]
        particlePoints = [12, 13, 17, 18, 22, 23, 7, 8, 12, 13]

        @testset for p1 in innerIndicesArr
            for (p2, p3) in zip(holePoints, particlePoints)
                @test kondoJArrayNext_updated[p1, p2] ≈ -kondoJArrayNext_updated[p1, p3] atol = TOLERANCE
                @test proceed_flags[p1, p2] ≈ proceed_flags[p1, p3] atol = TOLERANCE
            end
        end
        @testset for ((p1, p3), (p2, p4)) in Iterators.product(zip(holePoints, particlePoints), zip(holePoints, particlePoints))
            @test kondoJArrayNext_updated[p1, p2] ≈ kondoJArrayNext_updated[p3, p4] atol = TOLERANCE
            @test proceed_flags[p1, p2] ≈ proceed_flags[p3, p4] atol = TOLERANCE
        end
    end
end


@testset "RG flow" begin
    J_val = 0.1
    size_BZ = 5
    for W_val in [0.0, -J_val / 2, -1 * J_val]
        kondoJArray, dispersion = momentumSpaceRG(size_BZ, -2.0, J_val, W_val, ("p", "p"))

        @test (kondoJArray[3, 15, end] == kondoJArray[3, 11, end]
               == kondoJArray[23, 11, end] == kondoJArray[23, 15, end]
               ≠ 0)
        @test (kondoJArray[3, 9, end] == kondoJArray[9, 15, end]
               == kondoJArray[15, 19, end] == kondoJArray[19, 23, end]
               == kondoJArray[23, 17, end] == kondoJArray[17, 11, end]
               == kondoJArray[11, 7, end] == 0)
        @test (kondoJArray[3, 23, end] == kondoJArray[11, 15, end] ≠ 0)
        @test (kondoJArray[7, 19, end] == kondoJArray[9, 17, end] ≠ 0)
    end
end
