using Test
include("../source/helpers.jl")

@testset "Mapping 1D To 2D" begin
    @testset for num_kspace in SIZE_BZ
        points_subset = POINTS(num_kspace)
        @testset for (test, testPoint) in enumerate(points_subset)
            @test map1DTo2D(testPoint, num_kspace) ≈ [KX_VALUES[test], KY_VALUES[test]]
        end
        @test map1DTo2D(points_subset, num_kspace) ≈ [KX_VALUES, KY_VALUES]
    end
end

@testset "Mapping 2D To 1D" begin
    @testset for num_kspace in SIZE_BZ
        points_subset = POINTS(num_kspace)
        @testset for test in eachindex(KX_VALUES)
            @test map2DTo1D(KX_VALUES[test], KY_VALUES[test], num_kspace) ≈ points_subset[test]
        end
        @test map2DTo1D(KX_VALUES, KY_VALUES, num_kspace) ≈ points_subset
    end
end


FS_points = [3, 7, 9, 11, 15, 17, 19, 23]
FS_points_left = [3, 7, 11, 17, 23]
center_points = [13]
corner_points = [1, 5, 21, 25]
outside_points = [2, 4, 6, 10, 16, 20, 22, 24]
inside_points = [8, 12, 14, 18]
probe_energies = [0, -2 * HOP_T, 2 * HOP_T, -4 * HOP_T, 4 * HOP_T]
@testset "Tight-binding Dispersion" begin
    @test tightBindDisp(KX_VALUES, KY_VALUES) == [tightBindDisp(kx, ky) for (kx, ky) in zip(KX_VALUES, KY_VALUES)]
    dos, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    @test dispersion[FS_points] ≈ probe_energies[1] * ones(length(FS_points)) atol = 1e-10
    @test dispersion[inside_points] ≈ probe_energies[2] * ones(length(inside_points)) atol = 1e-10
    @test dispersion[outside_points] ≈ probe_energies[3] * ones(length(outside_points)) atol = 1e-10
    @test dispersion[center_points] ≈ probe_energies[4] * ones(length(center_points)) atol = 1e-10
    @test dispersion[corner_points] ≈ probe_energies[5] * ones(length(corner_points)) atol = 1e-10
end


@testset "Isoenergetic contours" begin
    dos, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    @test getIsoEngCont(dispersion, probe_energies[1]) ≈ FS_points atol = 1e-10
    @test getIsoEngCont(dispersion, probe_energies[2]) ≈ inside_points atol = 1e-10
    @test getIsoEngCont(dispersion, probe_energies[3]) ≈ outside_points atol = 1e-10
    @test getIsoEngCont(dispersion, probe_energies[4]) ≈ center_points atol = 1e-10
    @test getIsoEngCont(dispersion, probe_energies[5]) ≈ corner_points atol = 1e-10
end


@testset "Particle-hole Transformation" begin
    dos, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    points_set = [1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18, 21, 22, 23, 19, 20, 24, 25, 4, 5, 9, 10, 14, 15]
    holePoints_set = [13, 14, 15, 18, 19, 20, 23, 24, 25, 8, 9, 10, 13, 14, 15, 7, 8, 12, 13, 12, 13, 17, 18, 22, 23]
    @test particleHoleTransf(points_set, SIZE_BZ[1]) == holePoints_set
end


@testset "High-energy low-energy separation" begin
    dos, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    proceed_flags = fill(1, SIZE_BZ[1]^2, SIZE_BZ[1]^2)
    innerIndicesArr, excludedVertexPairs, mixedVertexPairs, cutoffPoints, cutoffHolePoints, proceed_flags = highLowSeparation(dispersion, abs(probe_energies[3]), proceed_flags, SIZE_BZ[1])
    @test sort(cutoffPoints) == sort(outside_points)
    @test unique(sort(cutoffHolePoints)) == sort(inside_points)
    @test sort(innerIndicesArr) == sort(FS_points_left)
    @test sort(excludedVertexPairs) == [(p1, p2) for p1 in [9, 15, 19] for p2 in [9, 15, 19] if p2 >= p1]
    @test proceed_flags[outside_points] == 0 .* outside_points
end
