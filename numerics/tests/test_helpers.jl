using Test
include("../source/helpers.jl")

@testset "Mapping 1D To 2D" begin
    @testset for size_BZ in SIZE_BZ
        points_subset = POINTS(size_BZ)
        @testset for (test, testPoint) in enumerate(points_subset)
            @test map1DTo2D(testPoint, size_BZ) ≈ [KX_VALUES[test], KY_VALUES[test]]
        end
        @test map1DTo2D(points_subset, size_BZ) ≈ [KX_VALUES, KY_VALUES]
    end
end

@testset "Mapping 2D To 1D" begin
    @testset for size_BZ in SIZE_BZ
        points_subset = POINTS(size_BZ)
        @testset for test in eachindex(KX_VALUES)
            @test map2DTo1D(KX_VALUES[test], KY_VALUES[test], size_BZ) ≈ points_subset[test]
        end
        @test map2DTo1D(KX_VALUES, KY_VALUES, size_BZ) ≈ points_subset
    end
end


@testset "Tight-binding Dispersion" begin
    @test tightBindDisp(KX_VALUES, KY_VALUES) == [tightBindDisp(kx, ky) for (kx, ky) in zip(KX_VALUES, KY_VALUES)]
    dOfStates, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    @test dispersion[FS_POINTS] ≈ PROBE_ENERGIES[1] * ones(length(FS_POINTS)) atol = 1e-10
    @test dispersion[INSIDE_POINTS] ≈ PROBE_ENERGIES[2] * ones(length(INSIDE_POINTS)) atol = 1e-10
    @test dispersion[OUTSIDE_POINTS] ≈ PROBE_ENERGIES[3] * ones(length(OUTSIDE_POINTS)) atol = 1e-10
    @test dispersion[CENTER_POINTS] ≈ PROBE_ENERGIES[4] * ones(length(CENTER_POINTS)) atol = 1e-10
    @test dispersion[CORNER_POINTS] ≈ PROBE_ENERGIES[5] * ones(length(CORNER_POINTS)) atol = 1e-10
end


@testset "Isoenergetic contours" begin
    dos, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    @test getIsoEngCont(dispersion, PROBE_ENERGIES[1]) ≈ FS_POINTS atol = 1e-10
    @test getIsoEngCont(dispersion, PROBE_ENERGIES[2]) ≈ INSIDE_POINTS atol = 1e-10
    @test getIsoEngCont(dispersion, PROBE_ENERGIES[3]) ≈ OUTSIDE_POINTS atol = 1e-10
    @test getIsoEngCont(dispersion, PROBE_ENERGIES[4]) ≈ CENTER_POINTS atol = 1e-10
    @test getIsoEngCont(dispersion, PROBE_ENERGIES[5]) ≈ CORNER_POINTS atol = 1e-10
end


@testset "Particle-hole Transformation" begin
    dos, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    points_set = [1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18, 21, 22, 23, 19, 20, 24, 25, 4, 5, 9, 10, 14, 15]
    holePoints_set = [13, 14, 15, 18, 19, 20, 23, 24, 25, 8, 9, 10, 13, 14, 15, 7, 8, 12, 13, 12, 13, 17, 18, 22, 23]
    @test particleHoleTransf(points_set, SIZE_BZ[1]) == holePoints_set
end


@testset "Density of states" begin
    dOfStates, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    @test dOfStates[3] == dOfStates[11] == dOfStates[15] == dOfStates[23]
    @test dOfStates[1] == dOfStates[5] == dOfStates[21] == dOfStates[25]
    @test dOfStates[2] == dOfStates[4] == dOfStates[22] == dOfStates[24]
    @test dOfStates[6] == dOfStates[10] == dOfStates[16] == dOfStates[20]
    @test dOfStates[7] == dOfStates[17] == dOfStates[19] == dOfStates[9]
end
