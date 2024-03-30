using Test
include("../source/helpers.jl")
include("../source/constants.jl")

const SIZE_BZ = [3, 101, 1001]
const KX_VALUES = [K_MIN, K_MAX, K_MIN, K_MAX, 0]
const KY_VALUES = [K_MIN, K_MIN, K_MAX, K_MAX, 0]

POINTS(num_kspace) = [
    1,
    num_kspace,
    num_kspace^2 - num_kspace + 1,
    num_kspace^2,
    Int(0.5 * num_kspace^2 + 0.5),
]

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


@testset "Tight-binding Dispersion" begin
    @test tightBindDisp(KX_VALUES, KY_VALUES) == [tightBindDisp(kx, ky) for (kx, ky) in zip(KX_VALUES, KY_VALUES)]
end

