using Test
using Random
include("../source/probes.jl")
include("../source/helpers.jl")


@testset "Flow of totalScatterProb" begin
    kondoJArray = rand(Float64, (SIZE_BZ[1]^2, SIZE_BZ[1]^2, 3))
    dOfStates, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    results_norm, results_bare, results_bool = scattProb(kondoJArray, 3, SIZE_BZ[1], dispersion, 0.0)
    @testset for p1 in 1:SIZE_BZ[1]^2
        results_compare = sum(kondoJArray[p1, FS_POINTS, 3] .^ 2)
        results_compare_bare = sum(kondoJArray[p1, FS_POINTS, 1] .^ 2)
        results_compare_bool = tolerantSign(results_compare ./ results_compare_bare, RG_RELEVANCE_TOL)
        if !(p1 in CORNER_POINTS || p1 in CENTER_POINTS)
            if results_compare_bool < 0
                results_compare = NaN
            end
            @test results_norm[p1] ≈ results_compare ./ results_compare_bare atol = 1e-10
            @test results_bool[p1] == results_compare_bool
        end
    end

end
