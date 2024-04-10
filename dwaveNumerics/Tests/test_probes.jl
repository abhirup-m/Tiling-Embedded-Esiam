using Test
using Random
include("../source/probes.jl")
include("../source/helpers.jl")


@testset "Flow of totalScatterProb" begin
    kondoJArray = rand(Float64, (SIZE_BZ[1]^2, SIZE_BZ[1]^2, 3))
    dOfStates, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    results_norm, results_bare, results_bool = scattProb(kondoJArray, 3, SIZE_BZ[1], dispersion, 0.0)
    @testset for p1 in 1:SIZE_BZ[1]^2
        if p1 in CORNER_POINTS || p1 in CENTER_POINTS
            @test results_norm[p1] == 0
            @test results_bool[p1] == -1
            continue
        end
        @test results_bare[p1] ≈ sum([kondoJArray[p1, p2, 1]^2 for p2 in FS_POINTS]) atol = 1e-10
        @test results_norm[p1] ≈ sum([kondoJArray[p1, p2, 3]^2 for p2 in FS_POINTS]) atol = 1e-10
        @test results_bool[p1] == tolerantSign(sum([kondoJArray[p1, p2, 3]^2 for p2 in FS_POINTS]) / sum([kondoJArray[p1, p2, 1]^2 for p2 in FS_POINTS]), RG_RELEVANCE_TOL)
    end

end
