using Test
using Random
include("../source/probes.jl")
include("../source/helpers.jl")


@testset "Flow of totalScatterProb" begin
    kondoJArray = rand(Float64, (SIZE_BZ[1]^2, SIZE_BZ[1]^2, 3))
    filePath = randstring(15)
    results_norm, results_unnorm, results_bool = totalScatterProb(kondoJArray, SIZE_BZ[1], 3, filePath)
    results_norm, results_unnorm, results_bool = reshape.([results_norm, results_unnorm, results_bool], (SIZE_BZ[1]^2))
    @test isfile(filePath)
    @test isfile(filePath * "-bool")
    if isfile(filePath)
        rm(filePath)
    end
    if isfile(filePath * "-bool")
        rm(filePath * "-bool")
    end
    @testset for p1 in 1:SIZE_BZ[1]^2
        @test results_unnorm[p1] ≈ sum([kondoJArray[p1, p2, 3]^2 for p2 in 1:SIZE_BZ[1]^2]) atol = 1e-10
        @test results_bool[p1] == tolerantSign(sum([kondoJArray[p1, p2, 3]^2 for p2 in 1:SIZE_BZ[1]^2]), sum([kondoJArray[p1, p2, 1]^2 for p2 in 1:SIZE_BZ[1]^2]))
        @test results_norm[p1] ≈ sum([kondoJArray[p1, p2, 3]^2 for p2 in 1:SIZE_BZ[1]^2]) / sum([kondoJArray[p1, p2, 1]^2 for p2 in 1:SIZE_BZ[1]^2]) atol = 1e-10
    end

end
