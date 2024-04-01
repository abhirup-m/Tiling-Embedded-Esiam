using Test
include("../source/rgFlow.jl")

@testset "Getting cutoff energies" begin
    cutOffEnergies = getCutOffEnergy(SIZE_BZ[1])
    for (i, E) in enumerate(cutOffEnergies)
        @test E ≈ [4, 2, 0][i] atol = 1e-10
    end
end

@testset "Initialise Kondo coupling matrix" begin
    kondoJArray = initialiseKondoJ(SIZE_BZ[1], "p", 1, 1)[:, :, 1]
    @test all(diag(kondoJArray[:, :]) .== 1)
    @test kondoJArray[[1, 6, 11, 16, 21], :] ≈ kondoJArray[[5, 10, 15, 20, 25], :] atol = 1e-10
    @test kondoJArray[[1, 2, 3, 4, 5], :] ≈ kondoJArray[[21, 22, 23, 24, 25], :] atol = 1e-10
    @test kondoJArray[[2, 7, 12, 17, 22], :] ≈ -kondoJArray[[14, 19, 24, 9, 14], :] atol = 1e-10
    @test kondoJArray[[6, 7, 8, 9, 10], :] ≈ -kondoJArray[[18, 19, 20, 17, 18], :] atol = 1e-10
    @test kondoJArray[[1, 2, 3, 4, 5], :] ≈ kondoJArray[[21, 22, 23, 24, 25], :] atol = 1e-10
    @test all([kondoJArray[p1, p2] == kondoJArray[p2, p1] for p1 in eachindex(kondoJArray[1, :]) for p2 in eachindex(kondoJArray[:, 1])])
    for p2 in eachindex(kondoJArray[:, 1])
        kx, ky = map1DTo2D(p2, SIZE_BZ[1])
        @test kondoJArray[1, p2] ≈ -0.5 * (cos(kx) + cos(ky)) atol = 1e-10
        @test kondoJArray[2, p2] ≈ -0.5 * (sin(kx) + cos(ky)) atol = 1e-10
        @test kondoJArray[3, p2] ≈ 0.5 * (cos(kx) - cos(ky)) atol = 1e-10
        @test kondoJArray[6, p2] ≈ -0.5 * (cos(kx) + sin(ky)) atol = 1e-10
        @test kondoJArray[7, p2] ≈ -0.5 * (sin(kx) + sin(ky)) atol = 1e-10
        @test kondoJArray[8, p2] ≈ 0.5 * (cos(kx) - sin(ky)) atol = 1e-10
        @test kondoJArray[11, p2] ≈ 0.5 * (-cos(kx) + cos(ky)) atol = 1e-10
        @test kondoJArray[12, p2] ≈ 0.5 * (-sin(kx) + cos(ky)) atol = 1e-10
        @test kondoJArray[13, p2] ≈ 0.5 * (cos(kx) + cos(ky)) atol = 1e-10
    end
end
