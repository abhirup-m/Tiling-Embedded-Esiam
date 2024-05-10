using Test
using Random
using ProgressMeter
include("../source/probes.jl")
include("../source/helpers.jl")
include("../source/rgFlow.jl")


function scattProbtest()
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
                @test results_norm[p1] ≈ results_compare ./ results_compare_bare rtol = TOLERANCE
                @test results_bool[p1] == results_compare_bool
            end
        end
    end
end


function test_correlation()
    J_val = 0.1
    size_BZ = 5
    orbitals = ("p", "p")
    Threads.@threads for W_val in range(0.0, stop=-J_val, length=4)
        kondoJArray, dispersion, fixedpointEnergy = momentumSpaceRG(size_BZ, -2.0, J_val, W_val, ("p", "p"))
        FS_indices = [point for point in 1:size_BZ^2 if abs(dispersion[point]) < TOLERANCE]
        nodes = [point for point in FS_indices if isapprox(abs.(map1DTo2D(point, size_BZ))...)]
        antinodes = [point for point in FS_indices if 0 ∈ map1DTo2D(point, size_BZ)]
        bathIntDict = Dict(indices => bathIntForm(W_val, orbitals[2], size_BZ, indices) for indices in Iterators.product(FS_indices, FS_indices, FS_indices, FS_indices))
        results, _, _ = spinFlipCorrMap(size_BZ, dispersion, kondoJArray, W_val, orbitals; trunc_dim = 3)
        println((results[3], results[9]))
        @testset "Correlation k-space symmetry W=$(round(W_val, digits=3))" begin
        for point in antinodes[2:end]
            @test abs(results[antinodes[1]]) > TOLERANCE
            @test results[antinodes[1]] ≈ results[point] atol = TOLERANCE
        end
        for point in nodes[2:end]
            @test abs(results[nodes[1]]) > TOLERANCE
            @test results[nodes[1]] ≈ results[point] atol = TOLERANCE
        end
        end
    end
end
