using Test
using Random
using ProgressMeter
include("../source/probes.jl")
include("../source/helpers.jl")
include("../source/rgFlow.jl")


@testset "Flow of totalScatterProb" begin
    kondoJArray = rand(Float64, (SIZE_BZ[1]^2, SIZE_BZ[1]^2, 3))
    dOfStates, dispersion = getDensityOfStates(tightBindDisp, SIZE_BZ[1])
    results, results_bool = scattProb(kondoJArray, SIZE_BZ[1], dispersion)
    allPoints = collect(1:SIZE_BZ[1]^2)
    @testset for p1 in allPoints
        targetStatesForPoint = collect(1:SIZE_BZ[1]^2)[abs.(dispersion).<=abs(dispersion[p1])]
        results_compare = sum(kondoJArray[p1, targetStatesForPoint, 3] .^ 2) / length(targetStatesForPoint)
        results_compare_bare = sum(kondoJArray[p1, targetStatesForPoint, 1] .^ 2) / length(targetStatesForPoint)
        results_scaled = results_compare_bare == 0 ? results_compare : results_compare / results_compare_bare
        if !(p1 in CORNER_POINTS || p1 in CENTER_POINTS)
            @test results[p1] ≈ results_scaled atol = TOLERANCE
        end
    end
end


J_val = 0.1
orbitals = ("p", "p")
for trunc_dim in [2,]
    @testset "Correlation k-space symmetry, BZ size=$size_BZ, cluster size=$trunc_dim" for size_BZ in [5, 9, 13]
        @showprogress for W_val in range(0.0, stop=-J_val, length=2)
            kondoJArray, dispersion = momentumSpaceRG(size_BZ, -2.0, J_val, W_val, ("p", "p"))
            FS_indices = [point for point in 1:size_BZ^2 if abs(dispersion[point]) < TOLERANCE]
            nodes = [point for point in FS_indices if isapprox(abs.(map1DTo2D(point, size_BZ))...)]
            antinodes = [point for point in FS_indices if 0 ∈ map1DTo2D(point, size_BZ)]
            results, _ = correlationMap(size_BZ, dispersion, kondoJArray, W_val, orbitals, i -> Dict(("+-+-", [2, 1, 2 * i + 1, 2 * i + 2]) => 1.0, ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1]) => 1.0); trunc_dim=trunc_dim)
            for (node, antinode) in zip(nodes[2:end], antinodes[2:end])
                @test results[antinodes[1]] ≈ results[antinode] atol = TOLERANCE
                @test results[nodes[1]] ≈ results[node] atol = TOLERANCE
            end
        end
    end
end
