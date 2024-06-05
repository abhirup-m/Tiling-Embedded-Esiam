using Test
using Random
using ProgressMeter
include("../source/probes.jl")
include("../source/helpers.jl")
include("../source/rgFlow.jl")


@testset "Flow of totalScatterProb" begin
    size_BZ = 9
    firstShellPoints = [1, size_BZ, size_BZ^2 - size_BZ + 1, size_BZ^2, trunc(Int, 0.5 * (size_BZ^2 + 1))]
    secondShellPoints = [2, size_BZ - 1, size_BZ + 1, 2 * size_BZ, size_BZ * (size_BZ - 2) + 1, size_BZ^2 - size_BZ, size_BZ^2 - size_BZ + 2, size_BZ^2 - 1]
    secondAndBeyondPoints = [p for p in 1:size_BZ^2 if p ∉ firstShellPoints]
    J_val = 0.1
    orbitals = ("p", "p")
    @testset for orbitals in [("p", "p"), ("d", "d")]
        for W_val in [0.0, -0.1 * J_val, -J_val]
            kondoJArray, dispersion = momentumSpaceRG(size_BZ, -2.0, J_val, W_val, orbitals)
            results, results_bool = scattProb(kondoJArray, size_BZ, dispersion)
            resultsCompare = zeros(size_BZ^2, size_BZ^2)
            resultsCompare[secondShellPoints] .= [sum(kondoJArray[p, secondAndBeyondPoints, 2] .^ 2) for p in secondShellPoints]
            @test resultsCompare[secondShellPoints] == results[secondShellPoints]
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
