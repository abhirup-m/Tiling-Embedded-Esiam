using Test
using fermions
using Random
using ProgressMeter

include("../source/constants.jl")
include("../source/helpers.jl")
include("../source/rgFlow.jl")
include("../source/probes.jl")
include("../source/main.jl")


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
            resultsCompareBool = zeros(size_BZ^2, size_BZ^2)
            resultsCompare[secondShellPoints] .= [sum(kondoJArray[p, secondAndBeyondPoints, 2] .^ 2) for p in secondShellPoints]
            resultsCompareBool[secondShellPoints] .= [sum(kondoJArray[p, secondAndBeyondPoints, 1] .^ 2) for p in secondShellPoints]
            @test resultsCompare[secondShellPoints] ./ resultsCompareBool[secondShellPoints] == results[secondShellPoints]
        end
    end
end


J_val = 0.1
orbitals = ("p", "p")
@testset "Correlation k-space symmetry, BZ size=$size_BZ, cluster size=$TRUNC_DIM" for size_BZ in [9, 13]
    @showprogress for W_val in range(0.0, stop=-J_val, length=2)
        savePaths = rgFlowData(size_BZ, -2.0, J_val, [W_val/J_val], orbitals)
        corrDefArray = [i -> Dict(("+-+-", [2, 1, 2 * i + 1, 2 * i + 2]) => 1.0, ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1]) => 1.0)]
        jldopen(savePaths[1], "r"; compress=true) do file
        kondoJArray = file["kondoJArray"]
        dispersion = file["dispersion"]
        W_val = file["W_val"]
        size_BZ = file["size_BZ"]
        orbitals = file["orbitals"]
        dispersion = file["dispersion"]
        FS_indices = [point for point in 1:size_BZ^2 if abs(dispersion[point]) < TOLERANCE]
        nodes = [point for point in FS_indices if isapprox(abs.(map1DTo2D(point, size_BZ))...)]
        antinodes = [point for point in FS_indices if 0 ∈ map1DTo2D(point, size_BZ)]

        res_pm = getCorrelations(size_BZ, dispersion, kondoJArray, 0.0, orbitals, corrDefArray)
        for (node, antinode) in zip(nodes[2:end], antinodes[2:end])
            @test res_pm[1][1][antinodes[1]] ≈ res_pm[1][1][antinode] atol = TOLERANCE
            @test res_pm[1][1][nodes[1]] ≈ res_pm[1][1][node] atol = TOLERANCE
        end
        if size_BZ == 9
            FS_sequences = [[45, 53], [53, 45], [45, 61], [61, 45], [61, 53], [53, 61]]
            basis = fermions.BasisStates(TRUNC_DIM * 2 + 2)
            results_FS = [0.0, 0.0, 0.0]
            for sequence in FS_sequences
                dispersionDict = Dict(zip(1:TRUNC_DIM, dispersion[sequence]))
                kondoDict = Dict(points => kondoJArray[sequence[collect(points)]..., end] for points in Iterators.product(1:TRUNC_DIM, 1:TRUNC_DIM))
                bathIntDict = Dict(points => bathIntForm(0.0, orbitals[2], size_BZ, sequence[collect(points)]) for points in Iterators.product(1:TRUNC_DIM, 1:TRUNC_DIM, 1:TRUNC_DIM, 1:TRUNC_DIM))
                hamiltonian = fermions.kondoKSpace(dispersionDict, kondoDict, bathIntDict; bathField=0.0, tolerance=TOLERANCE)
                matrix = fermions.generalOperatorMatrix(basis, hamiltonian)
                eigv, eigs = fermions.getSpectrum(matrix)
                correlationResults = fermions.gstateCorrelation(basis, eigv, eigs, corrDefArray[1].(1:TRUNC_DIM)) 
                for (index, res) in zip(sequence, correlationResults)
                    if index == 45
                        results_FS[1] += res / 4
                    elseif index == 53
                        results_FS[2] += res / 4
                    else
                        results_FS[3] += res / 4
                    end
                end
            end
            @test results_FS ≈ res_pm[1][1][[45, 53, 61]]
        end
        end
    end
end
