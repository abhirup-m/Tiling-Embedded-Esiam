using Test
using Combinatorics
using ProgressMeter
include("../../../fermionise/source/fermionise.jl")
include("../../../fermionise/source/correlations.jl")
include("../../../fermionise/source/models.jl")
include("../source/rgFlow.jl")
include("../source/probes.jl")


global J_val = 0.1
global orbitals = ["p", "p"]
global FS_indices = [3, 9, 15, 19, 23, 17, 11, 7]
global corrDefLeft = Dict(("+-+-", [2, 1, 3, 4]) => 1.0, ("+-+-", [1, 2, 4, 3]) => 1.0)
global corrDefMid = Dict(("+-+-", [2, 1, 5, 6]) => 1.0, ("+-+-", [1, 2, 6, 5]) => 1.0)
global corrDefRight = Dict(("+-+-", [2, 1, 7, 8]) => 1.0, ("+-+-", [1, 2, 8, 7]) => 1.0)

# given a set of momentum states (chosenIndices), return various steps 
# of the correlation calculation process (operator list, eigenvalues,
# correlation values) in order to run tests on them
function get_test_results(trunc_dim, chosenIndicesPerm, kondoJArray, W_val, bath_orbital, size_BZ, dispersion)
    basis = BasisStates(trunc_dim * 2 + 2)
    dispersionDict, kondoDict, kondoDictBare, bathIntDict = sampleKondoIntAndBathInt(chosenIndicesPerm, dispersion, kondoJArray, (W_val, bath_orbital, size_BZ))
    operatorList = kondoKSpace(dispersionDict, kondoDict, bathIntDict)
    matrix = generalOperatorMatrix(basis, operatorList)
    eigvals, eigvecs = getSpectrum(matrix)
    return operatorList, matrix, eigvals, eigvecs
end


function test_hermiticity()
    for size_BZ in [5, 9]
        @testset "Hamiltonian hermiticity, BZ size = $size_BZ, cluster size = $trunc_dim" for trunc_dim in [2, 3]
            for W_val in [0, -J_val / 10, -J_val]
                kondoJArray, dispersion, fixedpointEnergy = momentumSpaceRG(size_BZ, -2.0, J_val, W_val, ("p", "p"))
                FS_indices = [point for point in 1:size_BZ^2 if abs(dispersion[point]) < TOLERANCE]
                for chosenIndices in collect(combinations(FS_indices, trunc_dim))
                    basis = BasisStates(trunc_dim * 2 + 2)
                    dispersionDict, kondoDict, kondoDictBare, bathIntDict = sampleKondoIntAndBathInt(chosenIndices, dispersion, kondoJArray, (W_val, orbitals[2], size_BZ))
                    operatorList = kondoKSpace(dispersionDict, kondoDict, bathIntDict)
                    matrix = generalOperatorMatrix(basis, operatorList)
                    @test all([isapprox(subMatrix, subMatrix', atol=TOLERANCE) for subMatrix in values(matrix)])
                end
            end
        end
    end
end


function test_eigvals()
    for size_BZ in [5, 9]
        for trunc_dim in [2, 3]
            @testset "eigenvalue insenstivity to sequence, BZ size = $size_BZ, cluster size = $trunc_dim, W = $W_val" for W_val in [0, -J_val / 10, -J_val]
                kondoJArray, dispersion, fixedpointEnergy = momentumSpaceRG(size_BZ, -2.0, J_val, W_val, ("p", "p"))
                FS_indices = [point for point in 1:size_BZ^2 if abs(dispersion[point]) < TOLERANCE]
                for chosenIndices in collect(combinations(FS_indices, trunc_dim))
                    eigvalsSet = []
                    for chosenIndicesPerm in permutations(chosenIndices, trunc_dim)
                        operatorList, matrix, eigvals, eigvecs = get_test_results(trunc_dim, chosenIndicesPerm, kondoJArray, W_val, orbitals[2], size_BZ, dispersion)
                        push!(eigvalsSet, eigvals)
                    end
                    for eigvals in eigvalsSet
                        @test keys(eigvals) == keys(eigvalsSet[1])
                        @test all([isapprox(v1, v2, atol=TOLERANCE) for (v1, v2) in zip(values(eigvals), values(eigvalsSet[1]))])
                    end
                end
            end
        end
    end
end
