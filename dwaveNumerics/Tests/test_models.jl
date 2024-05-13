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
    kondoDict, kondoDictBare, bathIntDict = sampleKondoIntAndBathInt(chosenIndicesPerm, kondoJArray, (W_val, bath_orbital, size_BZ))
    operatorList = kondoKSpace(dispersion[chosenIndicesPerm], kondoDict, bathIntDict)
    matrix = generalOperatorMatrix(basis, operatorList)
    eigvals, eigvecs = getSpectrum(matrix)
    return operatorList, matrix, eigvals, eigvecs
end


function test_hermiticity()
    for size_BZ in [5, 9]
        @testset "Hamiltonian hermiticity, BZ size = $size_BZ, cluster size = $trunc_dim" for trunc_dim in [2, 3]
            for W_val in [0, -J_val/10, -J_val]
                kondoJArray, dispersion, fixedpointEnergy = momentumSpaceRG(size_BZ, -2.0, J_val, W_val, ("p", "p"))
                FS_indices = [point for point in 1:size_BZ^2 if abs(dispersion[point]) < TOLERANCE]
                for chosenIndices in collect(combinations(FS_indices, trunc_dim))
                    basis = BasisStates(trunc_dim * 2 + 2)
                    kondoDict, kondoDictBare, bathIntDict = sampleKondoIntAndBathInt(chosenIndices, kondoJArray, (W_val, orbitals[2], size_BZ))
                    operatorList = kondoKSpace(dispersion[chosenIndices], kondoDict, bathIntDict)
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
            @testset "eigenvalue insenstivity to sequence, BZ size = $size_BZ, cluster size = $trunc_dim, W = $W_val" for W_val in [0, -J_val/10, -J_val]
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





function test_correlation_old()
    @testset begin
        @testset "W = $W_val" for W_val in [-J_val]#[0.0, -J_val / 10, ]
            bathIntDict = Dict(indices => bathIntForm(W_val, orbitals[2], size_BZ, indices) for indices in Iterators.product(FS_indices, FS_indices, FS_indices, FS_indices))
            kondoJArray, dispersion, fixedpointEnergy = momentumSpaceRG(size_BZ, -2.0, J_val, W_val, ("p", "p"))
            kondoDict = Dict((i, j) => kondoJArray[i, j, end] for (i, j) in Iterators.product(FS_indices, FS_indices))
            for chosenIndexSets in [[[3, 9, 15], [15, 9, 3], [3, 17, 15], [15, 17, 3]],
                [[3, 7, 11], [11, 7, 3], [11, 19, 3], [3, 19, 11]],
                [[15, 3, 11], [11, 3, 15], [11, 23, 15], [15, 23, 11]],
                [[23, 15, 3], [3, 15, 23], [23, 11, 3], [3, 11, 23]],
            ]
                operatorLists = []
                matrices = []
                eigenvalSet = []
                eigenvecSet = []
                correlationsLeft = []
                correlationsMid = []
                correlationsRight = []
                for chosenIndices in chosenIndexSets
                    operatorList, matrix, eigvals, eigvecs, correlationLeft, correlationMid, correlationRight = get_test_results(chosenIndices, dispersion, kondoDict, bathIntDict)
                    push!(operatorLists, operatorList)
                    push!(matrices, matrix)
                    push!(eigenvalSet, eigvals)
                    push!(eigenvecSet, eigvecs)
                    push!(correlationsLeft, correlationLeft)
                    push!(correlationsMid, correlationMid)
                    push!(correlationsRight, correlationRight)
                end

                # check that all choices with two antinodes ((3,15) or (3,11) or (15, 11) etc) ) 
                # and a third point placed symmetrical to them will give identical Hamiltonians, 
                # matrices, eigenvals and vecs as well as identical correlations.
                @test allequal(operatorLists)
                @test allequal(matrices)
                @test allequal(eigenvalSet)
                @test allequal(eigenvecSet)
                println(chosenIndexSets)
                @test allequal(round.([correlationsLeft; correlationsRight], digits=trunc(Int, -log10(TOLERANCE))))
            end

            for chosenIndices in combinations(FS_indices, 3)
                startingSequence = copy(chosenIndices)
                operatorLists = []
                matrices = []
                eigenvalSet = []
                correlationsLeft = []
                correlationsMid = []
                correlationsRight = []
                mappings = []
                for sequence in collect(permutations(chosenIndices))[1:2]
                    push!(mappings, Dict())
                    for (i, v) in enumerate(startingSequence)
                        i_prime = collect(1:length(chosenIndices))[sequence.==v][1]
                        mappings[end][1] = 1
                        mappings[end][2] = 2
                        mappings[end][2*i+1] = 2 * i_prime + 1
                        mappings[end][2*i+2] = 2 * i_prime + 2
                    end
                    operatorList, matrix, eigvals, _, correlationLeft, correlationMid, correlationRight = get_test_results(sequence, dispersion, kondoDict, bathIntDict)
                    push!(operatorLists, operatorList)
                    push!(matrices, matrix)
                    push!(eigenvalSet, eigvals)
                    push!(correlationsLeft, [correlationLeft, correlationMid, correlationRight][sequence.==startingSequence[1]][1])
                    push!(correlationsMid, [correlationLeft, correlationMid, correlationRight][sequence.==startingSequence[2]][1])
                    push!(correlationsRight, [correlationLeft, correlationMid, correlationRight][sequence.==startingSequence[3]][1])
                end

                for (mapping, operatorList) in zip(mappings, operatorLists)
                    compareList = Dict((opType, [mapping[m] for m in members]) => value for ((opType, members), value) in operatorLists[1])
                    @test compareList == operatorList
                end

                for eigvals in eigenvalSet[2:end]
                    @test keys(eigvals) == keys(eigenvalSet[1])
                    @test all(isapprox(eigvals[k], eigenvalSet[1][k]; atol=TOLERANCE) for k in keys(eigvals))
                end
                # println(correlationsRight)
                # println(correlationsMid)
                # println(correlationsLeft)
                # @test allequal(round.(correlationsLeft, digits=trunc(Int, -log10(TOLERANCE))))
                # @test allequal(round.(correlationsMid, digits=trunc(Int, -log10(TOLERANCE))))
                # @test allequal(round.(correlationsRight, digits=trunc(Int, -log10(TOLERANCE))))
            end

            results = zeros(maximum(FS_indices))
            # for chosenIndices in combinations(FS_indices, 3)
            #     for chosenIndicesPerm in permutations(chosenIndices)
            #         # for chosenIndices in combinations([3, 9, 15, 19, 23, 17, 11, 7], 3)
            #         operatorList = kondoKSpace_test(chosenIndicesPerm, dispersion, kondoDict, bathIntDict)
            #         matrix = generalOperatorMatrix(basis, operatorList)
            #         eigvals, eigvecs = getSpectrum(matrix)
            #         for (i, index) in enumerate(chosenIndicesPerm)
            #             corrDef_i = Dict(("+-+-", [2, 1, 2 * i + 1, 2 * i + 2]) => 1.0, ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1]) => 1.0)
            #             correlations = simpleCorrelation(basis, eigvals, eigvecs, corrDef_i)
            #             # if 3 in chosenIndices || 23 in chosenIndices
            #             #     println(chosenIndices, sort(correlations[correlations.!=0]), index)
            #             # end
            #             results[index] += sum(abs.(correlations))
            #         end
            #     end
            #     # println(chosenIndices, round.([results[chosenIndices[1]], results[chosenIndices[2]], results[chosenIndices[3]]], digits=5))
            # end
            # @test results[3] ≈ results[15] atol = TOLERANCE
            # @test results[3] ≈ results[23] atol = TOLERANCE
        end

    end
end

function test_correlation()
    @testset "W = $W_val" for W_val in range(0.0, stop=-J_val, length=10)
        bathIntDict = Dict(indices => bathIntForm(W_val, orbitals[2], size_BZ, indices) for indices in Iterators.product(FS_indices, FS_indices, FS_indices, FS_indices))
        kondoJArray, dispersion, fixedpointEnergy = momentumSpaceRG(size_BZ, -2.0, J_val, W_val, ("p", "p"))
        kondoDict = Dict((i, j) => kondoJArray[i, j, end] for (i, j) in Iterators.product(FS_indices, FS_indices))
        results = zeros(maximum(FS_indices))
        @showprogress for chosenIndices in collect(combinations(FS_indices, 3))
            for chosenIndicesPerm in permutations(chosenIndices)
                operatorList = kondoKSpace_test(chosenIndicesPerm, dispersion, kondoDict, bathIntDict)
                matrix = generalOperatorMatrix(basis, operatorList)
                eigvals, eigvecs = getSpectrum(matrix)
                for (i, index) in enumerate(chosenIndicesPerm)
                    corrDef_i = Dict(("+-+-", [2, 1, 2 * i + 1, 2 * i + 2]) => 1.0, ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1]) => 1.0)
                    correlations = simpleCorrelation(basis, eigvals, eigvecs, corrDef_i)
                    results[index] += sum(abs.(correlations))
                end
            end
        end
        @test abs(results[3]) > TOLERANCE
        @test abs(results[9]) > TOLERANCE
        @test results[3] ≈ results[15] atol = TOLERANCE
        @test results[3] ≈ results[23] atol = TOLERANCE
        @test results[3] ≈ results[11] atol = TOLERANCE
        @test results[9] ≈ results[17] atol = TOLERANCE
        @test results[9] ≈ results[19] atol = TOLERANCE
        @test results[9] ≈ results[7] atol = TOLERANCE
    end
end
