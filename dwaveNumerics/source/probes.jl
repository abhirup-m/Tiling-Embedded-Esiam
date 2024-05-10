using LinearAlgebra
using Combinatorics
include("../../../fermionise/source/fermionise.jl")
include("../../../fermionise/source/correlations.jl")
include("../../../fermionise/source/models.jl")

function scattProb(kondoJArray::Array{Float64,3}, stepIndex::Int64, size_BZ::Int64, dispersion::Vector{Float64}, fixedpointEnergy::Float64)
    results = zeros(size_BZ^2)
    results_bare = zeros(size_BZ^2)
    E_cloud = dispersion[-fixedpointEnergy.<=dispersion.<=fixedpointEnergy]
    point2_arr = unique(getIsoEngCont(dispersion, E_cloud))
    Threads.@threads for point1 in 1:size_BZ^2
        kx, ky = map1DTo2D(point1, size_BZ)
        if (abs(kx) ≈ K_MAX && abs(ky) ≈ K_MAX) || (kx ≈ 0 && ky ≈ 0)
            continue
        end
        results[point1] = sum(kondoJArray[point1, point2_arr, stepIndex] .^ 2) / length(point2_arr)
        results_bare[point1] = sum(kondoJArray[point1, point2_arr, 1] .^ 2) / length(point2_arr)
    end
    results_bool = tolerantSign.(abs.(results), abs.(results_bare) .* RG_RELEVANCE_TOL)
    results[results_bool.<0] .= NaN
    return results ./ results_bare, results_bare, results_bool
end


function kondoCoupMap(k_vals::Tuple{Float64,Float64}, size_BZ::Int64, kondoJArrayFull::Array{Float64,3})
    kx, ky = k_vals
    kspacePoint = map2DTo1D(kx, ky, size_BZ)
    results = kondoJArrayFull[kspacePoint, :, end] .^ 2
    results_bare = kondoJArrayFull[kspacePoint, :, 1] .^ 2
    results_bool = tolerantSign.(abs.(results), RG_RELEVANCE_TOL / size_BZ)
    results[results_bool.<0] .= NaN
    return results, results_bare, results_bool
end


function spinFlipCorrMap(size_BZ::Int64, dispersion::Vector{Float64}, kondoJArray::Array{Float64,3}, W_val::Float64, orbitals::Tuple{String,String}; trunc_dim::Int64=3)

    # get size of each quadrant (number of points in the range [0, π])
    halfSize = trunc(Int, (size_BZ + 1) / 2)

    # get all possible indices of momentum states within the Brillouin zone
    k_indices = collect(1:size_BZ^2)

    # initialise zero matrix for storing correlations
    results = zeros(size_BZ^2)
    results_bare = zeros(size_BZ^2)
    contributorCounter = zeros(size_BZ^2)

    # generating basis states for constructing prototype Hamiltonians which
    # will be diagonalised to obtain correlations
    basis = BasisStates(trunc_dim * 2 + 2)

    kondoDict = Dict((i, j) => kondoJArray[i, j, end] for (i, j) in Iterators.product(k_indices, k_indices))
    kondoDictBare = Dict((i, j) => kondoJArray[i, j, 1] for (i, j) in Iterators.product(k_indices, k_indices))
    bathIntDict = Dict(indices => bathIntForm(W_val, orbitals[2], size_BZ, indices) for indices in Iterators.product(k_indices, k_indices, k_indices, k_indices))
    correlationOperators = [Dict(("+-+-", [2, 1, 2 * i + 1, 2 * i + 2]) => 1.0, ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1]) => 1.0) for i in 1:trunc_dim]
    for energy in [0]
        suitableIndices = k_indices[abs.(dispersion[k_indices]).<=energy+TOLERANCE]
        for chosenIndices in collect(combinations(suitableIndices, trunc_dim))
            for chosenIndicesPerm in permutations(chosenIndices)
                operatorList = kondoKSpace(chosenIndicesPerm, dispersion, kondoDict, bathIntDict)
                matrix = generalOperatorMatrix(basis, operatorList)
                eigvals, eigvecs = getSpectrum(matrix)
                results[chosenIndicesPerm] .+= [sum(abs.(gstateCorrelation(basis, eigvals, eigvecs, correlationOperators[i]))) for (i, index) in enumerate(chosenIndicesPerm)]
            end
            contributorCounter[chosenIndices] .+= factorial(trunc_dim)
        end
    end
    results ./= contributorCounter
    results_bool = Dict(k => tolerantSign.(abs(results[k]), abs.(results_bare[k]) .* RG_RELEVANCE_TOL) for k in keys(results))
    return results, results_bare, results_bool
end


function mapProbeNameToProbe(probeName, size_BZ, kondoJArrayFull, W_by_J, J_val, dispersion, orbitals, fixedpointEnergy)
    titles = Vector{LaTeXString}(undef, 3)
    cmaps = [DISCRETE_CGRAD, :matter, :matter]
    if probeName == "scattProb"
        results, results_bare, results_bool = scattProb(kondoJArrayFull, size(kondoJArrayFull)[3], size_BZ, dispersion, fixedpointEnergy)
    elseif probeName == "kondoCoupNodeMap"
        results, results_bare, results_bool = kondoCoupMap(node, size_BZ, kondoJArrayFull)
    elseif probeName == "kondoCoupAntinodeMap"
        results, results_bare, results_bool = kondoCoupMap(antinode, size_BZ, kondoJArrayFull)
    elseif probeName == "kondoCoupOffNodeMap"
        results, results_bare, results_bool = kondoCoupMap(offnode, size_BZ, kondoJArrayFull)
    elseif probeName == "kondoCoupOffAntinodeMap"
        results, results_bare, results_bool = kondoCoupMap(offantinode, size_BZ, kondoJArrayFull)
    elseif probeName == "spinFlipCorrMap"
        results, results_bool = spinFlipCorrMapCoarse(size_BZ, dispersion, kondoJArrayFull, W_by_J * J_val, orbitals)
    end
    return results, results_bool
end
