using LinearAlgebra
using CairoMakie
using Makie

function scattProb(kondoJArray::Array{Float64,3}, stepIndex::Int64, size_BZ::Int64, dispersion::Vector{Float64}, fixedpointEnergy::Float64)
    println(fixedpointEnergy)
    results = zeros(size_BZ^2)
    results_bare = zeros(size_BZ^2)
    E_cloud = dispersion[-fixedpointEnergy.<=dispersion.<=fixedpointEnergy]
    point2_arr = unique(getIsoEngCont(dispersion, E_cloud))
    Threads.@threads for point1 in 1:size_BZ^2
        kx, ky = map1DTo2D(point1, size_BZ)
        if (abs(kx) ≈ K_MAX && abs(ky) ≈ K_MAX) || (kx ≈ 0 && ky ≈ 0)
            continue
        end
        results[point1] = sum(kondoJArray[point1, point2_arr, stepIndex] .^ 2)
        results_bare[point1] = sum(kondoJArray[point1, point2_arr, 1] .^ 2)
    end
    results_bool = tolerantSign.(abs.(results ./ results_bare), RG_RELEVANCE_TOL)
    results[results_bool.<0] .= NaN
    return results, results_bare, results_bool
end


function kondoCoupMap(k_vals, size_BZ, kondoJArrayFull)
    kx, ky = k_vals
    kspacePoint = map2DTo1D(kx, ky, size_BZ)
    results = kondoJArrayFull[kspacePoint, :, end] .^ 2
    results_bare = kondoJArrayFull[kspacePoint, :, 1] .^ 2
    results_bool = tolerantSign.(abs.(results), RG_RELEVANCE_TOL / size_BZ)
    results[results_bool.<0] .= NaN
    return results, results_bare, results_bool
end


function plotHeatmaps(results_arr, xarr, yarr, fig, axes, cmaps, size_BZ)
    for (i, result) in enumerate(results_arr)
        reshaped_result = reshape(result, (size_BZ, size_BZ))
        hmap = heatmap!(axes[i], xarr, yarr, reshaped_result, colormap=cmaps[i],
        )
        Colorbar(fig[1, 2*i], hmap, colorrange=(minimum(result), maximum(result)))
    end
    return axes
end

