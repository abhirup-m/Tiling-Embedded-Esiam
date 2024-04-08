using LinearAlgebra
using CairoMakie
using Makie

function scattProb(kondoJArray::Array{Float64,3}, stepIndex::Int64)
    results = diag(kondoJArray[:, :, stepIndex] * kondoJArray[:, :, stepIndex]')
    results_bare = diag(kondoJArray[:, :, stepIndex] * kondoJArray[:, :, 1]')
    results_bool = tolerantSign.(abs.(results), RG_RELEVANCE_TOL) # [tolerantSign(results_norm_i, RG_RELEVANCE_TOL) for results_norm_i in results_norm]
    results[results_bool.<0] .= 0
    return results, results_bare, results_bool
end


function kondoCoupMap(k_vals, size_BZ, kondoJArrayFull)
    kx, ky = k_vals
    kspacePoint = map2DTo1D(kx, ky, size_BZ)
    results = kondoJArrayFull[kspacePoint, :, end] .^ 2
    results_bare = kondoJArrayFull[kspacePoint, :, 1] .^ 2
    results_bool = tolerantSign.(abs.(results), RG_RELEVANCE_TOL / size_BZ^2)
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

