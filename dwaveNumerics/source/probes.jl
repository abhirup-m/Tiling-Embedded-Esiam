using LinearAlgebra
using CairoMakie

function scattProb(kondoJArray::Array{Float64,3}, stepIndex::Int64)
    bare_J_squared = diag(kondoJArray[:, :, 1] * kondoJArray[:, :, 1]')
    results_unnorm = diag(kondoJArray[:, :, stepIndex] * kondoJArray[:, :, stepIndex]')
    results_norm = results_unnorm ./ bare_J_squared
    results_bool = [tolerantSign(results_norm_i, 1) for results_norm_i in results_norm]
    return results_norm, results_unnorm, results_bool
end


function kondoCoupFSMap(size_BZ, kondoJArrayFull)
    chunk = 0.25 * (size_BZ - 5)
    node = trunc(Int64, (1 + chunk) * size_BZ + 2 + chunk)
    antinode = trunc(Int64, 3 + 2 * chunk)
    results1 = kondoJArrayFull[node, :, end]
    results2 = kondoJArrayFull[antinode, :, end]
    return [results1, results2]
end


function kondoCoupMidwayMap(size_BZ, kondoJArrayFull)
    chunk = 0.25 * (size_BZ - 5)
    point1 = trunc(Int64, ((3 / 8) * size_BZ - 1) * size_BZ) + trunc(Int64, (3 / 8) * size_BZ)
    antinode = 3 + 2 * chunk
    point2 = trunc(Int64, antinode + size_BZ)
    kx, ky = map1DTo2D([point1, point2], size_BZ)
    println([kx, ky])
    results1 = kondoJArrayFull[point1, :, end]
    results2 = kondoJArrayFull[point2, :, end]
    return [results1, results2], [kx, ky]
end


function kondoCoupDiagMap(size_BZ, kondoJArrayFull)
    return [kondoJArrayFull[i, i, end] for i in 1:size_BZ^2]
end

function plotHeatmaps(results1, results2, xarr, yarr, fig, axes, cmap_left)
    replace!(results1, NaN => 0)
    replace!(results2, NaN => 0)
    minima = minimum.([results1, results2])
    maxima = maximum.([results1, results2])

    hm1 = heatmap!(axes[1], xarr, yarr, results1, colormap=cmap_left,
    )
    Colorbar(fig[1, 2], hm1, colorrange=(minima[1], maxima[1]))
    hm2 = heatmap!(axes[2], xarr, yarr, results2, colormap=:matter,
    )
    Colorbar(fig[1, 4], hm2, colorrange=(minima[2], maxima[2]))
    return axes
end

