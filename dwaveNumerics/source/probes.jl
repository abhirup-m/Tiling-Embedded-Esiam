using LinearAlgebra
using DelimitedFiles
using Plots

animName(orbitals, size_BZ, scale, W_by_J_min, W_by_J_max, J_val) = "kondoUb_kspaceRG_$(orbitals[1])_$(orbitals[2])_wave_$(size_BZ)_$(round(W_by_J_min, digits=4))_$(size_BZ)_$(round(W_by_J_max, digits=4))_$(round(J_val, digits=4))_$(SIZE[1] * scale)x$(SIZE[2] * scale)"
discreteCgrad = cgrad(:BuPu_3, 3, categorical=true)

function scattProb(kondoJArray::Array{Float64,3}, stepIndex::Int64)
    bare_J_squared = diag(kondoJArray[:, :, 1] * kondoJArray[:, :, 1]')
    results_unnorm = diag(kondoJArray[:, :, stepIndex] * kondoJArray[:, :, stepIndex]')
    results_norm = results_unnorm ./ bare_J_squared
    results_bool = [tolerantSign(results_norm_i, 1) for results_norm_i in results_norm]
    return results_norm, results_unnorm, results_bool
end


function kondoCoupFSMap(size_BZ, kondoJArrayFull)
    node = trunc(Int64, 0.25 * (size_BZ - 1) * (size_BZ + 1) + 1)
    antinode = trunc(Int64, trunc(Int, 0.5 * (size_BZ - 1) + 1))
    results1 = kondoJArrayFull[node, :, end]
    results2 = kondoJArrayFull[antinode, :, end]
    return [results1, results2]
end


function kondoCoupDiagMap(size_BZ, kondoJArrayFull)
    return [kondoJArrayFull[i, i, end] for i in 1:size_BZ^2]
end

function plotHeatmaps(results1, results2, k_vals, plots, cmap_left, titles)
    minima = minimum.([results1, results2])
    maxima = maximum.([results1, results2])
    heatmap!(plots[1], k_vals, k_vals, results1,
        cmap=cmap_left,
        clims=(minima[1], maxima[1]),
        title=titles[1],
        titlefontsize=11
    )
    heatmap!(plots[2], k_vals, k_vals, results2,
        cmap=:matter,
        clims=(minima[2], maxima[2]),
        title=titles[2],
        titlefontsize=11
    )
    return plots
end

