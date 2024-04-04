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

function manager(size_BZ::Int64, J_val::Float64, W_by_J_range::Vector{Float64}, orbitals::Vector{String}, probes::Vector{String}, figScale::Float64)
    @assert size_BZ % 2 != 0
    results_arr = zeros((length(probes), length(W_by_J_range), size_BZ^2))
    savePaths = ["data/$(orbitals)_$(size_BZ)_$(round(J_val, digits=3))_$(round(W_by_J, digits=3))" for W_by_J in W_by_J_range]
    for (j, W_by_J) in enumerate(W_by_J_range)
        kondoJArrayFull, dispersionArray = main(size_BZ, J_val, J_val * W_by_J, orbitals)
        writedlm(savePaths[j], kondoJArrayFull[:,:,[1, end]])
    end
    k_vals = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    for probe in probes
        pdfFileNames = []
        anim = @animate for (W_by_J, savePath) in zip(W_by_J_range, savePaths)
            kondoJArrayFull = readdlm(savePath)
            kondoJArrayFull = reshape(kondoJArrayFull, (size_BZ^2, size_BZ^2, 2))
            p1 = plot()
            p2 = plot()
            if probe == "scattProb"
                results_norm, results_unnorm, results_bool = scattProb(kondoJArrayFull, size(kondoJArrayFull)[3])
                results1 = results_bool
                results2 = log10.(results_norm)
                title_left = "relevance/irrelevanceo of \$\\Gamma(k)\$"
                title_right = "\$\\Gamma(k) = \\sum_q J(k,q)^2\$"
                cmap_left = discreteCgrad
                p1, p2 = plotHeatmaps(reshape(results1, (size_BZ, size_BZ)),
                    reshape(results2, (size_BZ, size_BZ)),
                    k_vals, [p1, p2], cmap_left, [title_left, title_right])
            elseif probe == "kondoCoupFSMap"
                node = trunc(Int64, 0.25 * (size_BZ - 1) * (size_BZ + 1) + 1)
                antinode = trunc(Int64, trunc(Int, 0.5 * (size_BZ - 1) + 1))
                results1 = kondoJArrayFull[node, :, end]
                results2 = kondoJArrayFull[antinode, :, end]
                title_left = "\$J(k,q_\\mathrm{node})\$"
                title_right = "\$J(k,q_{\\mathrm{antin.}})\$"
                cmap_left = :matter
                p1, p2 = plotHeatmaps(reshape(results1, (size_BZ, size_BZ)),
                    reshape(results2, (size_BZ, size_BZ)),
                    k_vals, [p1, p2], cmap_left, [title_left, title_right])
                scatter!(p1, [-1 / 2], [-1 / 2], markersize=6, markercolor=:grey, markerstrokecolor=:white, markerstrokewidth=4)
                scatter!(p2, [0], [-1], markersize=6, markercolor=:grey, markerstrokecolor=:white, markerstrokewidth=4)
            elseif probe == "kondoCoupDiagMap"
                node = trunc(Int64, 0.25 * (size_BZ - 1) * (size_BZ + 1) + 1)
                antinode = trunc(Int64, trunc(Int, 0.5 * (size_BZ - 1) + 1))
                results1 = [kondoJArrayFull[i, i, end] for i in 1:size_BZ^2]
                results2 = results1
                title_left = "\$J(k,k)\$"
                title_right = ""
                cmap_left = :matter
                p1, p2 = plotHeatmaps(reshape(results1, (size_BZ, size_BZ)),
                    reshape(results2, (size_BZ, size_BZ)),
                    k_vals, [p1, p2], cmap_left, [title_left, title_right])
            end
            plot(p1, p2, size=SIZE,
                top_margin=3mm, left_margin=[5mm 10mm], bottom_margin=5mm,
                xlabel="\$ \\mathrm{k_x/\\pi} \$", ylabel="\$ \\mathrm{k_y/\\pi} \$",
                dpi=100 * figScale,
            )
            pdfFileName = "fig_$(W_by_J).pdf"
            savefig(pdfFileName)
            push!(pdfFileNames, pdfFileName)
        end
        plotName = animName(orbitals, size_BZ, figScale, minimum(W_by_J_range), maximum(W_by_J_range), J_val)
        run(`pdfunite $pdfFileNames fig-$(plotName)-$(replace(probe, " " => "-")).pdf`)
        run(`rm $pdfFileNames`)
        gif(anim, plotName * "-$(replace(probe, " " => "-")).gif", fps=0.5)
        gif(anim, plotName * "-$(replace(probe, " " => "-")).mp4", fps=0.5)
    end
end
