using CairoMakie
theme_latexfonts()
using Makie
using LaTeXStrings

function plotHeatmaps(axisVals, axisLabels, figLabels, subFigTitles, collatedResults, saveName)
    # cmaps = [:PuBu_9, :seaborn_icefire_gradient, :seaborn_rocket_gradient]
    fig = Figure(dpi=600, figure_padding=0)
    colLimits = [[0.0, 0.0] for _ in collatedResults[1]]
    for (i, results_arr) in enumerate(collatedResults) for (j, result) in enumerate(results_arr)
        colLimits[j][1] = minimum((minimum(result), colLimits[j][1]))
        colLimits[j][2] = maximum((maximum(result), colLimits[j][2]))
    end end
    for (row, results_arr) in enumerate(collatedResults)
        Box(fig[2 * row - 1:2 * row + 1, 0:length(results_arr)+1], color=:white, strokecolor=:gray, strokewidth = 2.5)
        Label(fig[2 * row - 1, 1:length(results_arr)], subFigTitles[row], fontsize = 20, padding=(0, 0, -10, 10))
        for (col, result) in enumerate(results_arr)

            gl = GridLayout(fig[2 * row, col])
            reshaped_result = reshape(result, (length(axisVals[1]), length(axisVals[1])))
            ax = Axis(gl[1, 1], aspect=1.0, height=150, width=150, 
                                 xlabel=axisLabels[1], ylabel=axisLabels[2], 
                                 xlabelsize=16, ylabelsize=16
                                )
            scale=ifelse(all(x -> x==0, result), identity, Makie.pseudolog10)
            hm = heatmap!(ax, axisVals[1], axisVals[2], reshaped_result, colormap=cmap, colorscale=scale, colorrange=colLimits[col])
            Colorbar(gl[1, 2], limits=colLimits[col], scale=scale, colormap=cmap)
        end
    end
    for (i, text) in enumerate(figLabels)
        Label(fig[0, i], text, fontsize = 20, padding=(0, 0, -10, 10))
    end
    trim!(fig.layout)
    resize_to_layout!(fig)
    save(saveName, fig)
end

