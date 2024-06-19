using CairoMakie
using LaTeXStrings

function plotHeatmaps(axisVals, axisLabels, figLabels, subFigTitles, collatedResults, saveName)
    fig = Figure(dpi=600, figure_padding=0)
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
            if all(isnan.(reshaped_result)) || all(isinf.(reshaped_result))
                reshaped_result .= 0
            end
            hm = heatmap!(ax, axisVals[1], axisVals[2], reshaped_result)
            Colorbar(gl[1, 2], hm)
        end
    end
    for (i, text) in enumerate(figLabels)
        Label(fig[0, i], text, fontsize = 20)
    end
    # for (i, text) in enumerate(subFigTitles)
    #     Label(fig[i, 0], text, fontsize = 20)
    # end
    trim!(fig.layout)
    resize_to_layout!(fig)
    save(saveName, fig)
end

