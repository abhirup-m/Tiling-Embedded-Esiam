using PyPlot
using LaTeXStrings
const plt = PyPlot

function plotHeatmaps(axisVals, axisLabels, figLabels, collatedResults, saveName)
    nx, ny = length(collatedResults), length(collatedResults[1])
    fig, axes = PyPlot.subplots(nrows=nx, ncols=ny, squeeze=false, figsize=(2.4 * nx, 4.8 * ny), layout="compressed")
    fw, fh = fig.get_size_inches()
    for (i, results_arr) in enumerate(collatedResults)
        for (ax, result) in zip(axes[i, :], results_arr)
            reshaped_result = reshape(result, (length(axisVals[1]), length(axisVals[1])))
            img = ax.imshow(reshaped_result, origin="lower", extent=(axisVals[1][1], axisVals[1][end],axisVals[2][1], axisVals[2][end]));
            ax.set_xlabel(axisLabels[1])
            ax.set_ylabel(axisLabels[2])
            fig.colorbar(img);
        end
    end
    axes[1,1].set_title(figLabels[1])
    axes[1,2].set_title(figLabels[2])
    fig.savefig(saveName, bbox_inches="tight")
end

