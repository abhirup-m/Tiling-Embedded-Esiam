using PyPlot

function plotHeatmaps(x_arr, collatedResults, saveName)
    fig, axes = PyPlot.subplots(size(collatedResults)...)
    for (i, results_arr) in enumerate(collatedResults)
        for (ax, result) in zip(axes[i], results_arr)
            reshaped_result = reshape(result, (length(x_arr), length(y_arr)))
            image = ax.imshow(reshaped_result, aspect=1, origin="lower")
            fig.colorbar(image, cax=ax)
        end
    end
end

