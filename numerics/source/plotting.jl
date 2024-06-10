import PythonPlot
const plt = PythonPlot
using LaTeXStrings
using Measures
# DISCRETE_CGRAD = cgrad(:seaborn_rocket_gradient, 2, categorical=true)

function plotHeatmaps(axes, results_arr, x_arr, y_arr)
    for (ax, result) in zip(axes, results_arr)
        reshaped_result = reshape(result, (length(x_arr), length(y_arr)))
        image = ax.imshow(reshaped_result, aspect=1, origin="lower")
        fig.colorbar(image, cax=ax)
    end
    return fig, axes
end
