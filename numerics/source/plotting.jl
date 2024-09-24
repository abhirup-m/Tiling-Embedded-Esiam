using CairoMakie, Random, Measures, LaTeXStrings
set_theme!(merge(theme_ggplot2(), theme_latexfonts()))
const cmap = :cherry
function plotHeatmap(
        matrixData::Vector{Float64},
        axisVals::NTuple{2, Vector{Float64}},
        axisLabels::NTuple{2, LaTeXString},
        title::LaTeXString,
    )
    matrix = reshape(matrixData, length.(axisVals))

    figure = Figure(size = (500, 400), 
                    fontsize = 26, 
                   )
    ax = Axis(figure[1, 1],
              xlabel = axisLabels[1], 
              ylabel=axisLabels[2], 
              title=title,
             )
    heatmap!(ax, 
             axisVals..., 
             matrix; 
             colormap=cmap,
            )

    colorbarLimits = ifelse(minimum(matrixData) < maximum(matrixData), (minimum(matrixData), maximum(matrixData)), (-1, 1))
    Colorbar(figure[:, end+1]; 
             limits=colorbarLimits,
             colormap=cmap,
            )

    savename = joinpath("figures", randstring(5) * ".pdf")
    save(savename, figure)
    return savename
end

