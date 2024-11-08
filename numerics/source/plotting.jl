using CairoMakie, Random, Measures, LaTeXStrings, ColorSchemes

# set_theme!(theme_latexfonts())
set_theme!(merge(theme_ggplot2(), theme_latexfonts()))
update_theme!(
              figure_padding = 0,
              fontsize=28,
              Axis = (
                leftspinevisible = true,
                bottomspinevisible = true,
                rightspinevisible = true,
                topspinevisible = true,
                spinewidth = 1.5,
                     ),
              ScatterLines = (
                       linewidth = 6,
                       markersize=20,
                      ),
              Legend = (
                        patchsize=(50,20),
                        halign = :right,
                        valign = :top,
                       ),
             )

# colmap = ColorSchemes.cherry
function plotHeatmap(
        matrixData::Union{Vector{Int64}, Vector{Float64}},
        axisVals::NTuple{2, Vector{Float64}},
        axisLabels::NTuple{2, LaTeXString},
        title::LaTeXString,
        annotation::LaTeXString,
        colmap,
    )
    matrix = reshape(matrixData, length.(axisVals))

    figure = Figure(size=(300, 250))
    ax = Axis(figure[1, 1],
              xlabel = axisLabels[1], 
              ylabel=axisLabels[2], 
              title=title,
             )
    heatmap!(ax, 
             axisVals..., 
             matrix; 
             colormap=colmap,
            )

    fontsize = 28
    gl = GridLayout(figure[1, 1], tellwidth = false, tellheight = false, valign=:top, halign=:right)
    Box(gl[1, 1], color = RGBAf(0, 0, 0, 0.4), strokewidth=0, strokecolor=RGBAf(0, 0, 0, 0.8))
    Label(gl[1, 1], annotation, padding = (5, 5, 5, 5), fontsize=div(fontsize, 1.3), color=:white)

    colorbarLimits = ifelse(minimum(matrixData) < maximum(matrixData), (minimum(matrixData), maximum(matrixData)), (minimum(matrixData)-1e-10, minimum(matrixData)+1e-10))
    Colorbar(figure[1, 2]; 
             limits=colorbarLimits,
             colormap=colmap,
            )
    savename = joinpath("figures", randstring(5) * ".pdf")
    save(savename, figure)
    return savename
end

