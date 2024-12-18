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
              Lines = (
                       linewidth = 4,
                     ),
              Legend = (
                        patchsize=(50,20),
                        halign = :right,
                        valign = :top,
                       ),
             )

# colmap = ColorSchemes.cherry
function plotHeatmap(
        matrixData::Union{Matrix{Int64}, Matrix{Float64}, Vector{Int64}, Vector{Float64}},
        axisVals::NTuple{2, Vector{Float64}},
        axisLabels::NTuple{2, Union{String, LaTeXString}},
        title::Union{String, LaTeXString},
        annotation::Union{String, LaTeXString},
        colmap;
        figSize::NTuple{2, Int64}=(300, 250),
        figPad::Union{NTuple{4, Float64}, Float64}=0.,
        colorScale::Function=identity,
    )
    matrixData = reshape(matrixData, length.(axisVals))
    figure = Figure(size=figSize, figure_padding=figPad)
    ax = Axis(figure[1, 1],
              xlabel = axisLabels[1], 
              ylabel=axisLabels[2], 
              title=title,
             )
    heatmap!(ax, 
             axisVals..., 
             matrixData; 
             colormap=colmap,
             colorscale=colorScale,
            )

    fontsize = 28
    gl = GridLayout(figure[1, 1], tellwidth = false, tellheight = false, valign=:top, halign=:right)
    Box(gl[1, 1], color = RGBAf(0, 0, 0, 0.4), strokewidth=0, strokecolor=RGBAf(0, 0, 0, 0.8))
    Label(gl[1, 1], annotation, padding = (5, 5, 5, 5), fontsize=div(fontsize, 1.6), color=:white)

    if minimum(matrixData) < maximum(matrixData)
        colorbarLimits = (minimum(matrixData), maximum(matrixData))
    else
        colorbarLimits = (minimum(matrixData)*(1-1e-5) - 1e-5, minimum(matrixData)*(1+1e-5) + 1e-5)
    end
    Colorbar(figure[1, 2]; 
             limits=colorbarLimits,
             colormap=colmap,
            )
    savename = joinpath("raw_figures", randstring(5) * ".pdf")
    save(savename, figure)
    return savename
end


function plotPhaseDiagram(
        matrixData::Matrix{Int64},
        legend::Dict{Int64, String},
        axisVals::NTuple{2, Vector{Float64}},
        axisLabels::NTuple{2, Union{String, LaTeXString}},
        title::Union{String, LaTeXString},
        savename::String,
        colmap,
    )
    figure = Figure(size=(350, 300), figure_padding=(0, 8, 4, 8))
    ax = Axis(figure[1, 1],
              xlabel = axisLabels[1], 
              ylabel=axisLabels[2], 
              xticklabelsize=22,
              yticklabelsize=22,
              xscale=log10,
             )

    for v in unique(matrixData)
        scatter!(ax, [0.5 * (axisVals[1][1] + axisVals[1][end])], [0.5 * (axisVals[2][1] + axisVals[2][end])], color=colmap[v], marker=:rect, label=legend[v]=>(; markersize=15))
    end
    heatmap!(ax, 
             axisVals..., 
             matrixData; 
             colormap=colmap,
            )
    figure[0, 1] = axislegend(ax, orientation=:horizontal, margin=(0., 0., -10., -10.), patchcolor=:transparent, patchlabelgap=-10, colgap=0)
    save(savename, figure)
end


function plotSpecFunc(
        specFuncArr::Vector{Tuple{LaTeXString, Vector{Float64}}},
        freqValues::Vector{Float64},
        saveName::String,
    )
    f = Figure(figure_padding=4)
    ax = Axis(f[1, 1],
        xlabel = L"frequence ($\omega$)",
        ylabel = L"impurity spectral function $A(\omega)$",
    )
    linestyles = [:solid, (:dot, :dense), (:dash, :dense), (:dashdot, :dense), (:dashdotdot, :dense), (:dot, :loose)]
    for (i, (label, specFunc)) in enumerate(specFuncArr)
        lines!(freqValues, specFunc; label=label, linestyle=linestyles[((i - 1) % 6) + 1])
    end
    axislegend()
    save(saveName, current_figure())
    return nothing
end
