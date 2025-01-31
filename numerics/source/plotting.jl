using CairoMakie, Random, Measures, LaTeXStrings, ColorSchemes

const FONTSIZE = 28
# set_theme!(theme_latexfonts())
set_theme!(merge(theme_light(), theme_latexfonts()))
update_theme!(
              figure_padding = 0,
              fontsize=FONTSIZE,
              Axis = (
                leftspinevisible = true,
                bottomspinevisible = true,
                rightspinevisible = true,
                topspinevisible = true,
                spinewidth = 1.5,
                xticklabelsize = 24,
                yticklabelsize = 24,
                     ),
              ScatterLines = (
                       linewidth = 6,
                       markersize=20,
                      ),
              Lines = (
                       linewidth = 3,
                     ),
              Legend = (
                        patchsize=(50,20),
                        halign = :right,
                        valign = :top,
                       ),
             )


function plotHeatmap(
        matrixData::Union{Matrix{Int64}, Matrix{Float64}, Vector{Int64}, Vector{Float64}},
        axisVals::NTuple{2, Vector{Float64}},
        axisLabels::NTuple{2, Union{String, LaTeXString}},
        title::Union{String, LaTeXString},
        annotation::Union{String, LaTeXString},
        colmap; 
        colorbarLimits::Union{NTuple{2, Float64}, Nothing}=nothing,
        figSize::NTuple{2, Int64}=(300, 250),
        figPad::Union{NTuple{4, Float64}, Float64}=0.,
        colorScale::Function=identity,
    )
    matrixData = reshape(matrixData, length.(axisVals))

    if isnothing(colorbarLimits)
        nonNaNData = filter(!isnan, matrixData)
        if isempty(nonNaNData)
            colorbarLimits = (-1, 1)
        elseif minimum(nonNaNData) < maximum(nonNaNData)
            colorbarLimits = (minimum(nonNaNData), maximum(nonNaNData))
        else
            colorbarLimits = (minimum(nonNaNData)*(1-1e-5) - 1e-5, minimum(nonNaNData)*(1+1e-5) + 1e-5)
        end
    end
    figure = Figure(size=figSize, figure_padding=figPad)
    ax = Axis(figure[1, 1],
              xlabel = axisLabels[1], 
              ylabel=axisLabels[2], 
              title=title,
             )
    hm = heatmap!(ax, 
             axisVals..., 
             matrixData; 
             colormap=colmap,
             colorscale=colorScale,
             colorrange=colorbarLimits,
            )

    gl = GridLayout(figure[1, 1], tellwidth = false, tellheight = false, valign=:top, halign=:right)
    Box(gl[1, 1], color = RGBAf(0, 0, 0, 0.4), strokewidth=0, strokecolor=RGBAf(0, 0, 0, 0.8))
    Label(gl[1, 1], annotation, padding = (5, 5, 5, 5), fontsize=div(FONTSIZE, 1.3), color=:white)

    Colorbar(figure[1, 2], hm)
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


function plotLines(
        nameValuePairs::Dict{LaTeXString, Vector{Float64}},
        xvalues::Vector{Float64},
        xlabel::LaTeXString,
        ylabel::LaTeXString,
        saveName::String,
    )
    f = Figure(figure_padding=4)
    ax = Axis(f[1, 1],
        xlabel = xlabel,
        ylabel = ylabel,
    )
    linestyles = [:solid, (:dot, :dense), (:dash, :dense), (:dashdot, :dense), (:dashdotdot, :dense), (:dot, :loose)]
    for (i, (name, yvalues)) in enumerate(nameValuePairs)
        lines!(xvalues, yvalues; label=name, linestyle=linestyles[((i - 1) % 6) + 1])
    end
    axislegend()
    save(saveName, current_figure())
    return saveName
end
