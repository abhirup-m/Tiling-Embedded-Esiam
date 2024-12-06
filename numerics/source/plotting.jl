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
                        backgroundcolor=:white,
                        valign = :top,
                       ),
             )

# colmap = ColorSchemes.cherry
function plotHeatmap(
        matrixData::Union{Matrix{Int64}, Matrix{Float64}},
        axisVals::NTuple{2, Vector{Float64}},
        axisLabels::NTuple{2, Union{String, LaTeXString}},
        title::Union{String, LaTeXString},
        annotation::Union{String, LaTeXString},
        colmap,
        figSize::NTuple{2, Int64}=(300, 250),
    )
    figure = Figure(size=figSize)
    ax = Axis(figure[1, 1],
              xlabel = axisLabels[1], 
              ylabel=axisLabels[2], 
              title=title,
             )
    heatmap!(ax, 
             axisVals..., 
             matrixData; 
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
    savename = joinpath("raw_figures", randstring(5) * ".pdf")
    save(savename, figure)
    return savename
end


function plotHeatmap(
        matrixData::Union{Vector{Int64}, Vector{Float64}},
        axisVals::NTuple{2, Vector{Float64}},
        axisLabels::NTuple{2, Union{String, LaTeXString}},
        title::Union{String, LaTeXString},
        annotation::Union{String, LaTeXString},
        colmap,
        figSize::NTuple{2, Int64}=(300, 250),
    )
    matrixData = reshape(matrixData, length.(axisVals))
    return plotHeatmap(matrixData, axisVals, axisLabels, title, annotation, colmap, figSize=figSize)
end


function plotPhaseDiagram(
        matrixData::Matrix{Int64},
        legend::Dict{Int64, String},
        axisVals::NTuple{2, Vector{Float64}},
        axisLabels::NTuple{2, Union{String, LaTeXString}},
        title::Union{String, LaTeXString},
        savename::String,
        colmap,
        figSize::NTuple{2, Int64}=(400, 300),
    )
    figure = Figure(size=figSize, figure_padding = 4)
    ax = Axis(figure[1, 1],
              xlabel = axisLabels[1], 
              ylabel=axisLabels[2], 
              titlegap = 0,
             )
    colmap = [colmap[div(i * length(colmap), length(unique(matrixData)))] for i in matrixData |> unique |> sort]
    legendFlags = Dict(v => true for v in matrixData |> unique)
    for ri in axes(matrixData, 1)
        for ci in axes(matrixData, 2)
            val = matrixData[ri,ci]
            if legendFlags[val]
                scatter!(ax, axisVals[1][ci], axisVals[2][ri], color=colmap[val], marker=:rect, label=legend[val]=>(; markersize=15), markersize=5)
                legendFlags[matrixData[ri,ci]] = false
            else
                scatter!(ax, axisVals[1][ci], axisVals[2][ri], color=colmap[val], markersize=5)
            end
        end
    end
    figure[0, 1] = axislegend(ax, orientation=:horizontal, margin=(0., 0., -10., 0.), patchcolor=:transparent, patchlabelgap=-5)
    save(savename, figure)
end


function plotSpecFunc(
        specFuncArr::Dict{LaTeXString, Vector{Float64}},
        freqValues::Vector{Float64},
        saveName::String,
        title::LaTeXString,
    )
    f = Figure()
    ax = Axis(f[1, 1],
        title = title,
        xlabel = L"frequence~($\omega$)",
        ylabel = L"impurity spectral function~$A(\omega)$",
    )
    markers = [:circle, :rect, :diamond, :xcross, :star4]
    for (i, (label, specFunc)) in enumerate(specFuncArr)
        scatter!(freqValues, specFunc, label=label, marker=markers[i])
    end
    axislegend()
    save(saveName, current_figure())
    return nothing
end
