using CairoMakie, Random, Measures, LaTeXStrings, ColorSchemes

const FONTSIZE = 28
set_theme!(theme_latexfonts())
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
                xgridvisible=false,
                ygridvisible=false,
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
                        labelsize = 18,
                        backgroundcolor=(:gray, 0.2),
                        framecolor=:transparent,
                       ),
             )


function ColorbarLimits(
        quadrantResults::Vector{Float64},
    )
    nonNaNData = filter(!isnan, quadrantResults)
    if minimum(nonNaNData) < maximum(nonNaNData)
        colorbarLimits = (minimum(nonNaNData), maximum(nonNaNData))
    else
        colorbarLimits = (minimum(nonNaNData)*(1-1e-5) - 1e-5, minimum(nonNaNData)*(1+1e-5) + 1e-5)
    end
    return colorbarLimits
end


function ColorbarLimits(
        quadrantResults::Dict{Float64, Vector{Float64}}
    )
    gatheredResults = vcat(values(quadrantResults)...)
    return ColorbarLimits(gatheredResults)
end


function plotHeatmap(
        matrixData::Union{Matrix{Int64}, Matrix{Float64}, Vector{Int64}, Vector{Float64}},
        axisVals::NTuple{2, Vector{Float64}},
        axisLabels::NTuple{2, Union{String, LaTeXString}},
        title::Union{String, LaTeXString},
        annotation::Union{String, LaTeXString},
        colmap; 
        colorbarLimits::Union{NTuple{2, Float64}, Nothing}=nothing,
        figSize::NTuple{2, Int64}=(200, 200),
        figPad::Union{NTuple{4, Float64}, Float64}=0.,
        colorScale::Function=identity,
        marker::Union{Nothing, Vector{Float64}}=nothing,
        line::Vector{NTuple{2, Number}}=NTuple{2, Number}[],
        legendPos::NTuple{2, Symbol}=(:top, :right),
    )

    if isnothing(colorbarLimits)
        colorbarLimits = ColorbarLimits(matrixData)
    end
    matrixData = reshape(matrixData, length.(axisVals))
    figure = Figure(figure_padding=figPad)
    ax = Axis(figure[1, 1],
              xlabel = axisLabels[1], 
              ylabel=axisLabels[2], 
              title=title,
              width=figSize[1],
              height=figSize[2],
             )
    hm = heatmap!(ax, 
             axisVals..., 
             matrixData; 
             colormap=colmap,
             colorscale=colorScale,
             colorrange=colorbarLimits,
            )
    gl = GridLayout(figure[1, 1], tellwidth = false, tellheight = false, valign=legendPos[1], halign=legendPos[2])
    Box(gl[1, 1], color = RGBAf(0, 0, 0, 0.4), strokewidth=0, strokecolor=RGBAf(0, 0, 0, 0.8))
    Label(gl[1, 1], annotation, padding = (5, 5, 5, 5), fontsize=div(FONTSIZE, 1.3), color=:white)

    if !isempty(line)
        scatter!(ax, line .|> first, line .|> last, color = :maroon, markersize=4)
    end
    if !isnothing(marker)
        scatter!(ax, marker..., color=:gray, markersize=20)
    end

    Colorbar(figure[1, 2], hm)
    resize_to_layout!(figure)
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
    figure = Figure(figure_padding=(0, 30, 5, 10))
    ax = Axis(figure[1, 1],
              xlabel = axisLabels[1], 
              ylabel=axisLabels[2], 
              xticklabelsize=22,
              yticklabelsize=22,
              xscale=log10,
              width=250,
              height=200,
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
    resize_to_layout!(figure)
    save(savename, figure)
end


function plotLines(
        nameValuePairs::Union{Vector{Tuple{LaTeXString, Vector{Float64}}}, Vector{Tuple{String, Vector{Float64}}}},
        xvalues::Union{Vector{Int64}, Vector{Float64}},
        xlabel::LaTeXString,
        ylabel::LaTeXString,
        saveName::String;
        ylimits::Union{Nothing, NTuple{2, Float64}}=nothing,
        xlimits::Union{Nothing, NTuple{2, Float64}}=nothing,
        scatter::Bool=false,
        vlines::Vector{Tuple{LaTeXString, Float64}}=Tuple{LaTeXString, Float64}[],
        hlines::Vector{Tuple{LaTeXString, Float64}}=Tuple{LaTeXString, Float64}[],
        figSize::NTuple{2, Int64}=(300, 250),
        figPad::Union{Number, NTuple{4, Number}}=0,
        legendPos::Union{Symbol, NTuple{2, Float64}}=:rt,
        linewidth::Number=4,
        xscale::Function=identity,
        yscale::Function=identity,
        plotRange::Vector{Int64}=Int64[],
        saveForm::String="pdf",
        splitLegends=nothing,
        twin::Vector{Int64}=Int64[],
        twinLabel::LaTeXString=L"",
    )

    f = Figure(figure_padding=figPad)
    ax = Axis(f[1, 1],
        xlabel = xlabel,
        ylabel = ylabel,
        width=figSize[1],
        height=figSize[2],
        xscale=xscale,
        yscale=yscale,
        limits=(xlimits, ylimits),
    )
    if !isempty(twin)
        ax_twin = Axis(f[1, 1],
            xlabel = xlabel,
            ylabel = twinLabel,
            width=figSize[1],
            height=figSize[2],
            xscale=xscale,
            yscale=yscale,
            limits=(xlimits, ylimits),
            yaxisposition = :right,
        )
        hidespines!(ax_twin)
        hidexdecorations!(ax_twin)
    end
    needsLegend = false
    linestyles = [:solid, (:dot, :dense), (:dash, :dense), (:dashdot, :dense), (:dashdotdot, :dense), (:dot, :loose)]
    markers = [:circle, :rect, :diamond, :hexagon, :xcross]

    plots = []
    for (i, (name, yvalues)) in enumerate(nameValuePairs)
        if isempty(plotRange)
            plotRange = 1:length(yvalues)
        end
        if !scatter
            if isempty(name)
                pl = lines!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; linestyle=linestyles[((i - 1) % 6) + 1], linewidth=linewidth)
            else
                needsLegend = true
                pl = lines!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; label=name, linestyle=linestyles[((i - 1) % 6) + 1], linewidth=linewidth)
            end
        else
            if !isempty(name)
                needsLegend = true
                pl = scatter!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; label=name, marker=markers[((i - 1) % 5) + 1])
            else
                pl = scatter!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; marker=markers[((i - 1) % 5) + 1])
            end
        end
        push!(plots, pl)
    end

    if length(vlines) + length(hlines) > 0
        lineColors = cgrad(:Dark2_8, length(vlines) + length(hlines); categorical=true)
        for (i, (label, xloc)) in enumerate(vlines)
            if isempty(label)
                vlines!(ax, xloc, color=lineColors[i], linestyle=linestyles[i])
            else
                vlines!(ax, xloc, label=label, color=lineColors[i], linestyle=linestyles[i])
                needsLegend = true
            end
        end
        for (i, (label, yloc)) in enumerate(hlines)
            if isempty(label)
                hlines!(ax, yloc, color=lineColors[length(vlines) + i], linestyle=linestyles[i])
            else
                hlines!(ax, yloc, label=label, color=lineColors[length(vlines) + i], linestyle=linestyles[i])
                needsLegend = true
            end
        end
    end

    if needsLegend && length(nameValuePairs) > 1
        if isnothing(splitLegends)
            axislegend(position=legendPos)
        else
            for (inds, pos) in splitLegends
                axislegend(ax, plots[inds], nameValuePairs[inds] .|> first, position=pos)
            end
        end
    end

    resize_to_layout!(f)

    saveName = saveName * "." * saveForm
    save(saveName, current_figure())
    return saveName
end
