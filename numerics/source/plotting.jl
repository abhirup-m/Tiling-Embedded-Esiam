using CairoMakie, Random, Measures, LaTeXStrings, ColorSchemes

const FONTSIZE = 28
set_theme!(merge(theme_minimal(), theme_latexfonts()))
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
                       linewidth = 3,
                       markersize=14,
                       alpha=1.,
                      ),
              Scatter = (
                         alpha=1.,
                        ),
              Lines = (
                       linewidth = 3,
                       alpha=1.,
                     ),
              Legend = (
                        patchsize=(30,20),
                        halign = :right,
                        valign = :top,
                        labelsize = 14,
                        backgroundcolor=(:gray, 0.2),
                        rowgap=-5,
                        framevisible=true,
                        framewidth=0,
                        padding=5,
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
        marker::Union{Nothing, Tuple{Vector{Float64}, Symbol}}=nothing,
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
        lines!(ax, line .|> first, line .|> last, color = :black, linestyle=:dash)
    end
    if !isnothing(marker)
        scatter!(ax, marker[1]..., color=marker[2], markersize=20)
    end

    Colorbar(figure[1, 2], hm)
    resize_to_layout!(figure)
    savename = joinpath("raw_figures", randstring(5) * ".pdf")
    save(savename, figure)
    return savename
end


function plotPhaseDiagram(
        matrixData::Matrix{Float64},
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
              width=250,
              height=200,
             )

    hm = heatmap!(ax, 
             axisVals..., 
             matrixData; 
             colormap=colmap,
            )
    Colorbar(figure[:, end+1], hm)
    #=figure[0, 1] = axislegend(ax, orientation=:horizontal, margin=(0., 0., -10., -10.), patchcolor=:transparent, patchlabelgap=-10, colgap=0)=#
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
        scatterLines::Vector{Int64}=Int64[],
        scatter::Vector{Int64}=Int64[],
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
        needsLegend::Bool=false,
        colormap::Any=ColorSchemes.Paired_12,
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
    linestyles = [:solid, (:dot, :dense), (:dash, :dense), (:dashdot, :dense), (:dashdotdot, :dense), (:dot, :loose)]
    markers = [:circle, :rect, :diamond, :hexagon, :xcross]

    plots = []
    for (i, (name, yvalues)) in enumerate(nameValuePairs)
        if isempty(plotRange)
            plotRange = 1:length(yvalues)
        end
        if i ∈ scatter
            if !isempty(name)
                pl = scatter!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; label=name, marker=markers[((i - 1) % 5) + 1], color=colormap[i])
            else
                pl = scatter!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; marker=markers[((i - 1) % 5) + 1], color=colormap[i])
            end
        end
        if i ∈ scatterLines
            if isempty(name)
                pl = scatterlines!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; label=name, marker=markers[((i - 1) % 5) + 1], color=colormap[i])
            else
                pl = scatterlines!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; marker=markers[((i - 1) % 5) + 1], color=colormap[i])
            end
        end
        if i ∉ scatter && i ∉ scatterLines
            if isempty(name)
                pl = lines!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; linestyle=linestyles[((i - 1) % 6) + 1], linewidth=linewidth, color=colormap[i])
            else
                pl = lines!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; label=name, linestyle=linestyles[((i - 1) % 6) + 1], linewidth=linewidth, color=colormap[i])
            end
        end
        push!(plots, pl)
    end

    if length(vlines) + length(hlines) > 0
        vlines!(ax, [loc for (_, loc) in vlines], linestyle=:dash, label=[lab for (lab, _) in vlines], colormap=:balance, color=[i - 1 for i in eachindex(vlines)])
        hlines!(ax, [loc for (_, loc) in hlines], linestyle=:dash, label=[lab for (lab, _) in hlines], colormap=:balance, color=[length(vlines) + i - 1 for i in eachindex(hlines)])
        #=for (i, (label, xloc)) in enumerate(vlines)=#
        #=    if isempty(label)=#
        #=        vlines!(ax, xloc, colormap=ColorSchemes.coolwarm, color=i:i, linestyle=:dash)=#
        #=    else=#
        #=        vlines!(ax, xloc, label=label, colormap=ColorSchemes.coolwarm, color=i:i, linestyle=:dash)=#
        #=        needsLegend = true=#
        #=    end=#
        #=end=#
        #=for (i, (label, yloc)) in enumerate(hlines)=#
        #=    if isempty(label)=#
        #=        hlines!(ax, yloc, colormap=ColorSchemes.coolwarm, color=length(vlines) + i, linestyle=:dash)=#
        #=    else=#
        #=        hlines!(ax, yloc, label=label, colormap=ColorSchemes.coolwarm, color=length(vlines) + i, linestyle=:dash)=#
        #=        needsLegend = true=#
        #=    end=#
        #=end=#
    end

    if needsLegend
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
