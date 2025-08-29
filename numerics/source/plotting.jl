using CairoMakie, Makie, Random, Measures, LaTeXStrings, ColorSchemes

COLORS = vcat(ColorSchemes.Paired_12.colors[2:2:end], ColorSchemes.Paired_12.colors[1:2:end])

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
                        backgroundcolor=(:gray, 0.1),
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
        legendPos=(:top, :right),
    )

    if colorScale == log10
        matrixData[matrixData .== 0] .= minimum(matrixData[matrixData .> 0]) / 10
        matrixData[matrixData .< 0] .= NaN
    end
    if isnothing(colorbarLimits)
        colorbarLimits = ColorbarLimits(filter(!isnan, matrixData))
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
        axisVals::NTuple{2, Vector{Float64}},
        axisLabels::NTuple{2, Union{String, LaTeXString}},
        title::Union{String, LaTeXString},
        savename::String,
        colmap;
        figPad::Union{Number, NTuple{4, Number}}=0,
    )
    f = Figure(figure_padding=figPad)
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
        scatterLines::Union{Bool, Vector{Int64}}=false,
        scatter::Union{Bool, Vector{Int64}}=false,
        vlines::Any=[],
        hlines::Any=[],
        figSize::NTuple{2, Int64}=(300, 250),
        figPad::Union{Number, NTuple{4, Number}}=0,
        legendPos=:rt,
        linewidth::Number=3,
        xscale::Function=identity,
        yscale::Function=identity,
        plotRange::Vector{Int64}=Int64[],
        saveForm::String="pdf",
        splitLegends=nothing,
        twin::Vector{Int64}=Int64[],
        twinLabel::LaTeXString=L"",
        needsLegend::Bool=false,
        colormap::Any=:Paired_12,
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
    if typeof(scatter) == Bool
        scatter = scatter ? (1:length(nameValuePairs)) : Int64[]
    end
    if typeof(scatterLines) == Bool
        scatterLines = scatterLines ? (1:length(nameValuePairs)) : Int64[]
    end

    plots = []
    for (i, (name, yvalues)) in enumerate(nameValuePairs)
        if isempty(plotRange)
            plotRange = 1:length(yvalues)
        end
        if i ∈ scatter
            if !isempty(name)
                pl = CairoMakie.scatter!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; label=name, marker=markers[((i - 1) % 5) + 1], color=COLORS[mod1(i, length(COLORS))])
            else
                pl = CairoMakie.scatter!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; marker=markers[((i - 1) % 5) + 1], color=COLORS[mod1(i, length(COLORS))])
            end
        end
        if i ∈ scatterLines
            if !isempty(name)
                pl = scatterlines!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; label=name, marker=markers[((i - 1) % 5) + 1], color=COLORS[mod1(i, length(COLORS))])
            else
                pl = scatterlines!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; marker=markers[((i - 1) % 5) + 1], color=COLORS[mod1(i, length(COLORS))])
            end
        end
        if i ∉ scatter && i ∉ scatterLines
            if isempty(name)
                pl = lines!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; linewidth=linewidth, color=COLORS[mod1(i, length(COLORS))], linestyle=linestyles[i % 2 == 0 ? 2 : 1])
            else
                pl = lines!(i ∉ twin ? ax : ax_twin, xvalues[plotRange], yvalues[plotRange]; label=name, linewidth=linewidth, color=COLORS[mod1(i, length(COLORS))], linestyle=linestyles[i % 2 == 0 ? 2 : 1])
            end
        end
        push!(plots, pl)
    end

    if length(vlines) > 0
        for (i, (lab, loc)) in enumerate(vlines)
            push!(plots, vlines!(ax, loc, linestyle=:dash, linewidth=linewidth, label=lab, color=COLORS[mod1(length(nameValuePairs) + i, length(COLORS))]))
        end
    end
    if length(hlines) > 0
        for (i, (lab, loc)) in enumerate(hlines)
            push!(plots, hlines!(ax, loc, linestyle=:dash, linewidth=linewidth, label=lab, color=COLORS[mod1(length(nameValuePairs) + length(vlines) + i, length(COLORS))]))
        end
    end

    if needsLegend
        if isnothing(splitLegends)
            axislegend(position=legendPos)
        else
            for (inds, pos) in splitLegends
                axislegend(ax, plots[inds], vcat(nameValuePairs, vlines, hlines)[inds] .|> first, position=pos)
            end
        end
    end

    resize_to_layout!(f)

    saveName = saveName * "." * saveForm
    save(saveName, current_figure())
    return saveName
end


function plotLine(
        nameValuePairs::Union{Vector{Tuple{LaTeXString, Vector{Float64}}}, Vector{Tuple{String, Vector{Float64}}}},
        xvalues::Union{Vector{Int64}, Vector{Float64}},
        xlabel::Union{String, LaTeXString},
        ylabel::Union{String, LaTeXString},
        saveName::String;
        ylimits::Union{Nothing, NTuple{2, Float64}}=nothing,
        xlimits::Union{Nothing, NTuple{2, Float64}}=nothing,
        scatter::Bool=true,
        vlines::Any=[],
        hlines::Any=[],
        figSize::NTuple{2, Int64}=(150, 125),
        figPad::Union{AbsoluteLength, NTuple{4, AbsoluteLength}}=(-3.5mm, -1mm, -3.5mm, -1mm),
        legendPos=:best,
        xscale::Symbol=:identity,
        yscale::Symbol=:identity,
        needsLegend::Bool=true,
    )
    if typeof(figPad) == AbsoluteLength
        figPad = Tuple([figPad for _ in 1:4])
    end
    p = Plots.plot(size=figSize, left_margin=figPad[1], right_margin=figPad[2], bottom_margin=figPad[3], top_margin=figPad[4], xscale=xscale, yscale=yscale, legend=needsLegend, majorgrid=false, minorgrid=false, framestyle = :box, fontfamily="Computer Modern", thickness_scaling=1.2, palette=:Paired_12)
    Plots.xlabel!(p, xlabel)
    Plots.ylabel!(p, ylabel)
    if !isnothing(xlimits)
        Plots.xlims!(p, xlimits)
    end
    if !isnothing(ylimits)
        Plots.ylims!(p, ylimits)
    end

    for (i, (name, yvalues)) in enumerate(nameValuePairs)
        if scatter
            Plots.plot!(p, xvalues, yvalues, markersize=2.5, markershape=:circle, label=name, grid=false, legend_position=legendPos, dpi=10)
        else
            Plots.plot!(p, xvalues, yvalues, markersize=0, markershape=:circle, label=name, grid=false, legend_position=legendPos, dpi=10)
        end
    end

    if length(vlines) > 0
        for (lab, loc) in vlines
            Plots.vline!(p, [loc], label=lab, linestyle=:dash, legend_position=legendPos, dpi=10)
        end
    end
    if length(hlines) > 0
        for (lab, loc) in hlines
            Plots.hline!(p, [loc], label=lab, linestyle=:dash, legend_position=legendPos, dpi=10)
        end
    end

    Plots.savefig(p, saveName * ".pdf")
    return saveName * ".pdf"
end
