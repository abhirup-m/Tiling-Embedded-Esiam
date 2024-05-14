using CairoMakie
using Makie
using LaTeXStrings
set_theme!(theme_latexfonts())
update_theme!(fontsize=30)


function plotHeatmaps(results_arr, x_arr, y_arr, fig, axes, cmaps)
    for (i, result) in enumerate(results_arr)
        reshaped_result = reshape(result, (length(x_arr), length(y_arr)))
        hmap = CairoMakie.heatmap!(axes[i], x_arr, y_arr, reshaped_result, colormap=cmaps[i],
        )
        Colorbar(fig[1, 2*i], hmap, colorrange=(minimum(result), maximum(result)))
    end
    return axes
end


function mainPlotter(results::Vector{Float64}, results_bool::Vector{Int64}, probeName::String, size_BZ::Int64, titleText::LaTeXString)
    titles = Vector{LaTeXString}(undef, 2)
    cmaps = [DISCRETE_CGRAD, :matter, :matter]
    if probeName == "scattProb"
        titles[1] = L"\mathrm{rel(irrel)evance~of~}\Gamma(k)"
        titles[2] = L"\Gamma(k)/\Gamma^{(0)}(k)"
    elseif probeName == "kondoCoupNodeMap"
        titles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q_\mathrm{node})"
        titles[2] = L"J(k,q_\mathrm{node})"
        drawPoint = (-0.5, -0.5)
    elseif probeName == "kondoCoupAntinodeMap"
        titles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q_\mathrm{antin.})"
        titles[2] = L"J(k,q_\mathrm{antin.})"
        drawPoint = (0, -1)
    elseif probeName == "kondoCoupOffNodeMap"
        offnode = (-pi / 2 + 4 * pi / size_BZ, -pi / 2 + 4 * pi / size_BZ)
        titles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q^\prime_\mathrm{node})"
        titles[2] = L"J(k,q^\prime_\mathrm{node})"
        drawPoint = offnode ./ pi
    elseif probeName == "kondoCoupOffAntinodeMap"
        offantinode = (0.0, -pi + 4 * pi / size_BZ)
        titles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q^\prime_\mathrm{antin.})"
        titles[2] = L"J(k,q^\prime_\mathrm{antin.})"
        drawPoint = offantinode ./ pi
    elseif probeName == "spinFlipCorrMap"
        titles[1] = L"\mathrm{rel(irrel)evance~of~} "
        titles[2] = L"0.5\langle S_d^+ c^\dagger_{k \downarrow}c_{k\uparrow} + \text{h.c.}\rangle"
    end
    x_arr = y_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    fig = Figure()
    titlelayout = GridLayout(fig[0, 1:4])
    Label(titlelayout[1, 1:4], titleText, justification=:center, padding=(0, 0, -20, 0))
    axes = [Axis(fig[1, 2*i-1], xlabel=L"\mathrm{k_x}", ylabel=L"\mathrm{k_y}", title=title) for (i, title) in enumerate(titles[1:2])]
    axes = plotHeatmaps((results_bool, results), x_arr, y_arr,
        fig, axes[1:2], cmaps)
    if probeName in ["kondoCoupNodeMap", "kondoCoupAntinodeMap", "kondoCoupOffNodeMap", "kondoCoupOffAntinodeMap"]
        [CairoMakie.scatter!(ax, [drawPoint[1]], [drawPoint[2]], markersize=20, color=:grey, strokewidth=2, strokecolor=:white) for ax in axes]
    end
    [colsize!(fig.layout, i, Aspect(1, 1.0)) for i in [1, 3]]
    resize_to_layout!(fig)
    return fig
end
