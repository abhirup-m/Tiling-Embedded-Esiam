using Plots
using LaTeXStrings
using Measures
(@isdefined DISCRETE_CGRAD) || const DISCRETE_CGRAD = cgrad(:BuPu_3, 3, categorical=true)

function plotHeatmaps(results_arr, x_arr, y_arr, cmaps, subtitles)
    plots = []
    for (i, result) in enumerate(results_arr)
        reshaped_result = reshape(result, (length(x_arr), length(y_arr)))
        hmap = heatmap(x_arr, y_arr, reshaped_result, xlabel=L"$ak_x/\pi$", ylabel=L"$ak_y/\pi$", c=cmaps[i], title=subtitles[i], top_margin=3mm, left_margin=5mm, bottom_margin=5mm
        )
        push!(plots, hmap)
    end
    return plots
end


function mainPlotter(results_arr::Vector{Vector{Float64}}, probeName::String, size_BZ::Int64, titleText::LaTeXString)
    subtitles = Vector{LaTeXString}(undef, length(results_arr))
    cmaps = [DISCRETE_CGRAD, :matter, :matter]
    if probeName == "scattProb"
        subtitles[1] = L"\mathrm{rel(irrel)evance~of~}\Gamma(k)"
        subtitles[2] = L"\Gamma(k)/\Gamma^{(0)}(k)"
    elseif probeName == "kondoCoupNodeMap"
        subtitles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q_\mathrm{node})"
        subtitles[2] = L"J(k,q_\mathrm{node})"
        drawPoint = (-0.5, -0.5)
    elseif probeName == "kondoCoupAntinodeMap"
        subtitles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q_\mathrm{antin.})"
        subtitles[2] = L"J(k,q_\mathrm{antin.})"
        drawPoint = (0, -1)
    elseif probeName == "kondoCoupOffNodeMap"
        offnode = (-pi / 2 + 4 * pi / size_BZ, -pi / 2 + 4 * pi / size_BZ)
        subtitles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q^\prime_\mathrm{node})"
        subtitles[2] = L"J(k,q^\prime_\mathrm{node})"
        drawPoint = offnode ./ pi
    elseif probeName == "kondoCoupOffAntinodeMap"
        offantinode = (0.0, -pi + 4 * pi / size_BZ)
        subtitles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q^\prime_\mathrm{antin.})"
        subtitles[2] = L"J(k,q^\prime_\mathrm{antin.})"
        drawPoint = offantinode ./ pi
    elseif probeName == "spinFlipCorrMap"
        subtitles[1] = L"\mathrm{rel(irrel)evance~of~} "
        subtitles[2] = L"0.5\langle S_d^+ c^\dagger_{k \downarrow}c_{k\uparrow} + \text{h.c.}\rangle"
    elseif probeName == "tiledspinFlipCorrMap"
        subtitles[1] = L"\chi(k)"
        subtitles[2] = L"\chi(k_{N}, k)"
        subtitles[3] = L"\chi(k_{AN}, k)"
        cmaps = [:matter, :matter, :matter]
    end
    x_arr = y_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    plots = plotHeatmaps(results_arr, x_arr, y_arr, cmaps, subtitles)
    fig = plot(plots..., size=(900, 220), layout = grid(1, length(results_arr)), plot_title=titleText, plot_titlevspan=0.1)
    return fig
end
