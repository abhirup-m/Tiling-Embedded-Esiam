using LinearAlgebra
using CairoMakie
using Makie
using LaTeXStrings
include("../../../fermionise/source/fermionise.jl")
include("../../../fermionise/source/correlations.jl")
include("../../../fermionise/source/models.jl")

function scattProb(kondoJArray::Array{Float64,3}, stepIndex::Int64, size_BZ::Int64, dispersion::Vector{Float64}, fixedpointEnergy::Float64)
    results = zeros(size_BZ^2)
    results_bare = zeros(size_BZ^2)
    E_cloud = dispersion[-fixedpointEnergy.<=dispersion.<=fixedpointEnergy]
    point2_arr = unique(getIsoEngCont(dispersion, E_cloud))
    Threads.@threads for point1 in 1:size_BZ^2
        kx, ky = map1DTo2D(point1, size_BZ)
        if (abs(kx) ≈ K_MAX && abs(ky) ≈ K_MAX) || (kx ≈ 0 && ky ≈ 0)
            continue
        end
        results[point1] = sum(kondoJArray[point1, point2_arr, stepIndex] .^ 2) / length(point2_arr)
        results_bare[point1] = sum(kondoJArray[point1, point2_arr, 1] .^ 2) / length(point2_arr)
    end
    results_bool = tolerantSign.(abs.(results ./ results_bare), RG_RELEVANCE_TOL)
    results[results_bool.<0] .= NaN
    return results ./ results_bare, results_bare, results_bool
end


function kondoCoupMap(k_vals::Tuple{Float64,Float64}, size_BZ::Int64, kondoJArrayFull::Array{Float64,3})
    kx, ky = k_vals
    kspacePoint = map2DTo1D(kx, ky, size_BZ)
    results = kondoJArrayFull[kspacePoint, :, end] .^ 2
    results_bare = kondoJArrayFull[kspacePoint, :, 1] .^ 2
    results_bool = tolerantSign.(abs.(results), RG_RELEVANCE_TOL / size_BZ)
    results[results_bool.<0] .= NaN
    return results, results_bare, results_bool
end


function spinFlipCorrMap(size_BZ::Int64, dispersion::Vector{Float64}, kondoJArrayFull::Array{Float64,3}, W_val::Float64, orbitals::Tuple{String,String})
    halfSize = trunc(Int, (size_BZ + 1) / 2)
    kx_SEquadrant = range((K_MIN + K_MAX) / 2, stop=K_MAX, length=halfSize)
    ky_SEquadrant = range(K_MIN, stop=(K_MIN + K_MAX) / 2, length=halfSize)
    SEquadrant_kpairs = [(kx, ky) for ky in ky_SEquadrant for kx in kx_SEquadrant]

    results = zeros(halfSize^2)
    k_indices = collect(1:size_BZ^2)

    # operator list for the operator S_d^+ c^†_{k ↓} c_{k ↑} + h.c.
    spinFlipCorrOplist = [("+-+-", 1.0, [1, 2, 4, 3]), ("+-+-", 1.0, [2, 1, 3, 4])]

    trunc_dim = 5
    basis = BasisStates(trunc_dim * 2 + 2)
    bathIntFunc(points) = bathIntForm(W_val, orbitals[2], size_BZ, points)
    Threads.@threads for (kx, ky) in SEquadrant_kpairs
        if kx < ky
            continue
        end
        k_index = map2DTo1D(kx, ky, size_BZ)
        other_k_indices = k_indices[k_indices.≠k_index]
        chosenIndices = [[k_index]; other_k_indices[sortperm(abs.(kondoJArrayFull[k_index, other_k_indices, end]), rev=true)][1:trunc_dim-1]]
        oplist = KondoKSpace(chosenIndices, dispersion, 100 .* kondoJArrayFull[:, :, end], bathIntFunc)
        fixedPointHamMatrix = generalOperatorMatrix(basis, oplist)
        eigvals, eigstates = getSpectrum(fixedPointHamMatrix)
        results[SEquadrant_kpairs .== [(kx, ky)]] .= gstateCorrelation(basis, eigvals, eigstates, spinFlipCorrOplist)
        results[SEquadrant_kpairs .== [(ky, kx)]] .= results[SEquadrant_kpairs .== [(kx, ky)]]
    end
    results_bool = tolerantSign.(abs.(results), RG_RELEVANCE_TOL / 100)
    return results, results_bool, kx_SEquadrant, ky_SEquadrant
end


function plotHeatmaps(results_arr, x_arr, y_arr, fig, axes, cmaps)
    for (i, result) in enumerate(results_arr)
        reshaped_result = reshape(result, (length(x_arr), length(y_arr)))
        hmap = heatmap!(axes[i], x_arr, y_arr, reshaped_result, colormap=cmaps[i],
        )
        Colorbar(fig[1, 2*i], hmap, colorrange=(minimum(result), maximum(result)))
    end
    return axes
end


function mapProbeNameToProbe(probeName, size_BZ, kondoJArrayFull, W_by_J, J_val, dispersion, orbitals, fixedpointEnergy)
    titles = Vector{LaTeXString}(undef, 3)
    cmaps = [DISCRETE_CGRAD, :matter, :matter]
    if probeName == "scattProb"
        results, results_bare, results_bool = scattProb(kondoJArrayFull, size(kondoJArrayFull)[3], size_BZ, dispersion, fixedpointEnergy)
        titles[1] = L"\mathrm{rel(irrel)evance~of~}\Gamma(k)"
        titles[2] = L"\Gamma/\Gamma^{(0)}"
        titles[3] = L"\Gamma^{(0)}(k) = \sum_q J(k,q)^2"
        x_arr = y_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    elseif probeName == "kondoCoupNodeMap"
        node = (-pi / 2, -pi / 2)
        results, results_bare, results_bool = kondoCoupMap(node, size_BZ, kondoJArrayFull)
        titles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q_\mathrm{node})"
        titles[2] = L"J(k,q_\mathrm{node})"
        titles[3] = L"J^{(0)}(k,q_\mathrm{node})"
        drawPoint = (-0.5, -0.5)
        x_arr = y_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    elseif probeName == "kondoCoupAntinodeMap"
        antinode = (0.0, -pi)
        results, results_bare, results_bool = kondoCoupMap(antinode, size_BZ, kondoJArrayFull)
        titles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q_\mathrm{antin.})"
        titles[2] = L"J(k,q_\mathrm{antin.})"
        titles[3] = L"J^{(0)}(k,q_\mathrm{antin.})"
        drawPoint = (0, -1)
        x_arr = y_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    elseif probeName == "kondoCoupOffNodeMap"
        offnode = (-pi / 2 + 4 * pi / size_BZ, -pi / 2 + 4 * pi / size_BZ)
        results, results_bare, results_bool = kondoCoupMap(offnode, size_BZ, kondoJArrayFull)
        titles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q^\prime_\mathrm{node})"
        titles[2] = L"J(k,q^\prime_\mathrm{node})"
        titles[3] = L"J^{(0)}(k,q^\prime_{\mathrm{node}})"
        drawPoint = offnode ./ pi
        x_arr = y_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    elseif probeName == "kondoCoupOffAntinodeMap"
        offantinode = (0.0, -pi + 4 * pi / size_BZ)
        results, results_bare, results_bool = kondoCoupMap(offantinode, size_BZ, kondoJArrayFull)
        titles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q^\prime_\mathrm{antin.})"
        titles[2] = L"J(k,q^\prime_\mathrm{antin.})"
        titles[3] = L"J^{(0)}(k,q^\prime_{\mathrm{antin.}})"
        drawPoint = offantinode ./ pi
        x_arr = y_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    elseif probeName == "spinFlipCorrMap"
        results, results_bool, x_arr, y_arr = @time spinFlipCorrMap(size_BZ, dispersion, kondoJArrayFull, W_by_J * J_val, orbitals) 
        titles[1] = L"\mathrm{rel(irrel)evance~of~} "
        titles[2] = L"\langle S_d^+ c^\dagger_{k \downarrow}c_{k\uparrow} + \text{h.c.}\rangle"
    end
    fig = Figure()
    titlelayout = GridLayout(fig[0, 1:4])
    Label(titlelayout[1, 1:4], L"NW/J=%$(trunc(size_BZ * W_by_J, digits=1))", justification=:center, padding=(0, 0, -20, 0))
    axes = [Axis(fig[1, 2*i-1], xlabel=L"\mathrm{k_x}", ylabel=L"\mathrm{k_y}", title=title) for (i, title) in enumerate(titles[1:2])]
    axes = plotHeatmaps((results_bool, results), x_arr, y_arr,
        fig, axes[1:2], cmaps)
    if probeName in ["kondoCoupNodeMap", "kondoCoupAntinodeMap", "kondoCoupOffNodeMap", "kondoCoupOffAntinodeMap"]
        [scatter!(ax, [drawPoint[1]], [drawPoint[2]], markersize=20, color=:grey, strokewidth=2, strokecolor=:white) for ax in axes]
    end
    [colsize!(fig.layout, i, Aspect(1, 1.0)) for i in [1, 3]]
    resize_to_layout!(fig)
    return fig
end
