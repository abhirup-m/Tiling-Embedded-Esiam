using LinearAlgebra
using CairoMakie
using Makie
@everywhere include("../../../fermionise/source/fermionise.jl")
@everywhere include("../../../fermionise/source/models.jl")

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


function kondoCoupMap(k_vals, size_BZ, kondoJArrayFull)
    kx, ky = k_vals
    kspacePoint = map2DTo1D(kx, ky, size_BZ)
    results = kondoJArrayFull[kspacePoint, :, end] .^ 2
    results_bare = kondoJArrayFull[kspacePoint, :, 1] .^ 2
    results_bool = tolerantSign.(abs.(results), RG_RELEVANCE_TOL / size_BZ)
    results[results_bool.<0] .= NaN
    return results, results_bare, results_bool
end


function spinFlipCorrMap(k_index::Int64, size_BZ::Int64, kondoJArrayFull, W_val::Float64, orbitals::Tuple{String,String})
    k_indices = collect(1:size_BZ^2)
    # operator list for the operator S_d^+ c^†_{k ↓} c_{k ↑} + h.c.
    spinFlipCorrOplist = [("+-+-", 1.0, [1, 2, 4, 3]), ("+-+-", 1.0, [2, 1, 3, 4])]

    trunc_dim = 3
    basis = BasisStates(trunc_dim * 2 + 2)

    other_k_indices = k_indices[k_indices.≠k_index]
    chosenIndices = [[k_index]; other_k_indices[sortperm(kondoJArrayFull[k_index, other_k_indices, end], rev=true)][1:trunc_dim-1]]
    mapSeq = Dict(index => i for (i, index) in enumerate(chosenIndices))

    Ek_arr = Dict(mapSeq[index] => tightBindDisp(map1DTo2D(index, size_BZ)...) for index in chosenIndices)
    kondoDict = Dict(Tuple(mapSeq[p] for p in points) => kondoJArrayFull[points..., end]
                     for points in Iterators.product(chosenIndices, chosenIndices))
    bathIntDict = Dict(Tuple(mapSeq[p] for p in points) => bathIntForm(W_val, orbitals[2], size_BZ, points)
                       for points in Iterators.product(chosenIndices, chosenIndices, chosenIndices, chosenIndices))
    oplist = KondoKSpace(Ek_arr, kondoDict, bathIntDict)
    fixedPointHamMatrix = generalOperatorMatrix(basis, oplist)
    eigvals, eigstates = getSpectrum(fixedPointHamMatrix)
    return gstateCorrelation(basis, eigvals, eigstates, spinFlipCorrOplist)
end


function plotHeatmaps(results_arr, fig, axes, cmaps, size_BZ)
    k_vals = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    x_arr = y_arr = k_vals
    for (i, result) in enumerate(results_arr)
        reshaped_result = reshape(result, (size_BZ, size_BZ))
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
    elseif probeName == "kondoCoupNodeMap"
        node = (-pi / 2, -pi / 2)
        results, results_bare, results_bool = kondoCoupMap(node, size_BZ, kondoJArrayFull)
        titles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q_\mathrm{node})"
        titles[2] = L"J(k,q_\mathrm{node})"
        titles[3] = L"J^{(0)}(k,q_\mathrm{node})"
        drawPoint = (-0.5, -0.5)
    elseif probeName == "kondoCoupAntinodeMap"
        antinode = (0.0, -pi)
        results, results_bare, results_bool = kondoCoupMap(antinode, size_BZ, kondoJArrayFull)
        titles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q_\mathrm{antin.})"
        titles[2] = L"J(k,q_\mathrm{antin.})"
        titles[3] = L"J^{(0)}(k,q_\mathrm{antin.})"
        drawPoint = (0, -1)
    elseif probeName == "kondoCoupOffNodeMap"
        offnode = (-pi / 2 + 4 * pi / size_BZ, -pi / 2 + 4 * pi / size_BZ)
        results, results_bare, results_bool = kondoCoupMap(offnode, size_BZ, kondoJArrayFull)
        titles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q^\prime_\mathrm{node})"
        titles[2] = L"J(k,q^\prime_\mathrm{node})"
        titles[3] = L"J^{(0)}(k,q^\prime_{\mathrm{node}})"
        drawPoint = offnode ./ pi
    elseif probeName == "kondoCoupOffAntinodeMap"
        offantinode = (0.0, -pi + 4 * pi / size_BZ)
        results, results_bare, results_bool = kondoCoupMap(offantinode, size_BZ, kondoJArrayFull)
        titles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q^\prime_\mathrm{antin.})"
        titles[2] = L"J(k,q^\prime_\mathrm{antin.})"
        titles[3] = L"J^{(0)}(k,q^\prime_{\mathrm{antin.}})"
        drawPoint = offantinode ./ pi
    elseif probeName == "spinFlipCorrMap"
        results = zeros(size_BZ^2)
        @time @sync for k_index in 1:size_BZ^2
            @async results[k_index] = remotecall_fetch(spinFlipCorrMap, max(1, k_index % nprocs()), k_index, size_BZ, kondoJArrayFull, W_by_J * J_val, orbitals)
        end
        results_bool = tolerantSign.(abs.(results), RG_RELEVANCE_TOL)
        titles[1] = L"\mathrm{rel(irrel)evance~of~}J(k,q^\prime_\mathrm{antin.})"
        titles[2] = L"J(k,q^\prime_\mathrm{antin.})"
        titles[3] = L"J^{(0)}(k,q^\prime_{\mathrm{antin.}})"
    end
    fig = Figure()
    titlelayout = GridLayout(fig[0, 1:4])
    header = trunc(size_BZ * W_by_J, digits=1)
    Label(titlelayout[1, 1:4], L"NW/J=%$header", justification=:center, padding=(0, 0, -20, 0))
    axes = [Axis(fig[1, 2*i-1], xlabel=L"\mathrm{k_x}", ylabel=L"\mathrm{k_y}", title=title) for (i, title) in enumerate(titles[1:2])]
    axes = plotHeatmaps((results_bool, results),
        fig, axes[1:2], cmaps, size_BZ)
    if probeName in ["kondoCoupNodeMap", "kondoCoupAntinodeMap", "kondoCoupOffNodeMap", "kondoCoupOffAntinodeMap"]
        [scatter!(ax, [drawPoint[1]], [drawPoint[2]], markersize=20, color=:grey, strokewidth=2, strokecolor=:white) for ax in axes]
    end
    [colsize!(fig.layout, i, Aspect(1, 1.0)) for i in [1, 3]]
    resize_to_layout!(fig)
    return fig
end
