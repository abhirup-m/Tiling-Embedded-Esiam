using LinearAlgebra
using CairoMakie
using Makie
using LaTeXStrings
using Plots
using Combinatorics
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
    results_bool = tolerantSign.(abs.(results), abs.(results_bare) .* RG_RELEVANCE_TOL)
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

    # get size of each quadrant (number of points in the range [0, π])
    halfSize = trunc(Int, (size_BZ + 1) / 2)

    # get all possible indices of momentum states within the Brillouin zone
    k_indices = collect(1:size_BZ^2)

    contributorCounter = Dict{Int64,Float64}(i => 0.0 for i in k_indices)
    # initialise zero matrix for storing correlations
    results = Dict{Int64,Float64}(i => 0 for i in k_indices)
    results_bare = Dict{Int64,Float64}(i => 0 for i in k_indices)

    # number of k-states we will be keeping in any single Hamiltonian
    trunc_dim = 3

    # generating basis states for constructing prototype Hamiltonians which
    # will be diagonalised to obtain correlations
    basis = BasisStates(trunc_dim * 2 + 2)

    for energy in [0]
        suitableIndices = k_indices[abs.(dispersion[k_indices]).<=energy+TOLERANCE]

        for chosenIndices in collect(combinations(suitableIndices, trunc_dim))
            kondoDict = Dict((i, j) => kondoJArrayFull[i, j, end] for (i, j) in Iterators.product(chosenIndices, chosenIndices))
            bathIntDict = Dict(indices => bathIntForm(W_val, orbitals[2], size_BZ, indices) for indices in Iterators.product(chosenIndices, chosenIndices, chosenIndices, chosenIndices))
            operatorList = kondoKSpace_test(chosenIndices, dispersion, kondoDict, bathIntDict)

            fixedPointHamMatrix = generalOperatorMatrix(basis, operatorList)
            println(fixedPointHamMatrix[(3, 1)][1, :])
            println(basis[(3, 1)][1])
            E, X = getSpectrum(fixedPointHamMatrix)
            minimumEnergies = minimum.(values(E))
            minimumBlock = collect(keys(E))[argmin(minimumEnergies)]
            println(E[minimumBlock], minimumBlock)
            gstate = X[minimumBlock][2]
            for (b, s) in zip(basis[minimumBlock], gstate)
                if abs(s) > TOLERANCE
                    println(b, ": ", s)
                end
            end
            for (i, index) in enumerate(chosenIndices)
                contributorCounter[index] += 1
                corrDef = Dict(("n", [2 * i + 1]) => 1.0, ("n", [2 * i + 2]) => 1.0)
                # corrDef = Dict(("+-+-", [1, 2, 2 * i + 2, 2 * i + 1]) => 1.0, ("+-+-", [2, 1, 2 * i + 1, 2 * i + 2]) => 1.0)
                println(corrDef)
                corrOp = generalOperatorMatrix(basis, corrDef)
                results[index] += simpleCorrelation(gstate, corrOp[minimumBlock])
                println((i, index, simpleCorrelation(gstate, corrOp[minimumBlock])))
            end
        end
    end
    # merge!((v1, v2) -> v2 == 0 ? 0 : v1 / v2, results, contributorCounter)
    results_bool = Dict(k => tolerantSign.(abs(results[k]), abs.(results_bare[k]) .* RG_RELEVANCE_TOL) for k in keys(results))
    return results, results_bare, results_bool
end


function plotHeatmaps(results_arr, x_arr, y_arr, fig, axes, cmaps)
    for (i, result) in enumerate(results_arr)
        reshaped_result = reshape(result, (length(x_arr), length(y_arr)))
        hmap = CairoMakie.heatmap!(axes[i], x_arr, y_arr, reshaped_result, colormap=cmaps[i],
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
        results, results_bool = spinFlipCorrMapCoarse(size_BZ, dispersion, kondoJArrayFull, W_by_J * J_val, orbitals)
        titles[1] = L"\mathrm{rel(irrel)evance~of~} "
        titles[2] = L"0.5\langle S_d^+ c^\dagger_{k \downarrow}c_{k\uparrow} + \text{h.c.}\rangle"
        x_arr = y_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    end
    fig = Figure()
    titlelayout = GridLayout(fig[0, 1:4])
    Label(titlelayout[1, 1:4], L"NW/J=%$(trunc(size_BZ * W_by_J, digits=1))", justification=:center, padding=(0, 0, -20, 0))
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
