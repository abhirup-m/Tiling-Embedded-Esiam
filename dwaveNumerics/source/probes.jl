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


function spinFlipCorrMapCoarse(size_BZ::Int64, dispersion::Vector{Float64}, kondoJArrayFull::Array{Float64,3}, W_val::Float64, orbitals::Tuple{String,String})
    nodes = map2DTo1D([-pi / 2, pi / 2, pi / 2, -pi / 2], [-pi / 2, -pi / 2, pi / 2, pi / 2], size_BZ)
    antinodes = map2DTo1D([0, pi, 0, -pi], [-pi, 0, pi, 0], size_BZ)
    chosenPoints = [nodes; antinodes]
    correlationResults = zeros(size_BZ^2)
    bathIntFunc(points) = bathIntForm(W_val, orbitals[2], size_BZ, points)
    @time operatorList = KondoKSpace(chosenPoints, dispersion, kondoJArrayFull[:, :, end], bathIntFunc)
    basis = BasisStates(2 * length(chosenPoints) + 2; totOccupancy=[length(chosenPoints) + 1])
    @time fixedPointHamMatrix = generalOperatorMatrix(basis, operatorList;)
    @time eigvals, eigstates = getSpectrum(fixedPointHamMatrix)
    @time Threads.@threads for (i, point) in collect(enumerate(chosenPoints))
        upIndex = 2 + 2 * (i - 1) + 1
        downIndex = upIndex + 1
        spinFlipCorrOplist = Dict(("+-+-", [1, 2, downIndex, upIndex]) => 0.5, ("+-+-", [2, 1, upIndex, downIndex]) => 0.5)
        correlation = gstateCorrelation(basis, eigvals, eigstates, spinFlipCorrOplist)
        correlationResults[point] = correlation
    end
    correlationResultsBare = zeros(size_BZ^2)
    @time operatorList = KondoKSpace(chosenPoints, dispersion, kondoJArrayFull[:, :, 1], bathIntFunc)
    @time fixedPointHamMatrix = generalOperatorMatrix(basis, operatorList;)
    @time eigvals, eigstates = getSpectrum(fixedPointHamMatrix)
    @time Threads.@threads for (i, point) in collect(enumerate(chosenPoints))
        upIndex = 2 + 2 * (i - 1) + 1
        downIndex = upIndex + 1
        spinFlipCorrOplist = Dict(("+-+-", [1, 2, downIndex, upIndex]) => 0.5, ("+-+-", [2, 1, upIndex, downIndex]) => 0.5)
        correlation = gstateCorrelation(basis, eigvals, eigstates, spinFlipCorrOplist)
        correlationResultsBare[point] = correlation
    end
    correlationResultsBool = tolerantSign.(abs.(correlationResults), abs.(correlationResultsBare) .* RG_RELEVANCE_TOL)
    return abs.(correlationResults), correlationResultsBool
end


function spinFlipCorrMap(size_BZ::Int64, dispersion::Vector{Float64}, kondoJArrayFull::Array{Float64,3}, W_val::Float64, orbitals::Tuple{String,String})
    # get size of each quadrant (number of points in the range [0, π])
    halfSize = trunc(Int, (size_BZ + 1) / 2)

    # get the k-values that lie within the southeast quadrant ([0, π]×[0, π])>
    # We will calculate the correlation only for these values, because the values
    # in the other quadrants will be identical to these.
    kx_SEquadrant = range((K_MIN + K_MAX) / 2, stop=K_MAX, length=halfSize)
    ky_SEquadrant = range(K_MIN, stop=(K_MIN + K_MAX) / 2, length=halfSize)
    SEquadrant_kpairs = [(kx, ky) for ky in ky_SEquadrant for kx in kx_SEquadrant]

    # initialise zero matrix for storing correlations
    results = zeros(halfSize^2)
    results_bare = zeros(halfSize^2)

    # get all possible indices of momentum states within the Brillouin zone
    k_indices = collect(1:size_BZ^2)

    # operator list for the operator S_d^+ c^†_{k ↓} c_{k ↑} + h.c.
    spinFlipCorrOplist = Dict(("+-+-", [1, 2, 4, 3]) => 0.5, ("+-+-", [2, 1, 3, 4]) => 0.5)

    # number of k-states we will be keeping in any single Hamiltonian
    trunc_dim = 2

    # generating basis states for constructing prototype Hamiltonians which
    # will be diagonalised to obtain correlations
    basis = BasisStates(trunc_dim * 2 + 2)

    # inline function to return bath interaction matrix elements
    bathIntFunc(points) = bathIntForm(W_val, orbitals[2], size_BZ, points)

    # loop over all points in the south east quadrant
    Threads.@threads for (kx, ky) in SEquadrant_kpairs

        # only calculate the upper triangular block, because the
        # lower block can be obtained by transposing.
        if kx < ky
            continue
        end

        # obtain the 1D representation of the chosen (kx, ky) point
        k_index = map2DTo1D(kx, ky, size_BZ)

        # get all points which are not (kx, ky) and which lie inside the energy shell of (kx, ky)
        other_k_indices = k_indices[(k_indices.≠k_index).&(abs.(dispersion[k_indices]).<=abs(dispersion[k_index]))]

        # get the k_indices from other_k_indices which have the largest value of J_{k1, k2}
        # chosenIndices = [[k_index]; other_k_indices[sortperm(kondoJArrayFull[k_index, other_k_indices, end], rev=true)][1:trunc_dim-1]]
        nTuplesKstates = [[[k_index]; tuple] for tuple in combinations(other_k_indices, trunc_dim - 1)]
        operatorListSet = KondoKSpace(nTuplesKstates, dispersion, kondoJArrayFull[:, :, end], bathIntFunc)
        fixedPointHamMatrices = generalOperatorMatrix(basis, operatorListSet; tolerance=TOLERANCE^0.5)
        Threads.@threads for fixedPointHamMatrix in fixedPointHamMatrices
            # diagonalise and obtain correlation
            eigvals, eigstates = getSpectrum(fixedPointHamMatrix)
            correlation = gstateCorrelation(basis, eigvals, eigstates, spinFlipCorrOplist)

            # set the appropriate matrix element of the results matrix to this calculated value.
            # Also set the transposed element to the same value.
            results[SEquadrant_kpairs.==[(kx, ky)]] .+= correlation / length(nTuplesKstates)
            results[SEquadrant_kpairs.==[(ky, kx)]] .+= correlation / length(nTuplesKstates)
        end

        # do the same for the bare Hamiltonian
        operatorListSet = KondoKSpace(nTuplesKstates, dispersion, kondoJArrayFull[:, :, 1], bathIntFunc)
        fixedPointHamMatrices = generalOperatorMatrix(basis, operatorListSet; tolerance=TOLERANCE^1)
        Threads.@threads for fixedPointHamMatrix in fixedPointHamMatrices
            # diagonalise and obtain correlation
            eigvals, eigstates = getSpectrum(fixedPointHamMatrix)
            correlation = abs(gstateCorrelation(basis, eigvals, eigstates, spinFlipCorrOplist))

            # set the appropriate matrix element of the results matrix to this calculated value.
            # Also set the transposed element to the same value.
            results_bare[SEquadrant_kpairs.==[(kx, ky)]] .+= correlation / length(nTuplesKstates)
            results_bare[SEquadrant_kpairs.==[(ky, kx)]] .+= correlation / length(nTuplesKstates)
        end
    end
    # calculate whether the correlations are zero or non-zero.
    results_bool = tolerantSign.(abs.(results), abs.(results_bare) .* RG_RELEVANCE_TOL)
    return abs.(results), results_bool, kx_SEquadrant, ky_SEquadrant
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
