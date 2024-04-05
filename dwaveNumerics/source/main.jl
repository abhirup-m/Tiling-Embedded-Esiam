using ProgressMeter
using Makie, CairoMakie, Measures, LaTeXStrings
using JLD2
set_theme!(theme_latexfonts())
update_theme!(fontsize=24)

include("./rgFlow.jl")
include("./probes.jl")
animName(orbitals, size_BZ, scale, W_by_J_min, W_by_J_max, J_val) = "$(orbitals[1])_$(orbitals[2])_$(size_BZ)_$(round(W_by_J_min, digits=4))_$(size_BZ)_$(round(W_by_J_max, digits=4))_$(round(J_val, digits=4))_$(FIG_SIZE[1] * scale)x$(FIG_SIZE[2] * scale)"

function momentumSpaceRG(size_BZ::Int64, J_val::Float64, bathIntStr::Float64, orbitals::Vector{String})
    # ensure that [0, \pi] has odd number of states, so 
    # that the nodal point is well-defined.
    @assert (size_BZ - 5) % 4 == 0 "Size of Brillouin zone must be of the form N = 4n+5, n=0,1,2..."

    # ensure that the choice of orbitals is d or p
    impOrbital, bathOrbital = orbitals
    @assert bathOrbital in ["p", "d", "poff"]
    @assert impOrbital in ["p", "d", "poff"]

    densityOfStates, dispersionArray = getDensityOfStates(tightBindDisp, size_BZ)

    cutOffEnergies = getCutOffEnergy(size_BZ)

    # Kondo coupling must be stored in a 3D matrix. Two of the dimensions store the 
    # incoming and outgoing momentum indices, while the third dimension stores the 
    # behaviour along the RG flow. For example, J[i][j][k] reveals the value of J 
    # for the momentum pair (i,j) at the k^th Rg step.
    kondoJArray = initialiseKondoJ(size_BZ, impOrbital, trunc(Int, (size_BZ + 1) / 2), J_val)

    # define flags to track whether the RG flow for a particular J_{k1, k2} needs to be stopped 
    # (perhaps because it has gone to zero, or its denominator has gone to zero). These flags are
    # initialised to one, which means that by default, the RG can proceed for all the momenta.
    proceed_flags = fill(1, size_BZ^2, size_BZ^2)

    # Run the RG flow starting from the maximum energy, down to the penultimate energy (ΔE), in steps of ΔE
    @showprogress for (stepIndex, energyCutoff) in enumerate(cutOffEnergies[1:end-1])
        deltaEnergy = abs(cutOffEnergies[stepIndex+1] - cutOffEnergies[stepIndex])

        # set the Kondo coupling of all subsequent steps equal to that of the present step 
        # for now, so that we can just add the renormalisation to it later
        kondoJArray[:, :, stepIndex+1] = kondoJArray[:, :, stepIndex]

        # if there are no enabled flags (i.e., all are zero), stop the RG flow
        if all(==(0), proceed_flags)
            kondoJArray[:, :, stepIndex+2:end] .= kondoJArray[:, :, stepIndex]
            break
        end

        innerIndicesArr, excludedVertexPairs, mixedVertexPairs, cutoffPoints, cutoffHolePoints, proceed_flags = highLowSeparation(dispersionArray, energyCutoff, proceed_flags, size_BZ)

        # calculate the renormalisation for this step and for all k1,k2 pairs
        kondoJArrayNext, proceed_flags_updated = stepwiseRenormalisation(
            innerIndicesArr,
            excludedVertexPairs,
            mixedVertexPairs,
            energyCutoff,
            cutoffPoints,
            cutoffHolePoints,
            proceed_flags,
            kondoJArray[:, :, stepIndex],
            kondoJArray[:, :, stepIndex+1],
            bathIntStr,
            size_BZ,
            deltaEnergy,
            bathOrbital,
            densityOfStates,
        )
        kondoJArray[:, :, stepIndex+1] = kondoJArrayNext
        proceed_flags = proceed_flags_updated
    end
    return kondoJArray, dispersionArray
end


function mapProbeNameToProbe(probeName, size_BZ, kondoJArrayFull, xarr, yarr, W_by_J)
    if probeName == "scattProb"
        results_norm, _, results_bool = scattProb(kondoJArrayFull, size(kondoJArrayFull)[3])
        results1 = results_bool
        results2 = log10.(results_norm)
        title_left = L"\mathrm{relevance/irrelevance~of~}\Gamma(k)"
        title_right = L"\Gamma(k) = \sum_q J(k,q)^2"
        cmap_left = DISCRETE_CGRAD
    elseif probeName == "kondoCoupFSMap"
        results1, results2 = kondoCoupFSMap(size_BZ, kondoJArrayFull)
        title_left = L"J(k,q_\mathrm{node})"
        title_right = L"J(k,q_{\mathrm{antin.}})"
        cmap_left = :matter
    elseif probeName == "kondoCoupMidwayMap"
        (results1, results2), (kx_plot, ky_plot) = kondoCoupMidwayMap(size_BZ, kondoJArrayFull)
        title_left = L"J(k,q_\mathrm{node^\prime})"
        title_right = L"J(k,q_{\mathrm{antin.^\prime}})"
        cmap_left = :matter
    elseif probeName == "kondoCoupDiagMap"
        results1 = kondoCoupDiagMap(size_BZ, kondoJArrayFull)
        results2 = results1
        title_left = L"J(k,k)"
        title_right = ""
        cmap_left = :matter
    end
    fig = Figure()
    titlelayout = GridLayout(fig[0, 1:4])
    Label(titlelayout[1, 1:4], L"W/J=%$(W_by_J)", justification=:center, padding=(0, 0, -20, 0))
    ax1 = Axis(fig[1, 1], xlabel=L"\mathrm{k_x}", ylabel=L"\mathrm{k_y}", title=title_left)
    ax2 = Axis(fig[1, 3], xlabel=L"\mathrm{k_x}", ylabel=L"\mathrm{k_y}", title=title_right)
    ax1, ax2 = plotHeatmaps(reshape(results1, (size_BZ, size_BZ)),
        reshape(results2, (size_BZ, size_BZ)),
        xarr, yarr, fig, [ax1, ax2], cmap_left)
    if probeName == "kondoCoupFSMap"
        scatter!(ax1, [-1 / 2], [-1 / 2], markersize=20, color=:grey, strokewidth=2, strokecolor=:white)
        scatter!(ax2, [0], [-1], markersize=20, color=:grey, strokewidth=2, strokecolor=:white)
    end
    if probeName == "kondoCoupMidwayMap"
        scatter!(ax1, [kx_plot[1] / pi], [ky_plot[1] / pi], markersize=20, color=:grey, strokewidth=2, strokecolor=:white)
        scatter!(ax2, [kx_plot[2] / pi], [ky_plot[2] / pi], markersize=20, color=:grey, strokewidth=2, strokecolor=:white)
    end
    colsize!(fig.layout, 1, Aspect(1, 1.0))
    colsize!(fig.layout, 3, Aspect(1, 1.0))
    resize_to_layout!(fig)
    return fig
end


function manager(size_BZ::Int64, J_val::Float64, W_by_J_range::Vector{Float64}, orbitals::Vector{String}, probes::Vector{String}; figScale=1.0)
    @assert size_BZ % 2 != 0
    savePaths = ["data/$(orbitals)_$(size_BZ)_$(round(J_val, digits=3))_$(round(W_by_J, digits=3)).jld2" for W_by_J in W_by_J_range]
    for (j, W_by_J) in enumerate(W_by_J_range)
        kondoJArrayFull, _ = momentumSpaceRG(size_BZ, J_val, J_val * W_by_J, orbitals)
        file = jldopen(savePaths[j], "w")
        file["kondoJArrayEnds"] = kondoJArrayFull[:, :, [1, end]]
        close(file)
    end
    k_vals = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    for probeName in probes
        pdfFileNames = ["fig_$(W_by_J).pdf" for W_by_J in W_by_J_range]
        # anim = @animate 
        for (W_by_J, savePath, pdfFileName) in zip(W_by_J_range, savePaths, pdfFileNames)
            file = jldopen(savePath, "r")
            kondoJArrayEnds = file["kondoJArrayEnds"]
            close(file)
            kondoJArrayEnds = reshape(kondoJArrayEnds, (size_BZ^2, size_BZ^2, 2))
            fig = mapProbeNameToProbe(probeName, size_BZ, kondoJArrayEnds, k_vals, k_vals, W_by_J)
            display(fig)
            save(pdfFileName, fig, pt_per_unit=100)
        end
        plotName = animName(orbitals, size_BZ, figScale, minimum(W_by_J_range), maximum(W_by_J_range), J_val)
        run(`pdfunite $pdfFileNames $(plotName)-$(replace(probeName, " " => "-")).pdf`)
        run(`rm $pdfFileNames`)
        # gif(anim, plotName * "-$(replace(probeName, " " => "-")).gif", fps=0.5)
        # gif(anim, plotName * "-$(replace(probe, " " => "-")).mp4", fps=0.5)
    end
end
