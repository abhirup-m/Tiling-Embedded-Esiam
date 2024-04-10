using Distributed
if nprocs() == 1
    addprocs(4)
end
@everywhere using ProgressMeter
@everywhere using Makie, CairoMakie, Measures, LaTeXStrings
@everywhere using JLD2
set_theme!(theme_latexfonts())
update_theme!(fontsize=24)

@everywhere include("./rgFlow.jl")
@everywhere include("./probes.jl")
animName(orbitals, size_BZ, scale, W_by_J_min, W_by_J_max, J_val) = "$(orbitals[1])_$(orbitals[2])_$(size_BZ)_$(round(W_by_J_min, digits=4))_$(size_BZ)_$(round(W_by_J_max, digits=4))_$(round(J_val, digits=4))_$(FIG_SIZE[1] * scale)x$(FIG_SIZE[2] * scale)"

@everywhere function momentumSpaceRG(size_BZ::Int64, J_val::Float64, bathIntStr::Float64, orbitals::Vector{String}; progressbarEnabled=false)

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

    fixedpointEnergy = minimum(cutOffEnergies)
    @showprogress enabled = progressbarEnabled for (stepIndex, energyCutoff) in enumerate(cutOffEnergies[1:end-1])
        deltaEnergy = abs(cutOffEnergies[stepIndex+1] - cutOffEnergies[stepIndex])

        # set the Kondo coupling of all subsequent steps equal to that of the present step 
        # for now, so that we can just add the renormalisation to it later
        kondoJArray[:, :, stepIndex+1] = kondoJArray[:, :, stepIndex]

        # if there are no enabled flags (i.e., all are zero), stop the RG flow
        if all(==(0), proceed_flags)
            kondoJArray[:, :, stepIndex+2:end] .= kondoJArray[:, :, stepIndex]
            fixedpointEnergy = energyCutoff
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
    return kondoJArray, dispersionArray, fixedpointEnergy
end


function mapProbeNameToProbe(probeName, size_BZ, kondoJArrayFull, xarr, yarr, W_by_J, dispersion, fixedpointEnergy)
    titles = Vector{LaTeXString}(undef, 3)
    cmaps = [DISCRETE_CGRAD, :matter, :matter]
    if probeName == "scattProb"
        results, results_bare, results_bool = scattProb(kondoJArrayFull, size(kondoJArrayFull)[3], size_BZ, dispersion, fixedpointEnergy)
        titles[1] = L"\mathrm{relevance/irrelevance~of~}\Gamma(k)"
        titles[2] = L"\Gamma(k) = \sum_q J(k,q)^2"
        titles[3] = L"\Gamma^{(0)}(k) = \sum_q J(k,q)^2"
    elseif probeName == "kondoCoupNodeMap"
        node = (-pi / 2, -pi / 2)
        results, results_bare, results_bool = kondoCoupMap(node, size_BZ, kondoJArrayFull)
        titles[1] = L"\mathrm{relevance/irrelevance~of~}J(k,q_\mathrm{node})"
        titles[2] = L"J(k,q_\mathrm{node})"
        titles[3] = L"J^{(0)}(k,q_\mathrm{node})"
        drawPoint = (-0.5, -0.5)
    elseif probeName == "kondoCoupAntinodeMap"
        antinode = (0.0, -pi)
        results, results_bare, results_bool = kondoCoupMap(antinode, size_BZ, kondoJArrayFull)
        titles[1] = L"\mathrm{relevance/irrelevance~of~}J(k,q_\mathrm{antin.})"
        titles[2] = L"J(k,q_\mathrm{antin.})"
        titles[3] = L"J^{(0)}(k,q_\mathrm{antin.})"
        drawPoint = (0, -1)
    elseif probeName == "kondoCoupOffNodeMap"
        offnode = (-pi / 2 + 4 * pi / size_BZ, -pi / 2 + 4 * pi / size_BZ)
        results, results_bare, results_bool = kondoCoupMap(offnode, size_BZ, kondoJArrayFull)
        titles[1] = L"\mathrm{relevance/irrelevance~of~}J(k,q^\prime_\mathrm{node})"
        titles[2] = L"J(k,q^\prime_\mathrm{node})"
        titles[3] = L"J^{(0)}(k,q^\prime_{\mathrm{node}})"
        drawPoint = offnode ./ pi
    elseif probeName == "kondoCoupOffAntinodeMap"
        offantinode = (0.0, -pi + 4 * pi / size_BZ)
        results, results_bare, results_bool = kondoCoupMap(offantinode, size_BZ, kondoJArrayFull)
        titles[1] = L"\mathrm{relevance/irrelevance~of~}J(k,q^\prime_\mathrm{antin.})"
        titles[2] = L"J(k,q^\prime_\mathrm{antin.})"
        titles[3] = L"J^{(0)}(k,q^\prime_{\mathrm{antin.}})"
        drawPoint = offantinode ./ pi
    end
    fig = Figure()
    titlelayout = GridLayout(fig[0, 1:6])
    Label(titlelayout[1, 1:6], L"NW/J=%$(trunc(size_BZ * W_by_J, digits=1))", justification=:center, padding=(0, 0, -20, 0))
    axes = [Axis(fig[1, 2*i-1], xlabel=L"\mathrm{k_x}", ylabel=L"\mathrm{k_y}", title=title) for (i, title) in enumerate(titles)]
    axes = plotHeatmaps((results_bool, results, results_bare),
        xarr, yarr, fig, axes, cmaps, size_BZ)
    if probeName in ["kondoCoupNodeMap", "kondoCoupAntinodeMap", "kondoCoupOffNodeMap", "kondoCoupOffAntinodeMap"]
        [scatter!(ax, [drawPoint[1]], [drawPoint[2]], markersize=20, color=:grey, strokewidth=2, strokecolor=:white) for ax in axes]
    end
    [colsize!(fig.layout, i, Aspect(1, 1.0)) for i in [1, 3, 5]]
    resize_to_layout!(fig)
    return fig
end


function manager(size_BZ::Int64, J_val::Float64, W_by_J_range::Vector{Float64}, orbitals::Vector{String}, probes::Vector{String}; figScale=1.0)
    # ensure that [0, \pi] has odd number of states, so 
    # that the nodal point is well-defined.
    @assert (size_BZ - 5) % 4 == 0 "Size of Brillouin zone must be of the form N = 4n+5, n=0,1,2..., so that all the nodes and antinodes are well-defined."

    progressbarEnabled = length(W_by_J_range) == 1 ? true : false
    savePaths = ["data/$(orbitals)_$(size_BZ)_$(round(J_val, digits=3))_$(round(W_by_J, digits=3)).jld2" for W_by_J in W_by_J_range]
    @sync @showprogress enabled = !(progressbarEnabled) @distributed for (j, W_by_J) in collect(enumerate(W_by_J_range))
        kondoJArrayFull, dispersion, fixedpointEnergy = momentumSpaceRG(size_BZ, J_val, J_val * W_by_J, orbitals; progressbarEnabled=progressbarEnabled)
        file = jldopen(savePaths[j], "w")
        file["kondoJArrayEnds"] = kondoJArrayFull[:, :, [1, end]]
        file["dispersion"] = dispersion
        file["fixedpointEnergy"] = fixedpointEnergy
        close(file)
    end
    k_vals = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    for probeName in probes
        pdfFileNames = ["fig_$(W_by_J).pdf" for W_by_J in W_by_J_range]
        # anim = @animate 
        for (W_by_J, savePath, pdfFileName) in zip(W_by_J_range, savePaths, pdfFileNames)
            file = jldopen(savePath, "r")
            kondoJArrayEnds = file["kondoJArrayEnds"]
            dispersion = file["dispersion"]
            fixedpointEnergy = file["fixedpointEnergy"]
            close(file)
            kondoJArrayEnds = reshape(kondoJArrayEnds, (size_BZ^2, size_BZ^2, 2))
            fig = mapProbeNameToProbe(probeName, size_BZ, kondoJArrayEnds, k_vals, k_vals, W_by_J, dispersion, fixedpointEnergy)
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
