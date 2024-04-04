using ProgressMeter
using Plots, Measures, LaTeXStrings
include("./rgFlow.jl")
include("./probes.jl")

function momentumSpaceRG(size_BZ::Int64, J_val::Float64, bathIntStr::Float64, orbitals::Vector{String})
    # ensure that [0, \pi] has odd number of states, so 
    # that the nodal point is well-defined.
    @assert size_BZ != 0

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


function manager(size_BZ::Int64, J_val::Float64, W_by_J_range::Vector{Float64}, orbitals::Vector{String}, probes::Vector{String}, figScale::Float64)
    @assert size_BZ % 2 != 0
    savePaths = ["data/$(orbitals)_$(size_BZ)_$(round(J_val, digits=3))_$(round(W_by_J, digits=3))" for W_by_J in W_by_J_range]
    @time for (j, W_by_J) in enumerate(W_by_J_range)
        kondoJArrayFull, dispersionArray = momentumSpaceRG(size_BZ, J_val, J_val * W_by_J, orbitals)
        writedlm(savePaths[j], kondoJArrayFull[:, :, [1, end]])
    end
    k_vals = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    for probe in probes
        pdfFileNames = []
        anim = @animate for (W_by_J, savePath) in zip(W_by_J_range, savePaths)
            kondoJArrayFull = readdlm(savePath)
            kondoJArrayFull = reshape(kondoJArrayFull, (size_BZ^2, size_BZ^2, 2))
            p1 = plot()
            p2 = plot()
            if probe == "scattProb"
                results_norm, results_unnorm, results_bool = scattProb(kondoJArrayFull, size(kondoJArrayFull)[3])
                results1 = results_bool
                results2 = log10.(results_norm)
                title_left = "relevance/irrelevanceo of \$\\Gamma(k)\$"
                title_right = "\$\\Gamma(k) = \\sum_q J(k,q)^2\$"
                cmap_left = discreteCgrad
                p1, p2 = plotHeatmaps(reshape(results1, (size_BZ, size_BZ)),
                    reshape(results2, (size_BZ, size_BZ)),
                    k_vals, [p1, p2], cmap_left, [title_left, title_right])
            elseif probe == "kondoCoupFSMap"
                results1, results2 = kondoCoupFSMap(size_BZ, kondoJArrayFull)
                title_left = "\$J(k,q_\\mathrm{node})\$"
                title_right = "\$J(k,q_{\\mathrm{antin.}})\$"
                cmap_left = :matter
                p1, p2 = plotHeatmaps(reshape(results1, (size_BZ, size_BZ)),
                    reshape(results2, (size_BZ, size_BZ)),
                    k_vals, [p1, p2], cmap_left, [title_left, title_right])
                scatter!(p1, [-1 / 2], [-1 / 2], markersize=6, markercolor=:grey, markerstrokecolor=:white, markerstrokewidth=4)
                scatter!(p2, [0], [-1], markersize=6, markercolor=:grey, markerstrokecolor=:white, markerstrokewidth=4)
            elseif probe == "kondoCoupDiagMap"
                results1 = kondoCoupDiagMap(size_BZ, kondoJArrayFull)
                results2 = results1
                title_left = "\$J(k,k)\$"
                title_right = ""
                cmap_left = :matter
                p1, p2 = plotHeatmaps(reshape(results1, (size_BZ, size_BZ)),
                    reshape(results2, (size_BZ, size_BZ)),
                    k_vals, [p1, p2], cmap_left, [title_left, title_right])
            end
            p = plot(p1, p2, size=SIZE,
                top_margin=3mm, left_margin=[5mm 10mm], bottom_margin=5mm,
                xlabel="\$ \\mathrm{k_x/\\pi} \$", ylabel="\$ \\mathrm{k_y/\\pi} \$",
                dpi=100 * figScale,
            )
            pdfFileName = "fig_$(W_by_J).pdf"
            savefig(p, pdfFileName)
            push!(pdfFileNames, pdfFileName)
        end
        plotName = animName(orbitals, size_BZ, figScale, minimum(W_by_J_range), maximum(W_by_J_range), J_val)
        run(`pdfunite $pdfFileNames fig-$(plotName)-$(replace(probe, " " => "-")).pdf`)
        run(`rm $pdfFileNames`)
        gif(anim, plotName * "-$(replace(probe, " " => "-")).gif", fps=0.5)
        # gif(anim, plotName * "-$(replace(probe, " " => "-")).mp4", fps=0.5)
    end
end
