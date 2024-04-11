using Makie, CairoMakie, Measures, LaTeXStrings
using Distributed

# add four more processes, to allow distributed computing, only if this hasn't already been done before (hence the if check)
if nprocs() == 1
    addprocs(4)
end
@everywhere using JLD2
@everywhere using ProgressMeter
set_theme!(theme_latexfonts())
update_theme!(fontsize=24)

@everywhere include("./rgFlow.jl")
@everywhere include("./probes.jl")

# file name format for saving plots and animations
animName(orbitals, size_BZ, omega_by_t, scale, W_by_J_min, W_by_J_max, J_val) = "$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(W_by_J_min, digits=4))_$(round(W_by_J_max, digits=4))_$(round(J_val, digits=4))_$(FIG_SIZE[1] * scale)x$(FIG_SIZE[2] * scale)"

@everywhere function momentumSpaceRG(size_BZ::Int64, omega_by_t::Float64, J_val::Float64, bathIntStr::Float64, orbitals::Vector{String}; progressbarEnabled=false)

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
            omega_by_t,
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


function manager(size_BZ::Int64, omega_by_t::Float64, J_val::Float64, W_by_J_range::Vector{Float64}, orbitals::Vector{String}, probes::Vector{String}; figScale=1.0)
    # ensure that [0, \pi] has odd number of states, so 
    # that the nodal point is well-defined.
    @assert (size_BZ - 5) % 4 == 0 "Size of Brillouin zone must be of the form N = 4n+5, n=0,1,2..., so that all the nodes and antinodes are well-defined."

    # ensure that the requested probe is among the ones that can be calculated
    @assert issubset(Set(probes), Set(ALLOWED_PROBES))

    # determines whether the inner loops (RG iteration, for a single value of W)
    # should show progress bars.
    progressbarEnabled = length(W_by_J_range) == 1 ? true : false

    # file paths for saving RG flow results, specifically bare and fixed point values of J as well as the dispersion
    savePaths = ["data/$(orbitals)_$(size_BZ)_$(round(J_val, digits=3))_$(round(W_by_J, digits=3)).jld2" for W_by_J in W_by_J_range]

    # loop over all given values of W/J, get the fixed point distribution J(k1,k2) and save them in files
    @sync @showprogress enabled = !(progressbarEnabled) @distributed for (j, W_by_J) in collect(enumerate(W_by_J_range))
        kondoJArrayFull, dispersion, fixedpointEnergy = momentumSpaceRG(size_BZ, omega_by_t, J_val, J_val * W_by_J, orbitals; progressbarEnabled=progressbarEnabled)
        jldopen(savePaths[j], "w") do file
            file["kondoJArrayEnds"] = kondoJArrayFull[:, :, [1, end]]
            file["dispersion"] = dispersion
            file["fixedpointEnergy"] = fixedpointEnergy
        end
    end

    # loop over the various probes that has been requested
    for probeName in probes
        pdfFileNames = ["fig_$(W_by_J).pdf" for W_by_J in W_by_J_range]

        # loop over each value of W/J to calculate the probe for that value
        # anim = @animate 
        for (W_by_J, savePath, pdfFileName) in zip(W_by_J_range, savePaths, pdfFileNames)

            # load saved data
            jldopen(savePath, "r"; compress=true) do file
                kondoJArrayEnds = file["kondoJArrayEnds"]
                dispersion = file["dispersion"]
                fixedpointEnergy = file["fixedpointEnergy"]

                # reshape to 1D array of size N^2 into 2D array of size NxN, so that we can plot it as kx vs ky.
                kondoJArrayEnds = reshape(kondoJArrayEnds, (size_BZ^2, size_BZ^2, 2))

                # calculate and plot the probe result, then save the fig.
                fig = mapProbeNameToProbe(probeName, size_BZ, kondoJArrayEnds, W_by_J, dispersion, fixedpointEnergy)
                display(fig)
                save(pdfFileName, fig, pt_per_unit=100)
            end
        end
        plotName = animName(orbitals, size_BZ, figScale, minimum(W_by_J_range), maximum(W_by_J_range), J_val)
        run(`pdfunite $pdfFileNames $(plotName)-$(replace(probeName, " " => "-")).pdf`)
        run(`rm $pdfFileNames`)
        # gif(anim, plotName * "-$(replace(probeName, " " => "-")).gif", fps=0.5)
        # gif(anim, plotName * "-$(replace(probe, " " => "-")).mp4", fps=0.5)
    end
end
