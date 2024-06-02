using JLD2
using ProgressMeter
include("./rgFlow.jl")
include("./probes.jl")
include("./plotting.jl")

# file name format for saving plots and animations
animName(orbitals, size_BZ, omega_by_t, scale, W_by_J_min, W_by_J_max, J_val) = "$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(W_by_J_min, digits=4))_$(round(W_by_J_max, digits=4))_$(round(J_val, digits=4))_$(FIG_SIZE[1] * scale)x$(FIG_SIZE[2] * scale)"


function manager(size_BZ::Int64, omega_by_t::Float64, J_val::Float64, W_by_J_range::Vector{Float64}, orbitals::Tuple{String,String}, probes::Vector{String}; figScale::Float64=1.0, saveDir::String="./data/")

    # ensure that the requested probe is among the ones that can be calculated
    @assert issubset(Set(probes), Set(ALLOWED_PROBES))

    # determines whether the inner loops (RG iteration, for a single value of W)
    # should show progress bars.
    progressbarEnabled = length(W_by_J_range) == 1 ? true : false

    # file paths for saving RG flow results, specifically bare and fixed point values of J as well as the dispersion
    savePaths = [saveDir * "$(orbitals)_$(size_BZ)_$(round(J_val, digits=3))_$(round(W_by_J, digits=3)).jld2" for W_by_J in W_by_J_range]

    # create the data save directory if it doesn't already exist
    isdir(saveDir) || mkdir(saveDir)

    # loop over all given values of W/J, get the fixed point distribution J(k1,k2) and save them in files
    @showprogress Threads.@threads for (j, W_by_J) in collect(enumerate(W_by_J_range))
        kondoJArrayFull, dispersion = momentumSpaceRG(size_BZ, omega_by_t, J_val, J_val * W_by_J, orbitals; progressbarEnabled=progressbarEnabled)
        jldopen(savePaths[j], "w") do file
            file["kondoJArray"] = kondoJArrayFull[:, :, [1, end]]
            file["dispersion"] = dispersion
        end
    end

    # loop over the various probes that has been requested
    @showprogress for probeName in probes
        pdfFileNames = ["fig_$(W_by_J)_$probeName.pdf" for W_by_J in W_by_J_range]

        # loop over each value of W/J to calculate the probe for that value
        for (W_by_J, savePath, pdfFileName) in zip(W_by_J_range, savePaths, pdfFileNames)

            # load saved data
            jldopen(savePath, "r"; compress=true) do file
                kondoJArray = file["kondoJArray"]
                averageKondoScale = sum(abs.(kondoJArray[:, :, 1])) / length(kondoJArray[:, :, 1])
                @assert averageKondoScale > RG_RELEVANCE_TOL
                kondoJArray[:, :, end] .= ifelse.(abs.(kondoJArray[:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, kondoJArray[:, :, end], 0)
                dispersion = file["dispersion"]

                # reshape to 1D array of size N^2 into 2D array of size NxN, so that we can plot it as kx vs ky.
                kondoJArray = reshape(kondoJArray, (size_BZ^2, size_BZ^2, 2))

                # calculate and plot the probe result, then save the fig.
                results, results_bool = mapProbeNameToProbe(probeName, size_BZ, kondoJArray, W_by_J * J_val, dispersion, orbitals)
                fig = mainPlotter(results, results_bool, probeName, size_BZ, L"a")
                save(pdfFileName, fig, pt_per_unit=figScale)
            end
        end
        plotName = animName(orbitals, size_BZ, omega_by_t, figScale, minimum(W_by_J_range), maximum(W_by_J_range), J_val)
        run(`pdfunite $pdfFileNames $(plotName)-$(replace(probeName, " " => "-")).pdf`)
        run(`rm $pdfFileNames`)
    end
end

function transitionPoints(size_BZ_max::Int64, W_by_J_max::Float64, omega_by_t::Float64, J_val::Float64, orbitals::Tuple{String,String}; figScale::Float64=1.0, saveDir::String="./data/")
    size_BZ_min = 5
    size_BZ_vals = size_BZ_min:4:size_BZ_max
    antinodeTransition = Float64[]
    nodeTransition = Float64[]
    for (kvals, array) in zip([(-pi/2, -pi/2), (0.0, -pi)], [nodeTransition, antinodeTransition])
        W_by_J_bracket = [0, W_by_J_max]
        @showprogress for (i, size_BZ) in enumerate(size_BZ_vals)
            if i > 1
                W_by_J_bracket = [array[i-1], W_by_J_max]
            end
            kpoint = map2DTo1D(kvals..., size_BZ)
            while maximum(W_by_J_bracket) - minimum(W_by_J_bracket) > 0.1
                bools = []
                for W_by_J in [W_by_J_bracket[1], sum(W_by_J_bracket) / 2, W_by_J_bracket[2]]
                    kondoJArrayFull, dispersion = momentumSpaceRG(size_BZ, omega_by_t, J_val, J_val * W_by_J, orbitals)
                    results, results_bool = mapProbeNameToProbe("scattProb", size_BZ, kondoJArrayFull, W_by_J * J_val, dispersion, orbitals)
                    push!(bools, results_bool[kpoint] == 0)
                end
                if bools[1] == false && bools[3] == true
                    if bools[2] == true
                        W_by_J_bracket[2] = sum(W_by_J_bracket) / 2
                    else
                        W_by_J_bracket[1] = sum(W_by_J_bracket) / 2
                    end
                else
                    W_by_J_bracket[2] = W_by_J_bracket[1]
                    W_by_J_bracket[1] = 0
                end
            end
            push!(array, sum(W_by_J_bracket) / 2)
            println(kvals, array[end])
        end
    end
    println("A", antinodeTransition)
    println("N", nodeTransition)
end
