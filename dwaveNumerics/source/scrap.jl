
#### IMPORTS ####
using Distributed
using SharedArrays
addprocs(5)
params = (dir="/home/abhirup/work/mOrbRG/source",)
addprocs([("abhirup@10.20.90.179", :auto)]; params...)
@everywhere using ProgressMeter
@everywhere using Plots
@everywhere using Measures
@everywhere using LaTeXStrings
@everywhere using DelimitedFiles
@everywhere include("mOrbRG.jl")
@everywhere include("probes.jl")
const SIZE = (750, 300)
scale = 1.5
discreteCgrad = cgrad(:BuPu_3, 3, categorical=true, rev=true)
animName(orbital, num_kspace, scale, W_by_J_max, W_by_J_min) = "kondoUb_kspaceRG_$(orbital)_wave_$(num_kspace)_$(round(W_by_J_min, digits=3))_$(round(W_by_J_max, digits=3))_$(SIZE[1] * scale)x$(SIZE[2] * scale)"

function getRgFlow(num_kspace_half, J_val, W_by_J_range, orbitals)
    num_kspace = 2 * num_kspace_half + 1
    k_vals = collect(range(K_MIN, K_MAX, length=num_kspace)) ./ pi
    results1_arr = SharedArray{Float64}(length(W_by_J_range), num_kspace * num_kspace)
    results2_arr = SharedArray{Float64}(length(W_by_J_range), num_kspace * num_kspace)
    @time @sync @distributed for (j, W_by_J) in collect(enumerate(W_by_J_range))
        savePath = "data/$(orbitals[1])_$(num_kspace_half)_$(round(J_val, digits=3))_$(round(W_by_J, digits=3))"
        kondoJArray, _ = main(num_kspace_half, J_val, J_val * W_by_J, orbitals[1], orbitals[2])
        _, results_unnorm, results_bool = totalScatterProb(kondoJArray, num_kspace_half + 1, savePath)
        results1_arr[j, :] = results_bool
        results2_arr[j, :] = results_unnorm
    end
    max_r = maximum(log10.(results2_arr))
    min_r = minimum(log10.(results2_arr))
    pdfFileNames = []
    anim = @animate for (j, W_by_J) in enumerate(W_by_J_range)
        hm1 = heatmap(k_vals, k_vals, results1_arr[j, :, :],
            cmap=discreteCgrad)
        hm2 = heatmap(k_vals, k_vals, log10.(results2_arr[j, :, :]),
            cmap=:matter, clims=(min_r, max_r))
        p = plot(hm1, hm2, size=SIZE,
            plot_title="\$W/J=$(round(W_by_J, digits=3))\$",
            top_margin=3mm, left_margin=[5mm 10mm], bottom_margin=5mm,
            xlabel="\$ \\mathrm{k_x/\\pi} \$", ylabel="\$ \\mathrm{k_y/\\pi} \$",
            dpi=100 * scale
        )
        pdfFileName = "fig_$(j).pdf"
        savefig(p, pdfFileName)
        push!(pdfFileNames, pdfFileName)
    end
    saveName = animName(orbitals[1], num_kspace, scale, minimum(W_by_J_range), maximum(W_by_J_range))
    run(`pdfunite $pdfFileNames $saveName.pdf`)
    run(`rm $pdfFileNames`)
    gif(anim, saveName * ".gif", fps=0.5)
    gif(anim, saveName * ".mp4", fps=0.5)
end

getRgFlow(parse(Int64, ARGS[1]), parse(Float64, ARGS[2]), parse(Float64, ARGS[3]):parse(Float64, ARGS[4]):parse(Float64, ARGS[5]), [ARGS[6], ARGS[7]])
