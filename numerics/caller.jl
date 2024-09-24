using Distributed

if length(Sys.cpu_info()) > 20 && nprocs() == 1
    addprocs(div(length(Sys.cpu_info()), 2))
end
@everywhere using JLD2
@everywhere using LinearAlgebra
@everywhere using Combinatorics
@everywhere using fermions
@everywhere using ProgressMeter
@everywhere const TRUNC_DIM = 2

@everywhere include("./source/constants.jl")
@everywhere include("./source/helpers.jl")
@everywhere include("./source/rgFlow.jl")
@everywhere include("./source/models.jl")
@everywhere include("./source/probes.jl")
include("./source/plotting.jl")
const J_val = 0.1
const omega_by_t = -2.0
const orbitals = ("p", "p")
fractionBZ = 0.1
numShells = 5
size_BZ = 49
#=W_val_arr = -1.0 .* [0] ./ size_BZ=#
W_val_arr = -1.0 .* [0, 10, 20] ./ size_BZ
x_arr = collect(range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi)

node = map2DTo1D(float(π)/2, float(π)/2, size_BZ)
antinode = map2DTo1D(float(π), 0.0, size_BZ)
genpoint = map2DTo1D(float(3 * π / 4), 0.0, size_BZ)

densityOfStates, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
kondoJArrays = Dict()
@time Threads.@threads for W_val in W_val_arr
    kondoJArray, _ = momentumSpaceRG(size_BZ, omega_by_t, J_val, W_val, orbitals)
    averageKondoScale = sum(abs.(kondoJArray[:, :, 1])) / length(kondoJArray[:, :, 1])
    @assert averageKondoScale > RG_RELEVANCE_TOL
    kondoJArray[:, :, end] .= ifelse.(abs.(kondoJArray[:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, kondoJArray[:, :, end], 0)
    kondoJArrays[W_val] = kondoJArray
end

function probe(kondoJArrays, dispersion)
    saveNames = String[]
    for W_val in W_val_arr
        results_scaled, results_bool = scattProb(kondoJArrays[W_val], size_BZ, dispersion, fractionBZ)
        push!(saveNames, plotHeatmap(abs.(results_scaled), (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\Gamma/\Gamma_0~(~W/J=%$(round(W_val/J_val, digits=2))~)$"))
        println("Saved at $(saveNames[end]).")
    end
    run(`pdfunite $(saveNames) scattprob.pdf`)
end


function corr(kondoJArrays, dispersion)
    correlationFuncDict = Dict("SF" => (1, i -> [("+-+-", [2, 1, 2 * i + 1, 2 * i + 2], 1.0), ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1], 1.0)]))
    saveNames = String[]
    @showprogress for W_val in W_val_arr

        hamiltDetails = Dict(
                             "dispersion" => dispersion,
                             "kondoJArray" => kondoJArrays[W_val][:, :, end],
                             "W_val" => 0 * W_val,
                             "orbitals" => orbitals,
                             "size_BZ" => size_BZ,
                            )
        corrResults, corrResultsBool = correlationMap(hamiltDetails, numShells, correlationFuncDict, 10)
        for (name, results) in corrResults
            push!(saveNames, plotHeatmap(results, (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\chi_s(d, \vec{k})~(~W/J=%$(round(W_val/J_val, digits=2))~)$"))
            println("Saved at $(saveNames[end]).")
        end
    end
    run(`pdfunite $(saveNames) correlations.pdf`)
end

@time probe(kondoJArrays, dispersion);
# @time corr(kondoJArrays, dispersion);
