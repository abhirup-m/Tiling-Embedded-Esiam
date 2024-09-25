using Distributed

if length(Sys.cpu_info()) > 10 && nprocs() == 1
    addprocs(5)
end
using JLD2
using LinearAlgebra

@everywhere include("./source/constants.jl")
include("./source/helpers.jl")
include("./source/rgFlow.jl")
include("./source/probes.jl")
include("./source/plotting.jl")

global const J_val = 0.1
global const omega_by_t = -2.0
global const orbitals = ("p", "p")
global const maxSize = 5

numShells = 5
size_BZ = 33
W_val_arr = -1.0 .* [0, 5.6, 5.7, 5.8, 5.9, 5.92] ./ size_BZ
x_arr = collect(range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi)

densityOfStates, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
kondoJArrays = Dict{Float64, Array{Float64, 3}}()
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
        results_scaled, results_bool = scattProb(size_BZ, kondoJArrays[W_val], dispersion)
        push!(saveNames, plotHeatmap(results_bool, (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\Gamma/\Gamma_0$", L"$W/J=%$(round(W_val/J_val, digits=2))$"))
        println("\n Saved at $(saveNames[end]).")
    end
    run(`pdfunite $(saveNames) scattprob.pdf`)
end


function corr(kondoJArrays, dispersion)
    spinCorrelation = Dict("SF" => (1, i -> [
                                             ("nn", [1, 2, 2 * i + 1, 2 * i + 1], -0.25),
                                             ("nn", [1, 2, 2 * i + 2, 2 * i + 2], 0.25),
                                             ("nn", [2, 1, 2 * i + 1, 2 * i + 1], 0.25),
                                             ("nn", [2, 1, 2 * i + 1, 2 * i + 2], -0.25),
                                             ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1], -0.5),
                                             ("+-+-", [2, 1, 2 * i + 1, 2 * i + 2], -0.5),
                                            ])
                          )
    chargeCorrelation = Dict("SF" => (1, i -> [("nn", [2 * i + 1, 2 * i + 2], 0.5),
                                               ("hh", [2 * i + 2, 2 * i + 1], 0.5)])
                            )
    saveNames = String[]
    for W_val in W_val_arr

        hamiltDetails = Dict(
                             "dispersion" => dispersion,
                             "kondoJArray" => kondoJArrays[W_val][:, :, end],
                             "W_val" => 0 * W_val,
                             "orbitals" => orbitals,
                             "size_BZ" => size_BZ,
                             "bathIntForm" => bathIntForm,
                            )
        corrResults, corrResultsBool = correlationMap(hamiltDetails, numShells, spinCorrelation, maxSize)
        for (name, results) in corrResults
            push!(saveNames, plotHeatmap(results, (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\chi_s(d, \vec{k})$", L"$W/J=%$(round(W_val/J_val, digits=2))$"))
            println("\n Saved at $(saveNames[end]).")
        end

        hamiltDetails = Dict(
                             "dispersion" => dispersion,
                             "kondoJArray" => kondoJArrays[W_val][:, :, end],
                             "W_val" => W_val,
                             "orbitals" => orbitals,
                             "size_BZ" => size_BZ,
                             "bathIntForm" => bathIntForm,
                            )
        corrResults, corrResultsBool = correlationMap(hamiltDetails, numShells, chargeCorrelation, maxSize)
        for (name, results) in corrResults
            push!(saveNames, plotHeatmap(results, (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\chi_c(d, \vec{k})$", L"$W/J=%$(round(W_val/J_val, digits=2))$"))
            println("\n Saved at $(saveNames[end]).")
        end
    end
    run(`pdfunite $(saveNames) correlations.pdf`)
end

@time probe(kondoJArrays, dispersion);
@time corr(kondoJArrays, dispersion);
