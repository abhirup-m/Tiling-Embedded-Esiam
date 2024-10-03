using Distributed

if length(Sys.cpu_info()) > 10 && nprocs() == 1
    addprocs(5)
end
using JLD2
using LinearAlgebra

@everywhere include("./source/constants.jl")
include("./source/helpers.jl")
@everywhere include("./source/rgFlow.jl")
include("./source/probes.jl")
include("./source/plotting.jl")

global const J_val = 0.1
global const omega_by_t = -2.0
@everywhere global const orbitals = ("p", "p")
global const maxSize = 500

numShells = 3
size_BZ = 33
W_val_arr = -1.0 .* [0, 5.6, 5.7, 5.8, 5.9, 5.92] ./ size_BZ
x_arr = collect(range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi)

function RGFlow(W_val_arr, size_BZ)
    kondoJArrays = Dict{Float64, Array{Float64, 3}}()
    results = @showprogress desc="rg flow" pmap(w -> momentumSpaceRG(size_BZ, omega_by_t, J_val, w, orbitals), W_val_arr)
    dispersion = results[1][2]
    for (result, W_val) in zip(results, W_val_arr)
        #=kondoJArray, dispersion = momentumSpaceRG(size_BZ, omega_by_t, J_val, W_val, orbitals)=#
        averageKondoScale = sum(abs.(result[1][:, :, 1])) / length(result[1][:, :, 1])
        @assert averageKondoScale > RG_RELEVANCE_TOL
        result[1][:, :, end] .= ifelse.(abs.(result[1][:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, result[1][:, :, end], 0)
        kondoJArrays[W_val] = result[1]
    end
    return kondoJArrays, dispersion
end

function probe(kondoJArrays, dispersion)
    saveNames = String[]
    for W_val in W_val_arr
        results_scaled, results_bool = scattProb(size_BZ, kondoJArrays[W_val], dispersion)
        push!(saveNames, plotHeatmap(results_bool, (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\Gamma/\Gamma_0$", L"$W/J=%$(round(W_val/J_val, digits=2))$//$M_s=%$(maxSize)$"))
    end
    println("\n Saved at $(saveNames).")
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
    chargeCorrelation = Dict("CF" => (1, i -> [("nn", [2 * i + 1, 2 * i + 2], 0.5),
                                               ("hh", [2 * i + 2, 2 * i + 1], 0.5)])
                            )
    vneDef = Dict("vne_k" => (1, i -> [2 * i + 1, 2 * i + 2]))
    saveNames = String[]
    @time Threads.@threads for W_val in W_val_arr

        hamiltDetails = Dict(
                             "dispersion" => dispersion,
                             "kondoJArray" => kondoJArrays[W_val][:, :, end],
                             "W_val" => 0 * W_val,
                             "orbitals" => orbitals,
                             "size_BZ" => size_BZ,
                             "bathIntForm" => bathIntForm,
                            )

        corrResults, corrResultsBool = correlationMap(hamiltDetails, numShells, spinCorrelation, maxSize; vneFuncDict=vneDef)
        push!(saveNames, plotHeatmap(corrResults["SF"], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\chi_s(d, \vec{k})$", L"$W/J=%$(round(W_val/J_val, digits=2))$"))
        push!(saveNames, plotHeatmap(corrResults["vne_k"], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\mathrm{S}_\mathrm{EE}(\vec{k})$", L"$W/J=%$(round(W_val/J_val, digits=2))$"))

        hamiltDetails = Dict(
                             "dispersion" => dispersion,
                             "kondoJArray" => kondoJArrays[W_val][:, :, end],
                             "W_val" => W_val,
                             "orbitals" => orbitals,
                             "size_BZ" => size_BZ,
                             "bathIntForm" => bathIntForm,
                            )
        corrResults, corrResultsBool = correlationMap(hamiltDetails, numShells, chargeCorrelation, maxSize)
        push!(saveNames, plotHeatmap(corrResults["CF"], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\chi_c(d, \vec{k})$", L"$W/J=%$(round(W_val/J_val, digits=2))$"))
    end
    println("\n Saved at $(saveNames).")
    run(`pdfunite $(saveNames) correlations.pdf`)
end

@time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ)
@time probe(kondoJArrays, dispersion)
@time corr(kondoJArrays, dispersion)
