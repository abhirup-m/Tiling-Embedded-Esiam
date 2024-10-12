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
global const maxSize = 300

numShells = 4
size_BZ = 33
W_val_arr = -1.0 .* [0,] ./ size_BZ
#=W_val_arr = -1.0 .* [0, 5.6, 5.7, 5.8, 5.88, 5.92] ./ size_BZ=#
x_arr = collect(range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi)
label(W_val) = L"$W/J=%$(round(W_val/J_val, digits=2))$\n$M_s=%$(maxSize)$"

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
        push!(saveNames, plotHeatmap(results_bool, (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\Gamma/\Gamma_0$", label(W_val)))
    end
    println("\n Saved at $(saveNames).")
    run(`pdfunite $(saveNames) scattprob.pdf`)
end


function corr(kondoJArrays, dispersion)
    spinCorrelation = Dict("SF" => i -> [
                                             ("nn", [1, 2, 2 * i + 1, 2 * i + 1], -0.25),
                                             ("nn", [1, 2, 2 * i + 2, 2 * i + 2], 0.25),
                                             ("nn", [2, 1, 2 * i + 1, 2 * i + 1], 0.25),
                                             ("nn", [2, 1, 2 * i + 1, 2 * i + 2], -0.25),
                                             ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1], -0.5),
                                             ("+-+-", [2, 1, 2 * i + 1, 2 * i + 2], -0.5),
                                            ]
                          )
    chargeCorrelation = Dict("CF" => i -> [("nn", [2 * i + 1, 2 * i + 2], 0.5),
                                               ("hh", [2 * i + 2, 2 * i + 1], 0.5)]
                            )
    vneDef = Dict("vne_k" => i -> [2 * i + 1, 2 * i + 2])
    mutInfoDef = Dict(
                      "I2_k_d" => (nothing, (i, j) -> ([1, 2], [2 * i + 1, 2 * i + 2])),
                      "I2_k_N" => ((-π/2, -π/2), (i, j) -> ([2 * j + 1, 2 * j + 2], [2 * i + 1, 2 * i + 2])),
                      "I2_k_AN" => ((-π, 0.), (i, j) -> ([2 * j + 1, 2 * j + 2], [2 * i + 1, 2 * i + 2])),
                     )
    saveNamesCorr = String[]
    saveNamesVneSpin = String[]
    saveNamesVneCharge = String[]
    saveNamesMutInfoSpinImp = String[]
    saveNamesMutInfoSpinNode = String[]
    saveNamesMutInfoSpinAntiNode = String[]
    saveNamesMutInfoChargeImp = String[]
    saveNamesMutInfoChargeNode = String[]
    saveNamesMutInfoChargeAntiNode = String[]
    for W_val in W_val_arr

        hamiltDetails = Dict(
                             "dispersion" => dispersion,
                             "kondoJArray" => kondoJArrays[W_val][:, :, end],
                             "W_val" => 0 * W_val,
                             "orbitals" => orbitals,
                             "size_BZ" => size_BZ,
                             "bathIntForm" => bathIntForm,
                            )

        corrResults, corrResultsBool = correlationMap(hamiltDetails, numShells, spinCorrelation, maxSize; vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef)
        if !isempty(corrResults["I2_k_AN"][corrResults["I2_k_AN"] .< -1e-5])
            println("I2_k_AN", corrResults["I2_k_AN"][corrResults["I2_k_AN"] .< -1e-5])
        end
        push!(saveNamesCorr, plotHeatmap(corrResults["SF"], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\chi_s(d, \vec{k})$", label(W_val)))
        push!(saveNamesVneSpin, plotHeatmap(corrResults["vne_k"], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\mathrm{S}_\mathrm{EE}^{(s)}(\vec{k})$", label(W_val)))
        push!(saveNamesMutInfoSpinImp, plotHeatmap(corrResults["I2_k_d"], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$I_2^{(s)}(d,\vec{k})$", label(W_val)))
        push!(saveNamesMutInfoSpinNode, plotHeatmap(corrResults["I2_k_N"], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$I_2^{(s)}(k_\mathrm{N},\vec{k})$", label(W_val)))
        push!(saveNamesMutInfoSpinAntiNode, plotHeatmap(corrResults["I2_k_AN"], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$I_2^{(s)}(k_\mathrm{AN},\vec{k})$", label(W_val)))

        hamiltDetails["W_val"] = W_val
        corrResults, corrResultsBool = correlationMap(hamiltDetails, numShells, chargeCorrelation, maxSize; vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef)
        if !isempty(corrResults["I2_k_AN"][corrResults["I2_k_AN"] .< -1e-5])
            println("I2_k_AN", corrResults["I2_k_AN"][corrResults["I2_k_AN"] .< -1e-5])
        end
        push!(saveNamesCorr, plotHeatmap(corrResults["CF"], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\chi_c(d, \vec{k})$", label(W_val)))
        push!(saveNamesVneCharge, plotHeatmap(corrResults["vne_k"], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$\mathrm{S}^{(c)}_\mathrm{EE}(\vec{k})$", label(W_val)))
        push!(saveNamesMutInfoChargeImp, plotHeatmap(corrResults["I2_k_d"], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$I_2^{(c)}(d,\vec{k})$", label(W_val)))
        push!(saveNamesMutInfoChargeNode, plotHeatmap(corrResults["I2_k_N"], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$I_2^{(c)}(k_\mathrm{N},\vec{k})$", label(W_val)))
        push!(saveNamesMutInfoChargeAntiNode, plotHeatmap(corrResults["I2_k_AN"], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), L"$I_2^{(c)}(k_\mathrm{AN},\vec{k})$", label(W_val)))
    end
    run(`pdfunite $(saveNamesCorr) correlations.pdf`)
    run(`pdfunite $(saveNamesVneSpin) vnEntropySpin.pdf`)
    run(`pdfunite $(saveNamesMutInfoSpinImp) mutInfoSpinImp.pdf`)
    run(`pdfunite $(saveNamesMutInfoSpinNode) mutInfoSpinNode.pdf`)
    run(`pdfunite $(saveNamesMutInfoSpinAntiNode) mutInfoSpinAntiNode.pdf`)
    run(`pdfunite $(saveNamesVneCharge) vnEntropyCharge.pdf`)
    run(`pdfunite $(saveNamesMutInfoChargeImp) mutInfoChargeImp.pdf`)
    run(`pdfunite $(saveNamesMutInfoChargeNode) mutInfoChargeNode.pdf`)
    run(`pdfunite $(saveNamesMutInfoChargeAntiNode) mutInfoChargeAntiNode.pdf`)
end

@time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ)
@time probe(kondoJArrays, dispersion)
@time corr(kondoJArrays, dispersion)
