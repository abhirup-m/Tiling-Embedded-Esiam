using Distributed

if length(Sys.cpu_info()) > 10 && nprocs() == 1
    addprocs(div(length(Sys.cpu_info()), 10))
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
WmaxSize = 500

numShells = 1
size_BZ = 41
# W_val_arr = -1.0 .* [0.0] ./ size_BZ
W_val_arr = -1.0 .* [0, 3.5, 7.13, 7.3, 7.5, 7.564] ./ size_BZ
# W_val_arr = -1.0 .* [0, 2.8, 5.6, 5.7, 5.82, 5.89, 5.92] ./ size_BZ
x_arr = collect(range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi)
label(W_val) = L"$W/J=%$(round(W_val/J_val, digits=2))$\n$M_s=%$(maxSize)$"

function RGFlow(W_val_arr, size_BZ)
    kondoJArrays = Dict{Float64, Array{Float64, 3}}()
    results = @showprogress desc="rg flow" pmap(w -> momentumSpaceRG(size_BZ, omega_by_t, J_val, w, orbitals), W_val_arr)
    dispersion = results[1][2]
    for (result, W_val) in zip(results, W_val_arr)
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

    spinCorrelation = Dict("SF" => (nothing, (i, j) -> [
                                             ("nn", [1, 2, 2 * i + 1, 2 * i + 1], -0.25),
                                             ("nn", [1, 2, 2 * i + 2, 2 * i + 2], 0.25),
                                             ("nn", [2, 1, 2 * i + 1, 2 * i + 1], 0.25),
                                             ("nn", [2, 1, 2 * i + 1, 2 * i + 2], -0.25),
                                             ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1], -0.5),
                                             ("+-+-", [2, 1, 2 * i + 1, 2 * i + 2], -0.5),
                                            ]
                                   )
                          )
    chargeCorrelation = Dict(
                             "doubOcc" => (nothing, (i, j) -> [("nn", [2 * i + 1, 2 * i + 2], 1.), ("hh", [2 * i + 1, 2 * i + 2], 0.)]),
                             "cfnode" => ((-π/2, -π/2), (i, j) -> [("++--", [2 * i + 1, 2 * i + 2, 2 * j + 2, 2 * j + 1], 1.), 
                                                                ("++--", [2 * j + 1, 2 * j + 2, 2 * i + 2, 2 * i + 1], 1.)
                                                               ]),
                             "cfantinode" => ((-π, 0.), (i, j) -> [("++--", [2 * i + 1, 2 * i + 2, 2 * j + 2, 2 * j + 1], 1.), 
                                                                    ("++--", [2 * j + 1, 2 * j + 2, 2 * i + 2, 2 * i + 1], 1.)
                                                                   ])
                            )
    vneDef = Dict("vne_k" => i -> [2 * i + 1, 2 * i + 2])
    mutInfoDef = Dict(
                      "I2_k_d" => (nothing, (i, j) -> ([1, 2], [2 * i + 1, 2 * i + 2])),
                      "I2_k_N" => ((-π/2, -π/2), (i, j) -> ([2 * j + 1, 2 * j + 2], [2 * i + 1, 2 * i + 2])),
                      "I2_k_AN" => ((-π, 0.), (i, j) -> ([2 * j + 1, 2 * j + 2], [2 * i + 1, 2 * i + 2])),
                     )
    saveNames = Dict(name => [] for name in ["SF", "vne_k", "I2_k_d", "I2_k_N", "I2_k_AN", "doubOcc", "cfnode", "cfantinode"])
    plotTitles = Dict("SF" => L"$\chi_s(d, \vec{k})$",
                      "vne_k" => L"$\mathrm{S}_\mathrm{EE}^{(s)}(\vec{k})$",
                      "I2_k_d" => L"$I_2^{(s)}(d,\vec{k})$",
                      "I2_k_N" => L"$I_2^{(s)}(k_\mathrm{N},\vec{k})$",
                      "I2_k_AN" => L"$I_2^{(s)}(k_\mathrm{AN},\vec{k})$",
                      "doubOcc" => L"$\langle n_{\vec{k}\uparrow} n_{\vec{k}\downarrow}\rangle$",
                      "cfnode" => L"$\langle C_{\vec{k}}^+ C_{k_\mathrm{N}}^- + \mathrm{h.c.}\rangle$",
                      "cfantinode" => L"$\langle C_{\vec{k}}^+ C_{k_\mathrm{AN}}^- + \mathrm{h.c.}\rangle$",
                     )

    for W_val in W_val_arr

        hamiltDetails = Dict(
                             "dispersion" => dispersion,
                             "kondoJArray" => kondoJArrays[W_val][:, :, end],
                             "orbitals" => orbitals,
                             "size_BZ" => size_BZ,
                             "bathIntForm" => bathIntForm,
                            )
        for effective_Wval in unique([-0.0, W_val])
            println("\n W = $(W_val), eff_W=$(effective_Wval), $(numShells) shells, maxSize = $(maxSize)")
            hamiltDetails["W_val"] = effective_Wval
            effectiveNumShells = W_val == 0 ? numShells : 1
            effectiveMaxSize = W_val ≠ 0 ? (maxSize > WmaxSize ? WmaxSize : maxSize) : maxSize
            corrResults = nothing
            if effective_Wval == W_val == 0 # case of W = 0, for both spin and charge
                corrResults, _ = correlationMap(hamiltDetails, effectiveNumShells, merge(spinCorrelation, chargeCorrelation),
                                                effectiveMaxSize; vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef, bathIntLegs=3)
            elseif effective_Wval == 0 && W_val != 0 # case of W != 0, but setting effective W to 0 for spin
                corrResults, _ = correlationMap(hamiltDetails, effectiveNumShells, spinCorrelation, effectiveMaxSize;
                                                vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef, bathIntLegs=3)
            else # case of W != 0 and considering the actual W as effective W, for charge
                corrResults, _ = correlationMap(hamiltDetails, effectiveNumShells, chargeCorrelation, effectiveMaxSize;
                                                bathIntLegs=3)
            end

            for name in keys(corrResults)
                push!(saveNames[name], plotHeatmap(corrResults[name], (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"), plotTitles[name], label(W_val)))
            end
        end
    end
    f = open("saveData.txt", "a")
    for (name, files) in saveNames
        if isempty(files)
            continue
        end
        shellCommand = "pdfunite $(join(files, " ")) $(name).pdf"
        run(`sh -c $(shellCommand)`)
        write(f, shellCommand*"\n")
    end
    close(f)
end

@time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ)
@time probe(kondoJArrays, dispersion)
@time corr(kondoJArrays, dispersion)
