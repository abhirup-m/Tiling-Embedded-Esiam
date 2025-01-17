using Distributed, Random

if length(Sys.cpu_info()) > 10 && nprocs() == 1
    addprocs(20)
end
@everywhere using LinearAlgebra, CSV, JLD2, FileIO, CodecZlib

@everywhere include("./source/constants.jl")
include("./source/helpers.jl")
@everywhere include("./source/rgFlow.jl")
include("./source/probes.jl")
include("./source/plotting.jl")

global J_val = 0.1
@everywhere global orbitals = ("p", "p")
maxSize = 600
WmaxSize = 600

colmap = reverse(ColorSchemes.cherry)
numShells = 1
bathIntLegs = 1
NiceValues(size_BZ) = Dict{Int64, Vector{Float64}}(
                         13 => -1.0 .* [0., 1., 1.5, 1.55, 1.6, 1.61] ./ size_BZ,
                         25 => -1.0 .* [0, 3.63, 3.7, 3.8, 3.88, 3.9] ./ size_BZ,
                         33 => -1.0 .* [0, 2.8, 5.6, 5.89, 5.92, 5.93] ./ size_BZ,
                         41 => -1.0 .* [0, 3.5, 7.13, 7.3, 7.5, 7.564, 7.6] ./ size_BZ,
                         49 => -1.0 .* [0, 4.1, 8.19, 8.55, 8.77, 8.8] ./ size_BZ,
                         57 => -1.0 .* [0, 5., 10.2, 10.6, 10.852] ./ size_BZ,
                         69 => -1.0 .* [0, 12.5, 13.1, 13.34] ./ size_BZ,
                        )[size_BZ]

get_x_arr(size_BZ) = collect(range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi)
getlabelInt(W_val, size_BZ) = L"$W/J=%$(round(W_val/J_val, digits=2))$"
getlabelSize(W_val, size_BZ) = L"$\frac{W}{J}=%$(round(W_val/J_val, digits=2))$\n$L=%$(size_BZ)$"

function RGFlow(
        W_val_arr::Vector{Float64},
        size_BZ::Int64;
        loadData::Bool=false,
    )
    kondoJArrays = Dict{Float64, Array{Float64, 3}}()
    results = @showprogress desc="rg flow" pmap(w -> momentumSpaceRG(size_BZ, OMEGA_BY_t, J_val, w, orbitals; loadData=loadData), W_val_arr)
    dispersion = results[1][2]
    for (result, W_val) in zip(results, W_val_arr)
        averageKondoScale = sum(abs.(result[1][:, :, 1])) / length(result[1][:, :, 1])
        @assert averageKondoScale > RG_RELEVANCE_TOL
        result[1][:, :, end] .= ifelse.(abs.(result[1][:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, result[1][:, :, end], 0)
        kondoJArrays[W_val] = result[1]
    end
    return kondoJArrays, dispersion
end

function ScattProb(size_BZ::Int64)
    x_arr = get_x_arr(size_BZ)
    W_val_arr = NiceValues(size_BZ)
    @time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ)
    saveNames = String[]
    for W_val in W_val_arr
        results_scaled, results_bool = scattProb(size_BZ, kondoJArrays[W_val], dispersion)

        quadrantResult = results_bool[filter(p -> all(map1DTo2D(p, size_BZ) .≥ 0), 1:size_BZ^2)]
        push!(saveNames, plotHeatmap(abs.(quadrantResult), (x_arr[x_arr .≥ 0], x_arr[x_arr .≥ 0]), (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                                     L"$\Gamma/\Gamma_0$", getlabelInt(W_val, size_BZ), reverse(colmap)))
    end
    println("\n Saved at $(saveNames).")
    run(`pdfunite $(saveNames) scattprob.pdf`)
end


function KondoCouplingMap(
        size_BZ::Int64;
        loadData::Bool=false,
    )
    x_arr = get_x_arr(size_BZ)
    W_val_arr = NiceValues(size_BZ)[1:5]
    @time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=loadData)
    saveNames = Dict(name => [] for name in ["renormalised-map-node", "renormalised-map-antinode"])
    corrResults = Dict(W_val => Dict() for W_val in W_val_arr)
    bareMapNode = nothing
    bareMapAntinode = nothing
    for W_val in W_val_arr
        results, results_bare, results_bool = kondoCoupMap((π/2, π/2), size_BZ, kondoJArrays[W_val]; 
                                                           mapAmong=(kx, ky) -> abs(abs(kx) + abs(ky) - π) < 10^(-0.5),
                                                          )
        bareMapNode = results_bare
        corrResults[W_val] = Dict("renormalised-map-node" => results)
        results, results_bare, results_bool = kondoCoupMap((π/1, 0.), size_BZ, kondoJArrays[W_val]; 
                                                           mapAmong=(kx, ky) -> abs(abs(kx) + abs(ky) - π) < 10^(-0.5),
                                                          )
        bareMapAntinode = results_bare
        corrResults[W_val]["renormalised-map-antinode"] = results
    end
    push!(saveNames["renormalised-map-node"], plotHeatmap(bareMapNode, (x_arr, x_arr), 
                                       (L"$ak_x/\pi$", L"$ak_y/\pi$"), "", 
                                       getlabelInt(0., size_BZ), colmap;
                                       figSize=(400, 250), figPad=(0., 0., 0., 10.),
                                       #=colorScale=log10,=#
                                      ))
    push!(saveNames["renormalised-map-antinode"], plotHeatmap(bareMapAntinode, (x_arr, x_arr), 
                                       (L"$ak_x/\pi$", L"$ak_y/\pi$"), "", 
                                       getlabelInt(0., size_BZ), colmap;
                                       figSize=(400, 250), figPad=(0., 0., 0., 10.),
                                       #=colorScale=log10,=#
                                      ))
    for W_val in W_val_arr
        results = corrResults[W_val]
        for name in keys(results)
            push!(saveNames[name], plotHeatmap(results[name], (x_arr, x_arr), 
                                               (L"$ak_x/\pi$", L"$ak_y/\pi$"), "", 
                                               getlabelInt(W_val, size_BZ), colmap;
                                               figSize=(400, 250), figPad=(0., 0., 0., 10.),
                                               #=colorScale=log10,=#
                                              ))
        end
    end
    f = open("plotData.txt", "a")
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

function ImpurityCorrelations(
        size_BZ::Int64; 
        spinOnly::Bool=false,
        loadData::Bool=false,
        loadId::String="",
    )
    x_arr = get_x_arr(size_BZ)
    W_val_arr = NiceValues(size_BZ)
    @time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=loadData)
    spinCorrelation = Dict("SF" => [nothing, (i, j) -> [
                                             ("nn", [1, 2, 2 * i + 1, 2 * i + 1], -0.25),
                                             ("nn", [1, 2, 2 * i + 2, 2 * i + 2], 0.25),
                                             ("nn", [2, 1, 2 * i + 1, 2 * i + 1], 0.25),
                                             ("nn", [2, 1, 2 * i + 1, 2 * i + 2], -0.25),
                                             ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1], -0.5),
                                             ("+-+-", [2, 1, 2 * i + 1, 2 * i + 2], -0.5),
                                            ]
                                   ]
                          )
    node = map2DTo1D(-π/2, -π/2, size_BZ)
    antinode = map2DTo1D(-π, 0., size_BZ)

    chargeCorrelation = Dict(
                             "doubOcc" => [nothing, (i, j) -> [("nn", [2 * i + 1, 2 * i + 2], 1.), ("hh", [2 * i + 1, 2 * i + 2], 0.)]
                                          ],
                             "cfnode" => [node, (i, j) -> [("++--", [2 * i + 1, 2 * i + 2, 2 * j + 2, 2 * j + 1], 1.), 
                                                                ("++--", [2 * j + 1, 2 * j + 2, 2 * i + 2, 2 * i + 1], 1.)
                                                               ]
                                         ],
                             "cfantinode" => [antinode, (i, j) -> [("++--", [2 * i + 1, 2 * i + 2, 2 * j + 2, 2 * j + 1], 1.), 
                                                                    ("++--", [2 * j + 1, 2 * j + 2, 2 * i + 2, 2 * i + 1], 1.)
                                                                   ]
                                             ]
                            )
    vneDef = Dict("vne_k" => i -> [2 * i + 1, 2 * i + 2])
    mutInfoDef = Dict(
                      "I2_k_d" => (nothing, (i, j) -> ([1, 2], [2 * i + 1, 2 * i + 2])),
                      "I2_k_N" => (node, (i, j) -> ([2 * j + 1, 2 * j + 2], [2 * i + 1, 2 * i + 2])),
                      "I2_k_AN" => (antinode, (i, j) -> ([2 * j + 1, 2 * j + 2], [2 * i + 1, 2 * i + 2])),
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

    corrResults = Dict{Float64, Dict{String, Vector{Float64}}}(W_val => Dict() for W_val in W_val_arr)
    for W_val in W_val_arr

        hamiltDetails = Dict(
                             "dispersion" => dispersion,
                             "kondoJArray" => kondoJArrays[W_val][:, :, end],
                             "orbitals" => orbitals,
                             "size_BZ" => size_BZ,
                             "bathIntForm" => bathIntForm,
                            )
        for effective_Wval in unique([-0.0, W_val])
            if spinOnly && effective_Wval ≠ 0
                continue
            end
            println("\n W = $(W_val), eff_W=$(effective_Wval), $(numShells) shells, maxSize = $(maxSize)")
            hamiltDetails["W_val"] = effective_Wval
            effectiveNumShells = W_val == 0 ? numShells : 1
            effectiveMaxSize = W_val ≠ 0 ? (maxSize > WmaxSize ? WmaxSize : maxSize) : maxSize
            savePath = joinpath(SAVEDIR, "imp-corr-$(W_val)-$(effective_Wval)-$(size_BZ)-$(effectiveNumShells)-$(effectiveMaxSize)-$(bathIntLegs).jld2")
            if effective_Wval == W_val == 0 # case of W = 0, for both spin and charge
                results, _ = correlationMap(hamiltDetails, effectiveNumShells, merge(spinCorrelation, chargeCorrelation),
                                                effectiveMaxSize, savePath; 
                                                vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef, 
                                                bathIntLegs=bathIntLegs,
                                                noSelfCorr=["cfnode", "cfantinode"], 
                                                addPerStep=1, numProcs=2,
                                                loadData=loadData
                                               )
            elseif effective_Wval == 0 && W_val != 0 # case of W != 0, but setting effective W to 0 for spin
                results, _ = correlationMap(hamiltDetails, effectiveNumShells, spinCorrelation, effectiveMaxSize, savePath;
                                                vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef, 
                                                bathIntLegs=bathIntLegs, 
                                                addPerStep=1, numProcs=2,
                                                loadData=loadData
                                               )
            else # case of W != 0 and considering the actual W as effective W, for charge
                results, _ = correlationMap(hamiltDetails, effectiveNumShells, chargeCorrelation, effectiveMaxSize, savePath;
                                                bathIntLegs=bathIntLegs, noSelfCorr=["cfnode", "cfantinode"], 
                                                addPerStep=1, numProcs=2,
                                                loadData=loadData
                                               )
            end
            merge!(corrResults[W_val], results)
        end
    end
    for W_val in W_val_arr
        for name in keys(corrResults[W_val])
            quadrantResult = corrResults[W_val][name][filter(p -> all(map1DTo2D(p, size_BZ) .≥ 0), 1:size_BZ^2)]
            push!(saveNames[name], plotHeatmap(abs.(quadrantResult), (x_arr[x_arr .≥ 0], x_arr[x_arr .≥ 0]), (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                                               plotTitles[name], getlabelInt(W_val, size_BZ), colmap))
        end
    end
    f = open("plotData.txt", "a")
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


function LocalSpecFunc(
        size_BZ::Int64;
        fixHeight::Bool=false,
        loadData::Bool=false,
    )
    W_val_arr = NiceValues(size_BZ)[[1, 4, 5, 6]]
    if fixHeight
        @assert 0 ∈ W_val_arr
    end
    @time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ)
    freqValues = collect(-15:0.005:15)
    freqValuesZoom1 = 13.
    freqValuesZoom2 = 1.
    specFuncResults = Tuple{LaTeXString, Vector{Float64}}[]
    specFuncResultsTrunc = Tuple{LaTeXString, Vector{Float64}}[]
    standDev = (0.3, 0.2 .+ exp.(abs.(freqValues) ./ maximum(freqValues)))
    resonanceHeight = 0.
    effective_Wval = 0.

    standDevInner = standDev[1]
    for W_val in W_val_arr

        if W_val == W_val_arr[end]
            resonanceHeight = 0.
        end
        effectiveNumShells = W_val == 0 ? numShells : 1
        effectiveMaxSize = W_val ≠ 0 ? (maxSize > WmaxSize ? WmaxSize : maxSize) : maxSize
        savePath = joinpath(SAVEDIR, "imp-specfunc-$(W_val)-$(effective_Wval)-$(size_BZ)-$(effectiveNumShells)-$(effectiveMaxSize)-$(bathIntLegs)-$(maximum(freqValues))-$(length(freqValues))-$(GLOBALFIELD).jld2")
        if ispath(savePath) && loadData
            specFunc = jldopen(savePath)["impSpecFunc"]
        else
            hamiltDetails = Dict(
                                 "dispersion" => dispersion,
                                 "kondoJArray" => kondoJArrays[W_val][:, :, end],
                                 "orbitals" => orbitals,
                                 "size_BZ" => size_BZ,
                                 "bathIntForm" => bathIntForm,
                                 "W_val" => effective_Wval,
                                 "globalField" => GLOBALFIELD,
                                )
            specFunc, standDevInner = localSpecFunc(hamiltDetails, effectiveNumShells, 
                                                    impSpecFunc, freqValues, standDev[1], standDev[2],
                                                    effectiveMaxSize; resonanceHeight=resonanceHeight,
                                                    heightTolerance=1e-3, bathIntLegs=bathIntLegs,
                                                    addPerStep=1)

            roundDigits = trunc(Int, log(1/maximum(specFunc)) + 7)
            jldsave(savePath; impSpecFunc=round.(specFunc, digits=roundDigits))
        end
        if W_val == 0. && fixHeight
            resonanceHeight = specFunc[freqValues .≥ 0][1]
        end
        standDev = (standDevInner, standDev[2])
        push!(specFuncResults, (getlabelInt(W_val, size_BZ), specFunc[abs.(freqValues) .≤ freqValuesZoom1]))
        push!(specFuncResultsTrunc, (getlabelInt(W_val, size_BZ), specFunc[abs.(freqValues) .≤ freqValuesZoom2]))
    end
    plotSpecFunc(specFuncResults, freqValues[abs.(freqValues) .≤ freqValuesZoom1], "impSpecFunc_$(size_BZ).pdf")
    plotSpecFunc(specFuncResultsTrunc, freqValues[abs.(freqValues) .≤ 1], "impSpecFuncTrunc_$(size_BZ).pdf")
end


function KspaceLocalSpecFunc(
        kstate::Vector{Float64},
        size_BZ::Int64;
        loadData::Bool=false,
    )
    W_val_arr = NiceValues(size_BZ)[[3, 5, 6]]
    @time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ)
    freqValues = collect(-15:0.005:15)
    freqValuesZoom1 = 13.
    freqValuesZoom2 = 1.
    specFuncResults = Tuple{LaTeXString, Vector{Float64}}[]
    specFuncResultsTrunc = Tuple{LaTeXString, Vector{Float64}}[]
    standDev = (0.2, 0.01 .+ exp.(abs.(freqValues) ./ maximum(freqValues)))
    effective_Wval = 0.

    for W_val in W_val_arr

        effectiveNumShells = W_val == 0 ? numShells : 1
        effectiveMaxSize = W_val ≠ 0 ? (maxSize > WmaxSize ? WmaxSize : maxSize) : maxSize
        #=kstate = [-3π/4, -π/4]=#
        savePath = joinpath(SAVEDIR, "klocal-specfunc-$(W_val)-$(effective_Wval)-$(size_BZ)-$(map2DTo1D(kstate..., size_BZ))-$(effectiveNumShells)-$(effectiveMaxSize)-$(bathIntLegs)-$(maximum(freqValues))-$(length(freqValues))-$(GLOBALFIELD).jld2")
        if ispath(savePath) && loadData
            specFunc = jldopen(savePath)["specFuncKstate"]
        else
            hamiltDetailsUp = Dict(
                                 "dispersion" => dispersion,
                                 "kondoJArray" => kondoJArrays[W_val][:, :, end],
                                 "orbitals" => orbitals,
                                 "size_BZ" => size_BZ,
                                 "bathIntForm" => bathIntForm,
                                 "W_val" => effective_Wval,
                                 "globalField" => -GLOBALFIELD/100,
                                )
            hamiltDetailsDown = Dict(
                                 "dispersion" => dispersion,
                                 "kondoJArray" => kondoJArrays[W_val][:, :, end],
                                 "orbitals" => orbitals,
                                 "size_BZ" => size_BZ,
                                 "bathIntForm" => bathIntForm,
                                 "W_val" => effective_Wval,
                                 "globalField" => GLOBALFIELD/100,
                                )
            specFunc = 1. .* (
                               kspaceLocalSpecFunc(hamiltDetailsUp, effectiveNumShells, 
                                                    kspaceSpecFunc, freqValues, standDev[1], standDev[2],
                                                    effectiveMaxSize, kstate; bathIntLegs=bathIntLegs,
                                                    addPerStep=1)
                               #=.+ kspaceLocalSpecFunc(hamiltDetailsDown, effectiveNumShells, =#
                               #=                     kspaceSpecFunc, freqValues, standDev[1], standDev[2],=#
                               #=                     effectiveMaxSize, kstate; bathIntLegs=bathIntLegs,=#
                               #=                     addPerStep=1)=#
                              )

            if maximum(specFunc) > 0
                roundDigits = trunc(Int, log(1/maximum(specFunc)) + 7)
                specFunc = round.(specFunc, digits=roundDigits)
            end
            jldsave(savePath; specFuncKstate=specFunc)
        end
        push!(specFuncResults, (getlabelInt(W_val, size_BZ), specFunc[abs.(freqValues) .≤ freqValuesZoom1]))
        push!(specFuncResultsTrunc, (getlabelInt(W_val, size_BZ), specFunc[abs.(freqValues) .≤ freqValuesZoom2]))
    end
    plotSpecFunc(specFuncResults, freqValues[abs.(freqValues) .≤ freqValuesZoom1], "kspaceSpecFunc_$(size_BZ)-$(map2DTo1D(kstate..., size_BZ)).pdf")
    plotSpecFunc(specFuncResultsTrunc, freqValues[abs.(freqValues) .≤ 1], "kspaceSpecFuncTrunc_$(size_BZ)-$(map2DTo1D(kstate..., size_BZ)).pdf")
end


function RealSpaceOffDiagSpecFunc(
        size_BZ::Int64;
        loadData::Bool=false,
    )
    W_val_arr = NiceValues(size_BZ)[[1, 6]]
    @time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ)
    freqValues = collect(-15:0.005:15)
    freqValuesZoom1 = 13.
    freqValuesZoom2 = 1.
    specFuncResults = Tuple{LaTeXString, Vector{Float64}}[]
    specFuncResultsTrunc = Tuple{LaTeXString, Vector{Float64}}[]
    standDev = (0.1, 0.1 .+ exp.(abs.(freqValues) ./ maximum(freqValues)))
    effective_Wval = 0.

    standDevInner = standDev[1]
    for W_val in W_val_arr
        specFunc = 0 .* freqValues

        effectiveNumShells = W_val == 0 ? numShells : 1
        effectiveMaxSize = W_val ≠ 0 ? (maxSize > WmaxSize ? WmaxSize : maxSize) : maxSize
        fermiSurfacePoints1D = getIsoEngCont(dispersion, 0.)
        fermiSurfacePoints2D = map1DTo2D.(fermiSurfacePoints1D, size_BZ)
        filteredPoints1D = [fermiSurfacePoints1D[i] for (i, (kx, ky)) in enumerate(fermiSurfacePoints2D) if kx + ky ≤ 0 && ky ≥ 0]
        filteredPoints2D = map1DTo2D.(filteredPoints1D, size_BZ)

        @showprogress for (kpoint, kstate) in zip(filteredPoints1D, filteredPoints2D)
            kstate = map1DTo2D(kpoint, size_BZ)
            savePath = joinpath(SAVEDIR, "rlocal-specfunc-$(W_val)-$(effective_Wval)-$(size_BZ)-$(kpoint)-$(effectiveNumShells)-$(effectiveMaxSize)-$(bathIntLegs)-$(maximum(freqValues))-$(length(freqValues))-$(GLOBALFIELD).jld2")
            if ispath(savePath) && loadData
                specFunc = jldopen(savePath)["specFuncKstate"]
            else
                hamiltDetails = Dict(
                                     "dispersion" => dispersion,
                                     "kondoJArray" => kondoJArrays[W_val][:, :, end],
                                     "orbitals" => orbitals,
                                     "size_BZ" => size_BZ,
                                     "bathIntForm" => bathIntForm,
                                     "W_val" => effective_Wval,
                                     "globalField" => GLOBALFIELD,
                                    )
                specFuncKstate = kspaceLocalSpecFunc(hamiltDetails, effectiveNumShells, 
                                                        kspaceSpecFunc, freqValues, standDev[1], standDev[2],
                                                        effectiveMaxSize, kstate; bathIntLegs=bathIntLegs,
                                                        addPerStep=1)
                specFunc .+= 2 * cos(kstate[2]) .* specFuncKstate

                if maximum(specFuncKstate) > 0
                    roundDigits = trunc(Int, log(1/maximum(specFuncKstate)) + 7)
                    specFuncKstate = round.(specFuncKstate, digits=roundDigits)
                end
                jldsave(savePath; specFuncKstate=specFuncKstate)
            end
        end
        standDev = (standDevInner, standDev[2])
        #=specFunc ./= sum(specFunc) * (maximum(freqValues) - minimum(freqValues)) / (length(freqValues) - 1)=#
        push!(specFuncResults, (getlabelInt(W_val, size_BZ), specFunc[abs.(freqValues) .≤ freqValuesZoom1]))
        push!(specFuncResultsTrunc, (getlabelInt(W_val, size_BZ), specFunc[abs.(freqValues) .≤ freqValuesZoom2]))
    end
    plotSpecFunc(specFuncResults, freqValues[abs.(freqValues) .≤ freqValuesZoom1], "impSpecFunc_$(size_BZ).pdf")
    plotSpecFunc(specFuncResultsTrunc, freqValues[abs.(freqValues) .≤ 1], "impSpecFuncTrunc_$(size_BZ).pdf")
end


function PhaseDiagram(
        size_BZ::Int64,
        tolerance::Float64;
        loadData::Bool=false,
    )
    kondoJVals = 10 .^ (-1.5:0.01:-0.4)
    bathIntVals = collect(-0.0:-0.001:-0.5)
    phaseLabels = ["L-FL", "L-PG", "LM"]
    phaseDiagram = PhaseDiagram(size_BZ, OMEGA_BY_t, kondoJVals, bathIntVals, tolerance, Dict(phaseLabels .=> 1:3); loadData=loadData)
    plotPhaseDiagram(phaseDiagram, Dict(1:3 .=> phaseLabels), (kondoJVals, -1 .* bathIntVals),
                     (L"J/t", L"-W/t"), L"L=%$(size_BZ)", "phaseDiagram.pdf",  
                     colmap[[2, 5, 8]])
end


function TiledSpinCorr(
        size_BZ::Int64;
        loadData::Bool=false,
    )
    x_arr = get_x_arr(size_BZ)
    W_val_arr = NiceValues(size_BZ)
    @time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ)
    node = map2DTo1D(-π/2, -π/2, size_BZ)
    antinode = map2DTo1D(-π, 0., size_BZ)
    spinCorrelation = Dict("SF1" => [node, (i, j) -> [("+-+-", [2 * i + 1, 2, 2, 2 * j + 1], 1.)]]) # c^†_{k↑}c_{d↓}c^†_{d↓}c_{q↑}
    spinCorrelation["SF4"] = [node, (i, j) -> [("+-+-", [2 * i + 1, 2, 2 * j + 2, 1], 1.)]]         # c^†_{k↑}c_{d↓}c^†_{q↓}c_{d↑}
    spinCorrelation["SF7"] = [node, (i, j) -> [("+-+-", [2 * i + 1, 2, 2 * j + 2, 2 * j + 1], 1.)]] # c^†_{k↑}c_{d↓}c^†_{q↓}c_{q↑}
    spinCorrelation["SF5"] = [node, (i, j) -> [("+-+-", [2 * i + 1, 2 * i + 2, 2 * j + 2, 1], 1.)]] # c^†_{k↑}c_{k↓}c^†_{q↓}c_{d↑}
    spinCorrelation["SF6"] = [node, (i, j) -> [("+-+-", [2 * i + 1, 2 * i + 2, 2, 2 * j + 1], 1.)]] # c^†_{k↑}c_{k↓}c^†_{d↓}c_{q↑}
    spinCorrelation["SF2"] = [node, (i, j) -> [("+-+-", [1, 2 * i + 2, 2 * j + 2, 1], 1.)]]         # c^†_{d↑}c_{k↓}c^†_{q↓}c_{d↑}
    spinCorrelation["SF3"] = [node, (i, j) -> [("+-+-", [1, 2 * i + 2, 2, 2 * j + 1], 1.)]]         # c^†_{d↑}c_{k↓}c^†_{d↓}c_{q↑}
    spinCorrelation["SF8"] = [node, (i, j) -> [("+-+-", [1, 2 * i + 2, 2 * j + 2, 2 * j + 1], 1.)]] # c^†_{d↑}c_{k↓}c^†_{q↓}c_{q↑}

    saveNames = Dict(name => [] for name in ["tiledSF"])
    plotTitles = Dict("tiledSF" => L"$\chi_s(k_\text{N}, \vec{k})$",
                     )

    name = "tiledSF"
    effective_Wval = 0.
    hamiltDetailsDict = Dict(W_val => Dict(
                                           "dispersion" => dispersion,
                                           "kondoJArray" => kondoJArrays[W_val][:, :, end],
                                           "orbitals" => orbitals,
                                           "size_BZ" => size_BZ,
                                           "bathIntForm" => bathIntForm,
                                           "W_val" => effective_Wval,
                                          )
                             for W_val in W_val_arr
                            )

    effectiveNumShells = Dict(W_val => ifelse(W_val == 0, numShells, 1) for W_val in W_val_arr)
    savePaths = Dict(W_val => joinpath(SAVEDIR, "tiled-spin-corr-$(W_val)-$(effective_Wval)-$(size_BZ)-$(effectiveNumShells[W_val])-$(maxSize)-$(bathIntLegs).jld2") for W_val in W_val_arr)
    resultsArr = pmap(W_val -> correlationMap(hamiltDetailsDict[W_val], effectiveNumShells[W_val], spinCorrelation, maxSize,savePaths[W_val]; loadData=loadData), W_val_arr)
    for (W_val, (corrResults, _)) in zip(W_val_arr, resultsArr)
        tiledCorrelation = sum(values(corrResults)) / length(values(corrResults))
        quadrantResult = tiledCorrelation[filter(p -> all(map1DTo2D(p, size_BZ) .≥ 0), 1:size_BZ^2)]
        push!(saveNames[name], plotHeatmap(abs.(quadrantResult), (x_arr[x_arr .≥ 0], x_arr[x_arr .≥ 0]), (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                                           plotTitles[name], getlabelInt(W_val, size_BZ), colmap))
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

size_BZ = 49
#=@time ScattProb(size_BZ)=#
#=@time KondoCouplingMap(size_BZ; loadData=false)=#
#=@time ImpurityCorrelations(size_BZ; loadData=false)=#
#=@time LocalSpecFunc(size_BZ; loadData=false, fixHeight=true)=#
@time KspaceLocalSpecFunc([-π/2, -π/2], size_BZ; loadData=false)
@time KspaceLocalSpecFunc([-π, 0.], size_BZ; loadData=false)
@time KspaceLocalSpecFunc([-π/4, -3π/4], size_BZ; loadData=false)
#=@time RealSpaceOffDiagSpecFunc(size_BZ; loadData=false)=#
#=@time TiledSpinCorr(size_BZ; loadData=false)=#
#=@time PhaseDiagram(size_BZ, 1e-3; loadData=false)=#
