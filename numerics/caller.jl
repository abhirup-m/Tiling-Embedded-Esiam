using Distributed, Random

if length(Sys.cpu_info()) > 10 && nprocs() == 1
    addprocs(2)
end
@everywhere using LinearAlgebra, CSV, JLD2, FileIO, CodecZlib

@everywhere include("./source/constants.jl")
include("./source/helpers.jl")
@everywhere include("./source/rgFlow.jl")
include("./source/probes.jl")
include("./source/plotting.jl")

global J_val = 0.1
@everywhere global orbitals = ("p", "p")
maxSize = 500
WmaxSize = 500

colmap = ColorSchemes.thermal # ColorSchemes.thermal # reverse(ColorSchemes.cherry)
numShells = 1
bathIntLegs = 2
NiceValues(size_BZ) = Dict{Int64, Vector{Float64}}(
                         13 => -1.0 .* [0., 1., 1.5, 1.55, 1.6, 1.61] ./ size_BZ,
                         25 => -1.0 .* [0, 3.63, 3.7, 3.8, 3.88, 3.9] ./ size_BZ,
                         33 => -1.0 .* [0, 2.8, 5.6, 5.89, 5.92, 5.93] ./ size_BZ,
                         41 => -1.0 .* [0, 3.5, 7.13, 7.3, 7.5, 7.564, 7.6] ./ size_BZ,
                         49 => -1.0 .* [0, 4.1, 8.19, 8.55, 8.77, 8.8] ./ size_BZ,
                         57 => -1.0 .* [0, 5., 10.2, 10.6, 10.852] ./ size_BZ,
                         69 => -1.0 .* [0, 6., 12.5, 13.1, 13.34, 13.4] ./ size_BZ,
                         77 => -1.0 .* [0., 14.04, 14.5, 14.99] ./ size_BZ,
                        )[size_BZ]
transitionValues(size_BZ) = Dict{Int64, Float64}(
                         13 => -1.0 * 1.6 / size_BZ,
                         25 => -1.0 * 3.88 / size_BZ,
                         33 => -1.0 * 5.92 / size_BZ,
                         41 => -1.0 * 7.564 / size_BZ,
                         49 => -1.0 * 8.77 / size_BZ,
                         57 => -1.0 * 10.852 / size_BZ,
                         69 => -1.0 * 13.34 / size_BZ,
                         77 => -1.0 * 14.99 / size_BZ,
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


function ScattProb(
        size_BZ::Int64; 
        loadData::Bool=false
    )
    x_arr = get_x_arr(size_BZ)
    W_val_arr = NiceValues(size_BZ)
    @time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=loadData)
    saveNames = String[]
    results = [ScattProb(size_BZ, kondoJArrays[W_val], dispersion)[2] for W_val in W_val_arr]
    quadrantResults = [result[filter(p -> all(map1DTo2D(p, size_BZ) .≥ 0), 1:size_BZ^2)] 
                       for result in results]
    nonNaNData = filter(!isnan, vcat(values(quadrantResults)...))
    if minimum(nonNaNData) < maximum(nonNaNData)
        colorbarLimits = (minimum(nonNaNData), maximum(nonNaNData)) .+ 0.
    else
        colorbarLimits = (minimum(nonNaNData)*(1-1e-5) - 1e-5, minimum(nonNaNData)*(1+1e-5) + 1e-5)
    end
    for (W_val, result) in zip(W_val_arr, quadrantResults)
        push!(saveNames, 
              plotHeatmap(result, 
                          (x_arr[x_arr .≥ 0], x_arr[x_arr .≥ 0]),
                          (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                          L"$\Gamma/\Gamma_0$", 
                          getlabelInt(W_val, size_BZ), 
                          colmap;
                          colorbarLimits=colorbarLimits,
                         )
             )
    end
    println("\n Saved at $(saveNames).")
    run(`pdfunite $(saveNames) scattprob.pdf`)
end


function KondoCouplingMap(
        size_BZ::Int64;
    )
    x_arr = get_x_arr(size_BZ)
    W_val_arr = NiceValues(size_BZ)[[1, 3, 4]]
    @time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=true)
    saveNames = Dict(name => [] for name in ["node", "antinode"])
    titles = Dict(
                  "node" => L"J(k_\mathrm{N}, \vec{k})",
                  "antinode" => L"J(k_\mathrm{AN}, \vec{k})",
                 )
    corrResults = Dict(W_val => Dict() for W_val in W_val_arr)
    bareResults = Dict("node" => Float64[], "antinode" => Float64[])
    for W_val in W_val_arr
        results, bareResults["node"], results_bool = KondoCoupMap((π/2, π/2), size_BZ, kondoJArrays[W_val]; 
                                                           #=mapAmong=(kx, ky) -> abs(abs(kx) + abs(ky) - π) < 10^(-0.5),=#
                                                          )
        corrResults[W_val] = Dict("node" => results)
        results, bareResults["antinode"], results_bool = KondoCoupMap((π/1, 0.), size_BZ, kondoJArrays[W_val]; 
                                                           #=mapAmong=(kx, ky) -> abs(abs(kx) + abs(ky) - π) < 10^(-0.5),=#
                                                          )
        corrResults[W_val]["antinode"] = results
    end
    for name in keys(saveNames)
        nonNaNData = filter(!isnan, vcat([corrResults[W_val][name] for W_val in W_val_arr]...))
        append!(nonNaNData, filter(!isnan, bareResults[name]))
        if minimum(nonNaNData) < maximum(nonNaNData)
            colorbarLimits = (minimum(nonNaNData), maximum(nonNaNData)) .+ 0.
        else
            colorbarLimits = (minimum(nonNaNData)*(1-1e-5) - 1e-5, minimum(nonNaNData)*(1+1e-5) + 1e-5)
        end

        push!(saveNames[name], 
              plotHeatmap(bareResults[name], (x_arr, x_arr),
                          (L"$ak_x/\pi$", L"$ak_y/\pi$"), titles[name],
                          getlabelInt(0., size_BZ), colmap;
                          figSize=(450, 350),
                          #=colorbarLimits=colorbarLimits,=#
                          colorScale=Makie.pseudolog10,
                                          ))
        for W_val in W_val_arr
            push!(saveNames[name], plotHeatmap(corrResults[W_val][name], (x_arr, x_arr), 
                                               (L"$ak_x/\pi$", L"$ak_y/\pi$"), titles[name], 
                                               getlabelInt(W_val, size_BZ), colmap;
                                               figSize=(450, 350),
                                               #=colorbarLimits=colorbarLimits,=#
                                               colorScale=Makie.pseudolog10,
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


function AuxiliaryCorrelations(
        size_BZ::Int64; 
        spinOnly::Bool=false,
        loadData::Bool=false,
        loadId::String="",
    )
    x_arr = get_x_arr(size_BZ)
    W_val_arr = NiceValues(size_BZ)
    @time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=true)
    spinCorrelation = Dict{String, Tuple{Union{Nothing, Int64}, Function}}("SF" => (nothing, (i, j) -> [
                                             ("nn", [1, 2, 2 * i + 1, 2 * i + 1], -0.25),
                                             ("nn", [1, 2, 2 * i + 2, 2 * i + 2], 0.25),
                                             ("nn", [2, 1, 2 * i + 1, 2 * i + 1], 0.25),
                                             ("nn", [2, 1, 2 * i + 1, 2 * i + 2], -0.25),
                                             ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1], -0.5),
                                             ("+-+-", [2, 1, 2 * i + 1, 2 * i + 2], -0.5),
                                            ]
                                   )
                          )
    node = map2DTo1D(-π/2, -π/2, size_BZ)
    antinode = map2DTo1D(-π, 0., size_BZ)

    chargeCorrelation = Dict{String, Tuple{Union{Nothing, Int64}, Function}}(
                             "doubOcc" => (nothing, (i, j) -> [("nn", [2 * i + 1, 2 * i + 2], 1.), ("hh", [2 * i + 1, 2 * i + 2], 0.)]
                                          ),
                             "cfnode" => (node, (i, j) -> [("++--", [2 * i + 1, 2 * i + 2, 2 * j + 2, 2 * j + 1], 1.), 
                                                                ("++--", [2 * j + 1, 2 * j + 2, 2 * i + 2, 2 * i + 1], 1.)
                                                               ]
                                         ),
                             "cfantinode" => (antinode, (i, j) -> [("++--", [2 * i + 1, 2 * i + 2, 2 * j + 2, 2 * j + 1], 1.), 
                                                                    ("++--", [2 * j + 1, 2 * j + 2, 2 * i + 2, 2 * i + 1], 1.)
                                                                   ]
                                             )
                            )
    vneDef = Dict(
                  "vne_k" => i -> [2 * i + 1, 2 * i + 2]
                 )
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
                             "globalField" => GLOBALFIELD,
                            )
        #=for effective_Wval in unique([-0.0,])=#
        for effective_Wval in unique([-0.0, W_val])
            if spinOnly && effective_Wval ≠ 0
                continue
            end
            hamiltDetails["W_val"] = effective_Wval
            effectiveNumShells = W_val == 0 ? numShells : 1
            effectiveMaxSize = effective_Wval ≠ 0 ? (maxSize > WmaxSize ? WmaxSize : maxSize) : maxSize
            println("\n W = $(W_val), eff_W=$(effective_Wval), $(effectiveNumShells) shells, maxSize = $(effectiveMaxSize)")
            savePath = joinpath(SAVEDIR, "imp-corr-$(W_val)-$(effective_Wval)-$(size_BZ)-$(effectiveNumShells)-$(effectiveMaxSize)-$(bathIntLegs).jld2")
            if effective_Wval == W_val == 0 # case of W = 0, for both spin and charge
                results, _ = AuxiliaryCorrelations(hamiltDetails, effectiveNumShells, merge(spinCorrelation, chargeCorrelation),
                                                effectiveMaxSize, savePath; 
                                                vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef, 
                                                bathIntLegs=bathIntLegs,
                                                noSelfCorr=["cfnode", "cfantinode"], 
                                                addPerStep=1, numProcs=2,
                                                loadData=loadData
                                               )
            elseif effective_Wval == 0 && W_val != 0 # case of W != 0, but setting effective W to 0 for spin
                results, _ = AuxiliaryCorrelations(hamiltDetails, effectiveNumShells, spinCorrelation, effectiveMaxSize, savePath;
                                                vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef, 
                                                bathIntLegs=bathIntLegs, 
                                                addPerStep=1, numProcs=2,
                                                loadData=loadData
                                               )
            else # case of W != 0 and considering the actual W as effective W, for charge
                results, _ = AuxiliaryCorrelations(hamiltDetails, effectiveNumShells, chargeCorrelation, effectiveMaxSize, savePath;
                                                bathIntLegs=bathIntLegs, noSelfCorr=["cfnode", "cfantinode"], 
                                                addPerStep=1, numProcs=2,
                                                loadData=loadData
                                               )
            end
            merge!(corrResults[W_val], results)
        end
    end
    for (name, saveName) in saveNames
        quadrantResults = Dict(W_val => corrResults[W_val][name][filter(p -> all(map1DTo2D(p, size_BZ) .≥ 0), 1:size_BZ^2)] for W_val in W_val_arr)

        nonNaNData = filter(!isnan, vcat([quadrantResults[W_val] for W_val in W_val_arr]...))
        if minimum(nonNaNData) < maximum(nonNaNData)
            colorbarLimits = (minimum(nonNaNData), maximum(nonNaNData)) .+ 0.
        else
            colorbarLimits = (minimum(nonNaNData)*(1-1e-5) - 1e-5, minimum(nonNaNData)*(1+1e-5) + 1e-5)
        end
        for W_val in W_val_arr
            push!(saveName, plotHeatmap(quadrantResults[W_val],
                                        (x_arr[x_arr .≥ 0], x_arr[x_arr .≥ 0]),
                                        (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                                        plotTitles[name], 
                                        getlabelInt(W_val, size_BZ), 
                                        colmap;
                                        colorbarLimits=colorbarLimits,
                                       )
                 )
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


function AuxiliaryLocalSpecfunc(
        size_BZ::Int64;
        fixHeight::Bool=false,
        loadData::Bool=false,
    )
    W_val_arr = NiceValues(size_BZ)[[1,]]
    if fixHeight
        @assert 0 ∈ W_val_arr
    end
    kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=true)
    freqValues = collect(-20:0.0005:20)
    freqValuesZoom1 = 10.
    freqValuesZoom2 = 0.4
    specFuncFull = Tuple{LaTeXString, Vector{Float64}}[]
    specFuncTrunc = Tuple{LaTeXString, Vector{Float64}}[]
    imagSelfEnergy = Tuple{LaTeXString, Vector{Float64}}[]
    imagSelfEnergyTrunc = Tuple{LaTeXString, Vector{Float64}}[]
    realSelfEnergy = Tuple{LaTeXString, Vector{Float64}}[]
    standDev = (0.1, 0.0 .+ exp.(abs.(freqValues) ./ maximum(freqValues)))
    targetHeight = 0.
    effective_Wval = 0.
    standDevGuess = 0.1

    standDevInner = standDev[1]

    deltaFunctionGamma = 1e-2
    nonIntSpecFunc = nothing
    fermiPoints = getIsoEngCont(dispersion, 0.)
    for W_val in W_val_arr

        if abs(W_val) > abs(transitionValues(size_BZ))
            targetHeight = 0.
            kondoTemp = 1.
        else
            kondoTemp = (sum(abs.(kondoJArrays[W_val][fermiPoints, fermiPoints, end])) 
                         / sum(abs.(kondoJArrays[W_val][fermiPoints, fermiPoints, 1]))
                        )^0.75
        end
        effectiveNumShells = W_val == 0 ? numShells : 1
        savePath = joinpath(SAVEDIR, "imp-specfunc-$(W_val)-$(effective_Wval)-$(size_BZ)-$(effectiveNumShells)-$(maxSize)-$(bathIntLegs)-$(maximum(freqValues))-$(length(freqValues))-$(GLOBALFIELD).jld2")
        if ispath(savePath) && loadData
            specFunc = jldopen(savePath)["impSpecFunc"]
            println("Collected W=$(W_val) from saved data.")
        else
            hamiltDetails = Dict(
                                 "dispersion" => dispersion,
                                 "kondoJArray" => kondoJArrays[W_val][:, :, end],
                                 "orbitals" => orbitals,
                                 "size_BZ" => size_BZ,
                                 "bathIntForm" => bathIntForm,
                                 "W_val" => W_val,
                                 "globalField" => GLOBALFIELD,
                                 "J_val_bare" => J_val,
                                )
            @time specFunc, standDevGuess = AuxiliaryLocalSpecfunc(hamiltDetails, effectiveNumShells, 
                                                    ImpurityExcitationOperators, freqValues, standDev[1], 
                                                    standDev[2], maxSize; 
                                                    targetHeight=targetHeight, heightTolerance=1e-3,
                                                    bathIntLegs=bathIntLegs, addPerStep=1, 
                                                    standDevGuess=standDevInner,
                                                    kondoTemp=kondoTemp,
                                                   )
            roundDigits = trunc(Int, log(1/maximum(specFunc)) + 7)
            jldsave(savePath; impSpecFunc=round.(specFunc, digits=roundDigits))
        end
        if W_val == 0. && fixHeight
            targetHeight = specFunc[freqValues .≥ 0][1]
        end
        standDev = (standDevInner, standDev[2])
        push!(specFuncFull, (getlabelInt(W_val, size_BZ), specFunc[abs.(freqValues) .≤ freqValuesZoom1]))
        push!(specFuncTrunc, (getlabelInt(W_val, size_BZ), specFunc[abs.(freqValues) .≤ freqValuesZoom2]))

        if isnothing(nonIntSpecFunc)
            nonIntGamma = 2 / (π * specFunc[freqValues .≥ 0][1])
            nonIntSpecFunc = (1/π) * (nonIntGamma / 2) ./ (freqValues.^2 .+ (nonIntGamma / 2)^2)
            selfEnergyTol = 1e-2
            selfEnergy = SelfEnergy(nonIntSpecFunc, specFunc, freqValues)
            while abs(imag(selfEnergy)[freqValues .≥ 0][1]) > selfEnergyTol
                if imag(selfEnergy)[freqValues .≥ 0][1] > 0
                    nonIntGamma /= 1.005
                else
                    nonIntGamma *= 1.005
                end
                nonIntSpecFunc = (1/π) * (nonIntGamma / 2) ./ (freqValues.^2 .+ (nonIntGamma / 2)^2)
                selfEnergy = SelfEnergy(nonIntSpecFunc, specFunc, freqValues)
            end
        end

        selfEnergy = SelfEnergy(nonIntSpecFunc, specFunc, freqValues)
        push!(realSelfEnergy, (getlabelInt(W_val, size_BZ), real(selfEnergy)[abs.(freqValues) .≤ freqValuesZoom1]))
        push!(imagSelfEnergy, (getlabelInt(W_val, size_BZ), imag(selfEnergy)[abs.(freqValues) .≤ freqValuesZoom1]))
        push!(imagSelfEnergyTrunc, (getlabelInt(W_val, size_BZ), imag(selfEnergy)[abs.(freqValues) .≤ freqValuesZoom2]))

        for (k, v) in imagSelfEnergy
            v[abs.(v) .> 10] .= sign.(v)[abs.(v) .> 10] * 10
        end
    end
    plotLines(specFuncFull, 
              freqValues[abs.(freqValues) .≤ freqValuesZoom1], 
              L"\omega", 
              L"A(\omega)",
              "impSpecFunc_$(size_BZ).pdf",
             )
    plotLines(specFuncTrunc, 
              freqValues[abs.(freqValues) .≤ freqValuesZoom2],
              L"\omega", 
              L"A(\omega)",
              "impSpecFuncTrunc_$(size_BZ).pdf",
             )
    plotLines(realSelfEnergy, 
              freqValues[abs.(freqValues) .≤ freqValuesZoom1], 
              L"\omega", 
              L"\Sigma^\prime(\omega)",
              "sigmaReal_$(size_BZ).pdf",
             )
    plotLines(imagSelfEnergy, 
              freqValues[abs.(freqValues) .≤ freqValuesZoom1], 
              L"\omega", 
              L"\Sigma^{\prime\prime}(\omega)",
              "sigmaImag_$(size_BZ).pdf";
              ylimits=(-10., 10.),
             )
    plotLines(imagSelfEnergyTrunc, 
              freqValues[abs.(freqValues) .≤ freqValuesZoom2], 
              L"\omega", 
              L"\Sigma^{\prime\prime}(\omega)",
              "sigmaImag-trunc_$(size_BZ).pdf",
              #=ylimits=(-1., 1.),=#
             )
end


function AuxiliaryMomentumSpecfunc(
        size_BZ::Int64, 
        kpoint::NTuple{2, Float64};
        loadData::Bool=false,
    )
    W_val_arr = NiceValues(size_BZ)
    #=W_val_arr = NiceValues(size_BZ)[1:1]=#
    @time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=true)
    freqValues = collect(-15:0.005:15)
    freqValuesZoom1 = 6.
    freqValuesZoom2 = 0.3
    standDev = (0.1, 0.0 .+ exp.(0.01 .* abs.(freqValues) ./ maximum(freqValues)))
    effective_Wval = 0.
    results = Dict{Float64, Vector{Float64}}(W_val => 0 .* freqValues for W_val in W_val_arr)
    saveNames = String[]

    for W_val in W_val_arr

        savePath = joinpath(SAVEDIR, "momentumSpecFunc-$(W_val)-$(effective_Wval)-$(size_BZ)-$(numShells)-$(maxSize)-$(bathIntLegs)-$(maximum(freqValues))-$(length(freqValues))-$(kpoint)-$(GLOBALFIELD).jld2")
        if ispath(savePath) && loadData
            results[W_val]["momentumSpecFunc"] = jldopen(savePath)["momentumSpecFunc"]
            println("Collected W=$(W_val) from saved data.")
        else
            hamiltDetails = Dict(
                                 "dispersion" => dispersion,
                                 "kondoJArray" => kondoJArrays[W_val][:, :, end],
                                 "orbitals" => orbitals,
                                 "size_BZ" => size_BZ,
                                 "bathIntForm" => bathIntForm,
                                 "W_val" => effective_Wval,
                                 "globalField" => -GLOBALFIELD,
                                )
            results[W_val] .= LatticeKspaceDOS(hamiltDetails, numShells, 
                                        KspaceExcitationOperators, freqValues, standDev[1], 
                                        standDev[2], maxSize; onlyAt=kpoint,
                                       )
            jldsave(savePath; momentumSpecFunc=results[W_val])
        end
    end

    nonNaNData = filter(!isnan, vcat(values(results)...))
    if minimum(nonNaNData) < maximum(nonNaNData)
        colorbarLimits = (minimum(nonNaNData), maximum(nonNaNData))
    else
        colorbarLimits = (minimum(nonNaNData)*(1-1e-5) - 1e-5, minimum(nonNaNData)*(1+1e-5) + 1e-5)
    end
    plotLines(Dict(getlabelInt(W_val, size_BZ) => results[W_val][abs.(freqValues) .≤ freqValuesZoom1] for W_val in W_val_arr),
              freqValues[abs.(freqValues) .≤ freqValuesZoom1], 
              L"\omega", 
              L"A(\omega)",
              "auxMomentumSpecFunc-$(trunc(kpoint[1], digits=3))-$(trunc(kpoint[2], digits=3))-$(size_BZ)-$(maxSize).pdf",
             )
    plotLines(Dict(getlabelInt(W_val, size_BZ) => results[W_val][abs.(freqValues) .≤ freqValuesZoom2] for W_val in W_val_arr),
              freqValues[abs.(freqValues) .≤ freqValuesZoom2],
              L"\omega", 
              L"A(\omega)",
              "auxMomentumSpecFunc-$(trunc(kpoint[1], digits=3))-$(trunc(kpoint[2], digits=3))-$(size_BZ)-$(maxSize).pdf",
             )
end


function LatticeKspaceDOS(
        size_BZ::Int64;
        loadData::Bool=false,
    )
    x_arr = get_x_arr(size_BZ)
    #=W_val_arr = NiceValues(size_BZ)=#
    W_val_arr = NiceValues(size_BZ)
    kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=true)
    freqValues = collect(-15:0.001:15)
    freqValuesZoom1 = 6.
    freqValuesZoom2 = 0.3
    standDev = (0.1, 0.0 .+ exp.(0.01 .* abs.(freqValues) ./ maximum(freqValues)))
    effective_Wval = 0.
    targetHeight = 0.0
    standDevGuess = 0.1
    results = Dict{Float64, Dict{String, Vector{Float64}}}(W_val => Dict{String, Vector{Float64}}() 
                                                           for W_val in W_val_arr
                                                          )
    saveNames = Dict(name => [] for name in ["kspaceDOS", "quasipRes"])
    plotTitles = Dict("kspaceDOS" => L"$A_{\vec{k}}(\omega\to 0)$",
                      "quasipRes" => L"$Z_{\vec{k}}$",
                     )

    for W_val in W_val_arr
        savePath = joinpath(SAVEDIR, "kspaceDOS-$(W_val)-$(effective_Wval)-$(size_BZ)-$(numShells)-$(maxSize)-$(bathIntLegs)-$(maximum(freqValues))-$(length(freqValues))-$(GLOBALFIELD).jld2")
        if ispath(savePath) && loadData
            results[W_val]["kspaceDOS"] = jldopen(savePath)["kspaceDOS"]
            nodeIndex = map2DTo1D(-π/2, -π/2, size_BZ)
            results[W_val]["quasipRes"] = jldopen(savePath)["quasipRes"]
            targetHeight = jldopen(savePath)["locHeight"]
            println("Collected W=$(W_val) from saved data. $(targetHeight)")
        else
            hamiltDetails = Dict(
                                 "dispersion" => dispersion,
                                 "kondoJArray" => kondoJArrays[W_val][:, :, end],
                                 "orbitals" => orbitals,
                                 "size_BZ" => size_BZ,
                                 "bathIntForm" => bathIntForm,
                                 "W_val" => effective_Wval,
                                 "globalField" => -GLOBALFIELD,
                                )
            specFuncKSpace, specFunc, standDevGuess, results[W_val] = LatticeKspaceDOS(hamiltDetails, numShells, 
                                           KspaceExcitationOperators, freqValues, standDev[1], 
                                           standDev[2], maxSize; bathIntLegs=bathIntLegs,
                                           addPerStep=1, targetHeight=ifelse(abs(W_val) > abs(transitionValues(size_BZ)), 0., targetHeight),
                                           standDevGuess=standDevGuess,
                                          )
            targetHeight = specFunc[freqValues .≥ 0][1]

            jldsave(savePath; 
                    kspaceDOS=results[W_val]["kspaceDOS"],
                    quasipRes=results[W_val]["quasipRes"],
                    locHeight=specFunc[freqValues .≥ 0][1],
                   )
        end
    end

    for name in keys(saveNames)
        quadrantResults = Dict(W_val => results[W_val][name][filter(p -> all(map1DTo2D(p, size_BZ) .≤ 0), 1:size_BZ^2)] 
                               for W_val in W_val_arr)
        nonNaNData = filter(!isnan, vcat(values(quadrantResults)...))
        if minimum(nonNaNData) < maximum(nonNaNData)
            colorbarLimits = (minimum(nonNaNData), maximum(nonNaNData))
        else
            colorbarLimits = (minimum(nonNaNData)*(1-1e-5) - 1e-5, minimum(nonNaNData)*(1+1e-5) + 1e-5)
        end
        for W_val in W_val_arr
            push!(saveNames[name], plotHeatmap(abs.(quadrantResults[W_val]), (x_arr[x_arr .≥ 0], x_arr[x_arr .≥ 0]), 
                                               (L"$ak_x/\pi$", L"$ak_y/\pi$"), plotTitles[name], 
                                               getlabelInt(W_val, size_BZ), colmap;
                                               #=colorbarLimits=colorbarLimits,=#
                                              )
                 )
        end
    end

    f = open("plotData.txt", "a")
    for (name, files) in saveNames
        if isempty(files)
            continue
        end
        shellCommand = "pdfunite $(join(files, " ")) $(name)-$(size_BZ).pdf"
        run(`sh -c $(shellCommand)`)
        write(f, shellCommand*"\n")
    end
    close(f)
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
    kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=true)
    node = map2DTo1D(-π/2, -π/2, size_BZ)
    antinode = map2DTo1D(-π, 0., size_BZ)
    spinCorrelation = Dict()
    spinCorrelation["SF1"] = (node, (i, j) -> [("+-+-", [2 * i + 1, 2 * i + 2, 2, 1], 1.)]) # S_d^+c^†_{k↑}_{k↓}c^†_{q↓}_{q↑}S_d^-
    spinCorrelation["SF2"] = (node, (i, j) -> [("+-+-", [2 * j + 1, 2 * j + 2, 2, 1], 1.)]) # S_d^+c^†_{k↑}_{k↓}c^†_{q↓}_{q↑}S_d^-

    saveNames = Dict(name => [] for name in ["tiledSF"])
    plotTitles = Dict("tiledSF" => L"$\chi_s(k_\text{N}, \textbf{k})$",
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
                                           "globalField" => GLOBALFIELD,
                                          )
                             for W_val in W_val_arr
                            )

    effectiveNumShells = Dict(W_val => ifelse(W_val == 0, numShells, 1) for W_val in W_val_arr)
    savePaths = Dict(W_val => joinpath(SAVEDIR, "tiled-spin-corr-$(W_val)-$(effective_Wval)-$(size_BZ)-$(effectiveNumShells[W_val])-$(maxSize)-$(bathIntLegs).jld2") for W_val in W_val_arr)
    for W_val in W_val_arr
        results, _ = AuxiliaryCorrelations(
                                            hamiltDetailsDict[W_val], 
                                            effectiveNumShells[W_val], 
                                            spinCorrelation, 
                                            maxSize,savePaths[W_val]; 
                                            loadData=loadData,
                                            numProcs=nprocs(),
                                           )
        corrResults = Dict("sf" => results["SF1"] .* results["SF2"] )
        tiledCorrelation = sum(values(corrResults)) / length(values(corrResults))
        quadrantResult = tiledCorrelation[filter(p -> all(map1DTo2D(p, size_BZ) .≥ 0), 1:size_BZ^2)]
        push!(saveNames[name], 
              plotHeatmap(quadrantResult .|> abs, 
                          (x_arr[x_arr .≥ 0], x_arr[x_arr .≥ 0]), 
                          (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                          plotTitles[name], 
                          getlabelInt(W_val, size_BZ), 
                          colmap
                         )
             )
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


function TiledEntanglement(
        size_BZ::Int64;
        loadData::Bool=false,
    )
    x_arr = get_x_arr(size_BZ)
    W_val_arr = NiceValues(size_BZ)
    kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=true)

    vnEntropyArr = Float64[]
    saveNames = Dict("S_k"=> String[], 
                     "I2_k_kN"=> String[]
                    )
    node = map2DTo1D(-π/2, -π/2, size_BZ)

    vneDef = Dict{String, Function}(
                                    "SEE_k" => i -> [2 * i + 1,]
                                   )

    mutInfoDef = Dict{String, Tuple{Union{Int64, Nothing}, Function}}(
                      "I2_k_kN" => (node, (i, j) -> ([2 * i + 1,], [2 * j + 1,])),
                     )

    for W_val in W_val_arr
        println(W_val)
        #=println(kondoJArrays[W_val][:, :, end])=#
        hamiltDetailsDict = Dict("dispersion" => dispersion,
                                 "kondoJArray" => kondoJArrays[W_val][:, :, end],
                                 "orbitals" => orbitals,
                                 "size_BZ" => size_BZ,
                                 "bathIntForm" => bathIntForm,
                                 "W_val" => 0.,
                                 "globalField" => 1e1 * GLOBALFIELD,
                                )
        results, _ = AuxiliaryCorrelations(hamiltDetailsDict, numShells, Dict{String, Tuple{Union{Nothing, Int64}, Function}}(),
                                           maxSize, joinpath(SAVEDIR, randstring());
                                           vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef, 
                                           addPerStep=1, loadData=loadData,
                                          )

        quadrantResult = results["SEE_k"][filter(p -> all(map1DTo2D(p, size_BZ) .≥ 0), 1:size_BZ^2)]
        push!(saveNames["S_k"],
              plotHeatmap(quadrantResult,
                          (x_arr[x_arr .≥ 0], x_arr[x_arr .≥ 0]), 
                          (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                          L"$S_\text{EE}(k)$", 
                          getlabelInt(W_val, size_BZ), 
                          colmap
                         )
             )

        quadrantResult = results["I2_k_kN"][filter(p -> all(map1DTo2D(p, size_BZ) .≥ 0), 1:size_BZ^2)]
        push!(saveNames["I2_k_kN"],
              plotHeatmap(quadrantResult,
                          (x_arr[x_arr .≥ 0], x_arr[x_arr .≥ 0]), 
                          (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                          L"$I_2(k, k_N)$", 
                          getlabelInt(W_val, size_BZ), 
                          colmap
                         )
             )
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


size_BZ = 33
#=@time ScattProb(size_BZ; loadData=true)=#
#=@time KondoCouplingMap(size_BZ)=#
#=@time AuxiliaryCorrelations(size_BZ; loadData=false)=#
@time AuxiliaryLocalSpecfunc(size_BZ; loadData=true, fixHeight=true)
#=@time AuxiliaryMomentumSpecfunc(size_BZ, (-π/2, -π/2); loadData=false)=#
#=@time AuxiliaryMomentumSpecfunc(size_BZ, (-3π/4, -π/4); loadData=false)=#
#=@time LatticeKspaceDOS(size_BZ; loadData=true)=#
#=@time TiledSpinCorr(size_BZ; loadData=true)=#
#=@time PhaseDiagram(size_BZ, 1e-3; loadData=false)=#
#=@time TiledEntanglement(size_BZ; loadData=false);=#
