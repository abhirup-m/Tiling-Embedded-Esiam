using Distributed, Random

#=if length(Sys.cpu_info()) > 10 && nprocs() == 1=#
#=    addprocs(2)=#
#=end=#
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
                         25 => -1.0 .* [0, 2., 3.63, 3.7, 3.8, 3.88, 3.9] ./ size_BZ,
                         33 => -1.0 .* [0, 2.8, 5.6, 5.7, 5.89, 5.92, 5.93] ./ size_BZ,
                         41 => -1.0 .* [0, 3.5, 7.13, 7.3, 7.5, 7.564, 7.6] ./ size_BZ,
                         49 => -1.0 .* [0, 4.1, 8.19, 8.4, 8.55, 8.77, 8.8] ./ size_BZ,
                         57 => -1.0 .* [0, 5., 10.2, 10.6, 10.852] ./ size_BZ,
                         69 => -1.0 .* [0, 6., 12.5, 13.1, 13.34, 13.4] ./ size_BZ,
                         77 => -1.0 .* [0., 14.04, 14.5, 14.99] ./ size_BZ,
                        )[size_BZ]
pseudogapEnd(size_BZ) = Dict{Int64, Float64}(
                         13 => -1.0 * 1.6 / size_BZ,
                         25 => -1.0 * 3.88 / size_BZ,
                         33 => -1.0 * 5.92 / size_BZ,
                         41 => -1.0 * 7.564 / size_BZ,
                         49 => -1.0 * 8.77 / size_BZ,
                         57 => -1.0 * 10.852 / size_BZ,
                         69 => -1.0 * 13.34 / size_BZ,
                         77 => -1.0 * 14.99 / size_BZ,
                        )[size_BZ]
pseudogapStart(size_BZ) = Dict{Int64, Float64}(
                         13 => -1.0 * 1.5 / size_BZ,
                         25 => -1.0 * 3.63 / size_BZ,
                         33 => -1.0 * 5.6 / size_BZ,
                         41 => -1.0 * 7.13 / size_BZ,
                         49 => -1.0 * 8.19 / size_BZ,
                         57 => -1.0 * 10.2 / size_BZ,
                         69 => -1.0 * 12.5 / size_BZ,
                         77 => -1.0 * 14.04 / size_BZ,
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
    W_val_arr = NiceValues(size_BZ)[[6]]
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
    W_val_arr = NiceValues(size_BZ)[[1, 3, 4, 5]]
    @time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=true)
    node = map2DTo1D(-π/2, -π/2, size_BZ)
    saveNames = Dict(name => [] for name in ["node", "antinode"])
    titles = Dict(
                  "node" => L"J(k_\mathrm{N}, \vec{k})",
                  "antinode" => L"J(k_\mathrm{AN}, \vec{k})",
                 )
    corrResults = Dict(W_val => Dict() for W_val in W_val_arr)
    bareResults = Dict("node" => Float64[], "antinode" => Float64[])
    for W_val in W_val_arr
        results, bareResults["node"], results_bool = KondoCoupMap((-π/2, -π/2), size_BZ, kondoJArrays[W_val]; 
                                                           #=mapAmong=(kx, ky) -> abs(abs(kx) + abs(ky) - π) < 10^(-0.5),=#
                                                          )
        corrResults[W_val] = Dict("node" => results_bool)
        results, bareResults["antinode"], results_bool = KondoCoupMap((π/1, 0.), size_BZ, kondoJArrays[W_val]; 
                                                           #=mapAmong=(kx, ky) -> abs(abs(kx) + abs(ky) - π) < 10^(-0.5),=#
                                                          )
        corrResults[W_val]["antinode"] = results
    end
    for name in keys(saveNames)
        nonNaNData = vcat([corrResults[W_val][name] for W_val in W_val_arr]..., bareResults[name])
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


function ChannelDecoupling(
        size_BZ::Int64; 
        loadData::Bool=false,
    )
    W_val_arr = range(0.95 * pseudogapStart(size_BZ), pseudogapEnd(size_BZ), length=30) |> collect
    @time kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=loadData)
    pivotPoint = map2DTo1D(-π/2, -π/2, size_BZ)
    decoupledPoints = filter(p -> prod(map1DTo2D(p, size_BZ)) < 0 && abs(dispersion[p]) < 0.14 * maximum(dispersion), 1:size_BZ^2)
    encoupledPoints = filter(p -> prod(map1DTo2D(p, size_BZ)) > 0 && abs(dispersion[p]) < 0.14 * maximum(dispersion), 1:size_BZ^2)
    acrossQuadrantStrengthRatio = Float64[]

    for W_val in W_val_arr
        #=scattProbDecoup = kondoJArrays[W_val][pivotPoint, decoupledPoints, end] .|> abs |> sum=#
        #=scattProbEncoup = kondoJArrays[W_val][pivotPoint, encoupledPoints, end] .|> abs |> sum=#
        scattProbDecoup = maximum(kondoJArrays[W_val][pivotPoint, decoupledPoints, end])
        scattProbEncoup = maximum(kondoJArrays[W_val][pivotPoint, encoupledPoints, end])
        push!(acrossQuadrantStrengthRatio, scattProbDecoup / scattProbEncoup)
    end
    W_val_decouple = W_val_arr[abs.(acrossQuadrantStrengthRatio) .< 1e-2][1]

    plotLines(Tuple{LaTeXString, Vector{Float64}}[("", acrossQuadrantStrengthRatio[1:1:end])],
              -1 .* W_val_arr[1:1:end] ./ J_val,
              L"-W/J", 
              L"J^T_\text{max}/J_\text{max}^N",
              "quadrantMaxKondo-$(size_BZ).pdf";
              scatter=true,
              vlines=[("PG start", - 1 .* pseudogapStart(size_BZ) / J_val), ("PG end", -1 .* pseudogapEnd(size_BZ) / J_val), (L"$W^*$", -1 .* W_val_decouple / J_val)],
              figPad=(0, 10, 0, 2),
              legendPos="lb",
             )
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
            effectiveNumShells = W_val == 0 ? numShells : 1
            effectiveMaxSize = effective_Wval ≠ 0 ? (maxSize > WmaxSize ? WmaxSize : maxSize) : maxSize
            savePaths = corrName -> joinpath(SAVEDIR, "$(W_val)-$(effective_Wval)-$(size_BZ)-$(effectiveNumShells)-$(maxSize)-$(bathIntLegs)-$(corrName).jld2")
            hamiltDetails["W_val"] = effective_Wval
            println("\n W = $(W_val), eff_W=$(effective_Wval), $(effectiveNumShells) shells, maxSize = $(effectiveMaxSize)")
            if effective_Wval == W_val == 0 # case of W = 0, for both spin and charge
                results, _ = AuxiliaryCorrelations(hamiltDetails, effectiveNumShells, merge(spinCorrelation, chargeCorrelation),
                                                effectiveMaxSize; 
                                                savePath=savePaths,
                                                vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef, 
                                                bathIntLegs=bathIntLegs,
                                                noSelfCorr=["cfnode", "cfantinode"], 
                                                addPerStep=1, numProcs=2,
                                                loadData=loadData
                                               )
            elseif effective_Wval == 0 && W_val != 0 # case of W != 0, but setting effective W to 0 for spin
                results, _ = AuxiliaryCorrelations(hamiltDetails, effectiveNumShells, spinCorrelation, effectiveMaxSize;
                                                savePath=savePaths,
                                                vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef, 
                                                bathIntLegs=bathIntLegs, 
                                                addPerStep=1, numProcs=2,
                                                loadData=loadData
                                               )
            else # case of W != 0 and considering the actual W as effective W, for charge
                results, _ = AuxiliaryCorrelations(hamiltDetails, effectiveNumShells, chargeCorrelation, effectiveMaxSize;
                                                savePath=savePaths,
                                                bathIntLegs=bathIntLegs, noSelfCorr=["cfnode", "cfantinode"], 
                                                addPerStep=1, numProcs=2,
                                                loadData=loadData
                                               )
            end
            merge!(corrResults[W_val], results)
        end
    end
    for (name, saveName) in saveNames
        if spinOnly && name ∈ ["doubOcc", "cfnode", "cfantinode"]
            continue
        end
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
    #=W_val_arr = range(0.0, 1.0 * pseudogapEnd(size_BZ), length=10) |> collect=#
    #=W_val_arr = [[0.6 * pseudogapStart(size_BZ)]; range(0.9 * pseudogapStart(size_BZ), pseudogapEnd(size_BZ), length=10) |> collect]=#
    W_val_arr = NiceValues(size_BZ)[[1,6]]
    if fixHeight
        @assert 0 ∈ W_val_arr
    end
    kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=true)
    freqValues = collect(-200:0.005:200)
    specFuncFull = Tuple{LaTeXString, Vector{Float64}}[]
    imagSelfEnergy = Tuple{LaTeXString, Vector{Float64}}[]
    realSelfEnergy = Tuple{LaTeXString, Vector{Float64}}[]
    quasipResidueArr = Float64[]
    standDev = (1.5, exp.(3 .* abs.(freqValues).^0.5 ./ maximum(freqValues)^0.5))
    targetHeight = 0.
    effective_Wval = 0.
    standDevGuess = 0.1
    freqScaleFactor = 8 * HOP_T
    freqValuesZoom1 = 40/freqScaleFactor
    freqValuesZoom2 = 10/freqScaleFactor

    standDevInner = standDev[1]

    nonIntSpecFunc = nothing
    fermiPoints = getIsoEngCont(dispersion, 0.)
    for W_val in W_val_arr
        effectiveNumShells = W_val == 0 ? numShells : 1
        savePath = joinpath(SAVEDIR, "imp-specfunc-$(W_val)-$(effective_Wval)-$(size_BZ)-$(effectiveNumShells)-$(maxSize)-$(bathIntLegs)-$(maximum(freqValues))-$(length(freqValues))-$(GLOBALFIELD).jld2")
        if ispath(savePath) && loadData
            specFunc = jldopen(savePath)["impSpecFunc"]
            quasipResidue = jldopen(savePath)["quasipResidue"]
            centerSpec = jldopen(savePath)["centerSpec"]
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
                                 "kondoJArray_bare" => kondoJArrays[W_val][:, :, 1],
                                 "imp_corr" => 2 * (25 + 10 * abs(W_val)),
                                )
            @time specFunc, standDevGuess, quasipResidue, centerSpecFuncArr = AuxiliaryLocalSpecfunc(hamiltDetails, effectiveNumShells, 
                                                    ImpurityExcitationOperators, freqValues, standDev[1], 
                                                    standDev[2], maxSize; 
                                                    targetHeight=ifelse(abs(W_val) > abs(pseudogapEnd(size_BZ)), 0., targetHeight), 
                                                    heightTolerance=1e-3,
                                                    bathIntLegs=2, addPerStep=1, 
                                                    standDevGuess=standDevInner,
                                                   )
            println("QPR = ", quasipResidue)
            centerSpec = sum(centerSpecFuncArr)
            roundDigits = trunc(Int, log(1/maximum(specFunc)) + 7)
            jldsave(savePath; impSpecFunc=round.(specFunc, digits=roundDigits), quasipResidue=quasipResidue, centerSpec=centerSpec)
        end
        if W_val == 0. && fixHeight
            targetHeight = specFunc[freqValues .≥ 0][1]
        end

        standDev = (standDevInner, standDev[2])

        if isnothing(nonIntSpecFunc)
            nonIntSpecFunc = centerSpec
        end

        selfEnergy = SelfEnergyHelper(specFunc, freqValues, nonIntSpecFunc; normalise=true, pinBottom=abs(W_val) ≤ abs(pseudogapEnd(size_BZ)))

        push!(specFuncFull, (getlabelInt(W_val, size_BZ), specFunc))
        push!(quasipResidueArr, quasipResidue)
        if abs(W_val) > abs(W_val_arr[1])
            push!(realSelfEnergy, (getlabelInt(W_val, size_BZ), real(selfEnergy) ./ freqScaleFactor))
            push!(imagSelfEnergy, (getlabelInt(W_val, size_BZ), -1 .* imag(selfEnergy) ./ freqScaleFactor))
        end

    end
    println(quasipResidueArr)
    plotLines(specFuncFull, 
              freqValues ./ freqScaleFactor,
              L"\omega", 
              L"A(\omega)",
              "impSpecFunc_$(size_BZ)-$(maxSize).pdf";
              xlimits=(-freqValuesZoom1, freqValuesZoom1),
              linewidth=1.5,
              figPad=5,
             )
    plotLines(specFuncFull, 
              freqValues ./ freqScaleFactor,
              L"\omega", 
              L"A(\omega)",
              "impSpecFuncTrunc_$(size_BZ)-$(maxSize).pdf";
              xlimits=(-freqValuesZoom2, freqValuesZoom2),
             )
    plotLines(realSelfEnergy, 
              freqValues ./ freqScaleFactor,
              L"\omega", 
              L"-\Sigma^\prime(\omega) / D",
              "sigmaReal_$(size_BZ)-$(maxSize).pdf";
              xlimits=(-freqValuesZoom1, freqValuesZoom1),
             )
    plotLines(imagSelfEnergy,
              freqValues ./ freqScaleFactor,
              L"\omega", 
              L"-\Sigma^{\prime\prime}(\omega) / D",
              "sigmaImag_$(size_BZ)-$(maxSize).pdf";
              ylimits=(-0.1, 1e1),
              xlimits=(-1 * freqValuesZoom1, 1 * freqValuesZoom1),
              linewidth=1.5,
              figPad=10.,
             )
    plotLines(imagSelfEnergy,
              freqValues ./ freqScaleFactor,
              L"\omega", 
              L"-\Sigma^{\prime\prime}(\omega)/D",
              "sigmaImag-trunc_$(size_BZ)-$(maxSize).pdf";
              xlimits=(-freqValuesZoom2, freqValuesZoom2),
              ylimits=(0., 5/8),
              linewidth=1.5,
              figPad=5,
             )
    plotLines(Tuple{LaTeXString, Vector{Float64}}[("", quasipResidueArr)], 
              -1 .* W_val_arr / J_val,
              L"-W/J", 
              L"Z_\text{imp}",
              "localQPResidue_$(size_BZ)-$(maxSize).pdf";
              scatter=true,
              vlines=Tuple{AbstractString, Float64}[("", - 1 .* pseudogapStart(size_BZ) / J_val), ("", -1 .* pseudogapEnd(size_BZ) / J_val)],
              yscale=log10,
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
        singleThread::Bool=false,
    )
    x_arr = get_x_arr(size_BZ)
    W_val_arr = NiceValues(size_BZ)
    #=W_val_arr = NiceValues(size_BZ)[[1, 2]]=#
    kondoJArrays, dispersion = RGFlow(W_val_arr, size_BZ; loadData=true)
    freqValues = collect(-200:0.1:200)
    freqValuesZoom1 = 6.
    freqValuesZoom2 = 0.3
    standDev = (1., exp.(abs.(freqValues) ./ maximum(freqValues)))
    effective_Wval = 0.
    targetHeight = 0.0
    standDevGuess = standDev[1]
    saveNames = Dict(name => [] for name in ["kspaceDOS", "quasipRes", "selfEnergyKspace"])
    quadrantResults = Dict{String, Dict{Float64, Vector{Float64}}}(name => Dict{Float64, Vector{Float64}}() 
                                                                   for name in keys(saveNames)
                                                          )
    plotTitles = Dict("kspaceDOS" => L"$A_{\vec{k}}(\omega\to 0)$",
                      "quasipRes" => L"$Z_{\vec{k}}$",
                      "selfEnergyKspace" => L"$-\Sigma^{\prime\prime}\left(\vec{k}, \omega \to 0\right)$",
                     )
    nonIntSpecBzone = nothing

    for W_val in W_val_arr
        savePath = joinpath(SAVEDIR, "kspaceDOS-$(W_val)-$(effective_Wval)-$(size_BZ)-$(numShells)-$(maxSize)-$(bathIntLegs)-$(maximum(freqValues))-$(length(freqValues))-$(GLOBALFIELD).jld2")
        hamiltDetails = Dict(
                             "dispersion" => dispersion,
                             "kondoJArray" => kondoJArrays[W_val][:, :, end],
                             "orbitals" => orbitals,
                             "size_BZ" => size_BZ,
                             "bathIntForm" => bathIntForm,
                             "W_val" => effective_Wval,
                             "globalField" => -GLOBALFIELD,
                             "kondoJArray_bare" => kondoJArrays[W_val][:, :, 1],
                             "imp_corr" => 2 * (25 + 10 * abs(W_val)),
                            )
        specFuncKSpace, specFunc, standDevGuess, results, nonIntSpecBzone = LatticeKspaceDOS(hamiltDetails, numShells, 
                                       KspaceExcitationOperators, freqValues, standDev[1], 
                                       standDev[2], maxSize, savePath; loadData=loadData, bathIntLegs=bathIntLegs,
                                       addPerStep=1, targetHeight=ifelse(abs(W_val) > abs(pseudogapEnd(size_BZ)), 0., targetHeight),
                                       standDevGuess=standDevGuess, nonIntSpecBzone=nonIntSpecBzone,
                                       selfEnergyWindow=0.19,
                                       singleThread=false,
                                      )
        for (name, val) in results
            quadrantResults[name][W_val] = results[name][filter(p -> all(map1DTo2D(p, size_BZ) .≤ 0), 1:size_BZ^2)]
        end
        targetHeight = specFunc[freqValues .≥ 0][1]
    end

    for name in keys(saveNames)
        colorbarLimits = ColorbarLimits(quadrantResults[name])
        for W_val in W_val_arr
            push!(saveNames[name], plotHeatmap(quadrantResults[name][W_val], (x_arr[x_arr .≥ 0], x_arr[x_arr .≥ 0]), 
                                               (L"$ak_x/\pi$", L"$ak_y/\pi$"), plotTitles[name], 
                                               getlabelInt(W_val, size_BZ), colmap;
                                               colorbarLimits=colorbarLimits,
                                               colorScale=ifelse(name == "selfEnergyKspace", log10, identity),
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
    savePaths = Dict(W_val => corrName -> joinpath(SAVEDIR, "$(W_val)-$(effective_Wval)-$(size_BZ)-$(effectiveNumShells[W_val])-$(maxSize)-$(bathIntLegs)-$(corrName).jld2") for W_val in W_val_arr)
    for W_val in W_val_arr
        results, _ = AuxiliaryCorrelations(
                                            hamiltDetailsDict[W_val], 
                                            effectiveNumShells[W_val], 
                                            spinCorrelation, 
                                            maxSize;
                                            savePath=savePaths[W_val],
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

    vnEntropy = Dict{Float64, Vector{Float64}}()
    mutInfoNode = Dict{Float64, Vector{Float64}}()
    mutInfoEdge = Dict{Float64, Vector{Float64}}()

    pivotPoints = Dict{Float64, Int64}()
    saveNames = Dict("S_k"=> String[], 
                     "I2_k_kN"=> String[],
                     "I2_k_edge"=> String[],
                    )
    node = map2DTo1D(-π/2, -π/2, size_BZ)

    vneDef = Dict{String, Function}(
                                    "SEE_k" => i -> [2 * i + 1,]
                                   )

    cutoffEnergy = dispersion[div(size_BZ - 1, 2) + 2 - numShells]
    SWIndices = [p for p in 1:size_BZ^2 if map1DTo2D(p, size_BZ)[1] < 0
                 && map1DTo2D(p, size_BZ)[2] ≤ 0 
                 && abs(cutoffEnergy) ≥ abs(dispersion[p])
                ]
    savePaths = Dict(W_val => corrName -> joinpath(SAVEDIR, "$(W_val)-0.0-$(size_BZ)-$(1)-$(maxSize)-$(bathIntLegs)-$(corrName).jld2") for W_val in W_val_arr)

    @time Threads.@threads for W_val in W_val_arr
        scattProbBool = ScattProb(size_BZ, kondoJArrays[W_val], dispersion)[2]
        connectedPoints = filter(p -> scattProbBool[p] > 0, SWIndices)
        if !isempty(connectedPoints)
            pivot = sort(connectedPoints)[1]
        else
            pivot = sort(SWIndices)[end]
        end
        pivotPoints[W_val] = pivot

        mutInfoDef = Dict{String, Tuple{Union{Int64, Nothing}, Function}}(
                          "I2_k_edge" => (pivot, (i, j) -> ([2 * i + 1,], [2 * j + 1,])),
                          "I2_k_kN" => (node, (i, j) -> ([2 * i + 1,], [2 * j + 1,])),
                         )
        hamiltDetailsDict = Dict("dispersion" => dispersion,
                                 "kondoJArray" => kondoJArrays[W_val][:, :, end],
                                 "orbitals" => orbitals,
                                 "size_BZ" => size_BZ,
                                 "bathIntForm" => bathIntForm,
                                 "W_val" => 0.,
                                 "globalField" => 1e1 * GLOBALFIELD,
                                )
        results, _ = AuxiliaryCorrelations(hamiltDetailsDict, numShells, Dict{String, Tuple{Union{Nothing, Int64}, Function}}(),
                                           maxSize; savePath=savePaths[W_val],
                                           vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef, 
                                           addPerStep=1, loadData=loadData,
                                          )
        vnEntropy[W_val] = results["SEE_k"]
        mutInfoNode[W_val] = results["I2_k_kN"]
        mutInfoEdge[W_val] = results["I2_k_edge"]
    end
    for W_val in W_val_arr
        quadrantResult = vnEntropy[W_val][filter(p -> all(map1DTo2D(p, size_BZ) .≤ 0), 1:size_BZ^2)]
        push!(saveNames["S_k"],
              plotHeatmap(quadrantResult,
                          (x_arr[x_arr .≤ 0], x_arr[x_arr .≤ 0]), 
                          (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                          L"$S_\text{EE}(k)$", 
                          getlabelInt(W_val, size_BZ), 
                          colmap
                         )
             )

        quadrantResult = mutInfoNode[W_val][filter(p -> all(map1DTo2D(p, size_BZ) .≤ 0), 1:size_BZ^2)]
        push!(saveNames["I2_k_kN"],
              plotHeatmap(quadrantResult,
                          (x_arr[x_arr .≤ 0], x_arr[x_arr .≤ 0]), 
                          (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                          L"$I_2(k, k_N)$", 
                          getlabelInt(W_val, size_BZ), 
                          colmap;
                          marker=map1DTo2D(node, size_BZ) ./ π,
                         )
             )

        quadrantResult = mutInfoEdge[W_val][filter(p -> all(map1DTo2D(p, size_BZ) .≤ 0), 1:size_BZ^2)]
        push!(saveNames["I2_k_edge"],
              plotHeatmap(quadrantResult,
                          (x_arr[x_arr .≤ 0], x_arr[x_arr .≤ 0]), 
                          (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                          L"$I_2(k, k_\text{edge})$", 
                          getlabelInt(W_val, size_BZ), 
                          colmap;
                          marker=map1DTo2D(pivotPoints[W_val], size_BZ) ./ π,
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

size_BZ = 77
#=@time ChannelDecoupling(size_BZ; loadData=true)=#
#=@time ScattProb(size_BZ; loadData=true)=#
#=@time KondoCouplingMap(size_BZ)=#
#=@time AuxiliaryCorrelations(size_BZ; loadData=true, spinOnly=true)=#
#=@time AuxiliaryLocalSpecfunc(size_BZ; loadData=false, fixHeight=false)=#
#=@time AuxiliaryMomentumSpecfunc(size_BZ, (-π/2, -π/2); loadData=false)=#
#=@time AuxiliaryMomentumSpecfunc(size_BZ, (-3π/4, -π/4); loadData=false)=#
@time LatticeKspaceDOS(size_BZ; loadData=false, singleThread=true)
#=@time TiledSpinCorr(size_BZ; loadData=true)=#
#=@time PhaseDiagram(size_BZ, 1e-3; loadData=false)=#
#=@time TiledEntanglement(size_BZ; loadData=true);=#
