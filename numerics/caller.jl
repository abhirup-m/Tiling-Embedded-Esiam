using Distributed

if length(Sys.cpu_info()) > 10 && nprocs() == 1
    addprocs(0)
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
global const maxSize = 800
WmaxSize = 800

colmap = reverse(ColorSchemes.cherry)
numShells = 1
size_BZ = 57
bathIntLegs = 2
W_val_arr = -1.0 .* [10.86, 10.88] ./ size_BZ
#=W_val_arr = -1.0 .* [0, 4.1, 8.19, 8.55, 8.77] ./ size_BZ=#
# W_val_arr = -1.0 .* [0, 3.5, 7.13, 7.3, 7.5, 7.564, 7.6] ./ size_BZ
# W_val_arr = -1.0 .* [0, 2.8, 5.6, 5.7, 5.82, 5.89, 5.92] ./ size_BZ

x_arr = collect(range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi)
getlabelInt(W_val) = L"$W/J=%$(round(W_val/J_val, digits=2))$"
getlabelSize(W_val) = L"$\frac{W}{J}=%$(round(W_val/J_val, digits=2))$\n$L=%$(size_BZ)$"

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

        quadrantResult = results_bool[filter(p -> all(map1DTo2D(p, size_BZ) .≥ 0), 1:size_BZ^2)]
        push!(saveNames, plotHeatmap(abs.(quadrantResult), (x_arr[x_arr .≥ 0], x_arr[x_arr .≥ 0]), (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                                     L"$\Gamma/\Gamma_0$", getlabelSize(W_val), reverse(colmap)))
    end
    println("\n Saved at $(saveNames).")
    run(`pdfunite $(saveNames) scattprob.pdf`)
end


function corr(kondoJArrays, dispersion)

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
    saveNamesPolished = Dict(name => [] for name in ["SF", "vne_k", "I2_k_d", "I2_k_N", "I2_k_AN", "doubOcc", "cfnode", "cfantinode"])
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
                                                effectiveMaxSize; vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef, bathIntLegs=bathIntLegs,
                                                noSelfCorr=["cfnode", "cfantinode"], addPerStep=2)
            elseif effective_Wval == 0 && W_val != 0 # case of W != 0, but setting effective W to 0 for spin
                corrResults, _ = correlationMap(hamiltDetails, effectiveNumShells, spinCorrelation, effectiveMaxSize;
                                                vneFuncDict=vneDef, mutInfoFuncDict=mutInfoDef, bathIntLegs=bathIntLegs, addPerStep=2)
            else # case of W != 0 and considering the actual W as effective W, for charge
                corrResults, _ = correlationMap(hamiltDetails, effectiveNumShells, chargeCorrelation, effectiveMaxSize;
                                                bathIntLegs=bathIntLegs, noSelfCorr=["cfnode", "cfantinode"], addPerStep=2)
            end

            for name in keys(corrResults)
                push!(saveNames[name], plotHeatmap(abs.(corrResults[name]), (x_arr, x_arr), (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                                                   plotTitles[name], getlabelSize(W_val), colmap))
            end
            for name in keys(corrResults)
                quadrantResult = corrResults[name][filter(p -> all(map1DTo2D(p, size_BZ) .≥ 0), 1:size_BZ^2)]
                push!(saveNamesPolished[name], plotHeatmap(abs.(quadrantResult), (x_arr[x_arr .≥ 0], x_arr[x_arr .≥ 0]), (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                                                   plotTitles[name], getlabelSize(W_val), colmap))
            end
        end
    end
    f = open("saveData.txt", "a")
    for (name, files) in saveNamesPolished
        if isempty(files)
            continue
        end
        shellCommand = "pdfunite $(join(files, " ")) $(name).pdf"
        run(`sh -c $(shellCommand)`)
        write(f, shellCommand*"\n")
    end
    close(f)
end


function LocalSpecFunc(kondoJArrays, dispersion)

    function specFuncDictFunc(
            numBathPoints::Int64,
        )
        #=if bathIndices == [1]=#
        #=    return Dict("create" => [("+", [1], 1.), ("+", [2], 1.)], "destroy" => [("-", [1], 1.), ("-", [2], 1.)])=#
        #=else=#
        #=    dict = Dict{String, Vector{Tuple{String,Vector{Int64}, Float64}}}("create" => [], "destroy" => [])=#
        #=    for index in bathIndices[1:1]=#
        #=        append!(dict["create"], [("+-+", [2, 1, 2 * index - 1], 1.), ("+-+", [1, 2, 2 * index], 1.), ])=#
        #=        append!(dict["destroy"], [("+--", [1, 2, 2 * index - 1], 1.), ("+--", [2, 1, 2 * index], 1.), ])=#
        #=    end=#
        #=    return dict=#
        #=end=#

        siamSpecDict = Dict{String, Vector{Tuple{String,Vector{Int64}, Float64}}}("create" => [], "destroy" => [])
        append!(siamSpecDict["create"], [("+", [1], 1.), ("+", [2], 1.)])
        append!(siamSpecDict["destroy"], [("-", [1], 1.), ("-", [2], 1.)])
        kondoSpecDict = Dict{String, Vector{Tuple{String,Vector{Int64}, Float64}}}("create" => [], "destroy" => [])
        for index in 1:numBathPoints
            append!(kondoSpecDict["create"], [("+-+", [2, 1, 2 * index + 1], 1.), ("+-+", [1, 2, 2 * index + 2], 1.),])
            append!(kondoSpecDict["destroy"], [("+--", [1, 2, 2 * index + 1], 1.), ("+--", [2, 1, 2 * index + 2], 1.),])
        end
        return siamSpecDict, kondoSpecDict
    end
    freqValues = collect(-11:0.005:11)
    specFuncResults = Tuple{LaTeXString, Vector{Float64}}[]
    specFuncResultsTrunc = Tuple{LaTeXString, Vector{Float64}}[]
    standDev = (0.1, 0.1 .+ exp.(abs.(freqValues) ./ maximum(freqValues))) # 0.1 .+ 0 .* abs.(freqValues) / maximum(freqValues)

    for W_val in W_val_arr

        hamiltDetails = Dict(
                             "dispersion" => dispersion,
                             "kondoJArray" => kondoJArrays[W_val][:, :, end],
                             "orbitals" => orbitals,
                             "size_BZ" => size_BZ,
                             "bathIntForm" => bathIntForm,
                             "W_val" => 0. * W_val,
                             "globalField" => 1e-5,
                            )
        effectiveNumShells = W_val == 0 ? numShells : 1
        effectiveMaxSize = W_val ≠ 0 ? (maxSize > WmaxSize ? WmaxSize : maxSize) : maxSize
        specFunc, standDevInner = localSpecFunc(hamiltDetails, effectiveNumShells, 
                                                specFuncDictFunc, freqValues, standDev[1], standDev[2],
                                                effectiveMaxSize; resonanceHeight=0.42,
                                                heightTolerance=1e-3, bathIntLegs=bathIntLegs,
                                                addPerStep=1)
        standDev = (standDevInner, standDev[2])
        push!(specFuncResults, (getlabelInt(W_val), specFunc))
        push!(specFuncResultsTrunc, (getlabelInt(W_val), specFunc[abs.(freqValues) .≤ 1]))
    end
    plotSpecFunc(specFuncResults, freqValues, "impSpecFunc_$(size_BZ).pdf")
    plotSpecFunc(specFuncResultsTrunc, freqValues[abs.(freqValues) .≤ 1], "impSpecFuncTrunc_$(size_BZ).pdf")
end


function PhaseDiagram()
    J_val_arr = collect(0.1:0.003:0.3)
    W_val_arr = collect(0:-0.003:-0.3)
    phaseLabels = ["L-FL", "L-PG", "LM"]
    @time phaseDiagram = PhaseDiagram(J_val_arr, W_val_arr, Dict(phaseLabels .=> 1:3))
    plotPhaseDiagram(phaseDiagram, Dict(1:3 .=> phaseLabels), (J_val_arr, -1 .* W_val_arr), 
                     (L"J/t", L"-W/t"), L"L=%$(size_BZ)", "phaseDiagram.pdf",  
                     colmap[[2, 5, 8]], (350, 300))
end


function Correlations2Point(kondoJArrays, dispersion)

    node = map2DTo1D(-π/2, -π/2, size_BZ)
    inter = map2DTo1D(-3π/4, -π/4, size_BZ)
    antinode = map2DTo1D(-π, 0., size_BZ)
    probePoints = [node, inter, antinode]

    spinCorrelation = Dict("SF1" => [-1, (i, j) -> [("+-+-", [2 * i + 1, 2, 2, 2 * j + 1], 1.)]]) # c^†_{k↑}c_{d↓}c^†_{d↓}c_{q↑}
    spinCorrelation["SF4"] = [-1, (i, j) -> [("+-+-", [2 * i + 1, 2, 2 * j + 2, 1], 1.)]]         # c^†_{k↑}c_{d↓}c^†_{q↓}c_{d↑}
    spinCorrelation["SF7"] = [-1, (i, j) -> [("+-+-", [2 * i + 1, 2, 2 * j + 2, 2 * j + 1], 1.)]] # c^†_{k↑}c_{d↓}c^†_{q↓}c_{q↑}
    spinCorrelation["SF5"] = [-1, (i, j) -> [("+-+-", [2 * i + 1, 2 * i + 2, 2 * j + 2, 1], 1.)]] # c^†_{k↑}c_{k↓}c^†_{q↓}c_{d↑}
    spinCorrelation["SF6"] = [-1, (i, j) -> [("+-+-", [2 * i + 1, 2 * i + 2, 2, 2 * j + 1], 1.)]] # c^†_{k↑}c_{k↓}c^†_{d↓}c_{q↑}
    spinCorrelation["SF2"] = [-1, (i, j) -> [("+-+-", [1, 2 * i + 2, 2 * j + 2, 1], 1.)]]         # c^†_{d↑}c_{k↓}c^†_{q↓}c_{d↑}
    spinCorrelation["SF3"] = [-1, (i, j) -> [("+-+-", [1, 2 * i + 2, 2, 2 * j + 1], 1.)]]         # c^†_{d↑}c_{k↓}c^†_{d↓}c_{q↑}
    spinCorrelation["SF8"] = [-1, (i, j) -> [("+-+-", [1, 2 * i + 2, 2 * j + 2, 2 * j + 1], 1.)]] # c^†_{d↑}c_{k↓}c^†_{q↓}c_{q↑}

    saveNamesPolished = Dict(name => [] for name in ["SF-N", "SF-IN", "SF-AN"])
    plotTitles = Dict(
                      "SF-N" => L"$\chi_s(k_\mathrm{N}, \vec{k})$",
                      "SF-IN" => L"$\chi_s(k_\mathrm{MID}, \vec{k})$",
                      "SF-AN" => L"$\chi_s(k_\mathrm{AN}, \vec{k})$",
                     )

    for W_val in W_val_arr

        hamiltDetails = Dict(
                             "dispersion" => dispersion,
                             "kondoJArray" => kondoJArrays[W_val][:, :, end],
                             "orbitals" => orbitals,
                             "size_BZ" => size_BZ,
                             "bathIntForm" => bathIntForm,
                            )

        hamiltDetails["W_val"] = 0.
        effectiveNumShells = 1
        effectiveMaxSize = maxSize
        results, _ = correlationMap2Point(hamiltDetails, effectiveNumShells, 
                                          spinCorrelation, effectiveMaxSize;
                                          probePoints=probePoints,
                                         )
        corrResults = Dict()
        corrResults["SF-N"] = sum([v[1, :] for v in values(results)]) ./ 6
        corrResults["SF-IN"] = sum([v[2, :] for v in values(results)]) ./ 6
        corrResults["SF-AN"] = sum([v[3, :] for v in values(results)]) ./ 6

        for name in keys(corrResults)
            quadrantResult = corrResults[name][filter(p -> all(map1DTo2D(p, size_BZ) .≥ 0), 1:size_BZ^2)]
            push!(saveNamesPolished[name], plotHeatmap(abs.(quadrantResult), (x_arr[x_arr .≥ 0], x_arr[x_arr .≥ 0]), (L"$ak_x/\pi$", L"$ak_y/\pi$"),
                                               plotTitles[name], getlabelSize(W_val), colmap))
        end
    end
    f = open("saveData.txt", "a")
    for (name, files) in saveNamesPolished
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
#=@time corr(kondoJArrays, dispersion)=#
#=@time LocalSpecFunc(kondoJArrays, dispersion)=#
#=@time Correlations2Point(kondoJArrays, dispersion)=#
#=@time PhaseDiagram()=#
