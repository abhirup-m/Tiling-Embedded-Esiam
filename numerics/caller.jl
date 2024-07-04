using JLD2
using ProgressMeter

const TRUNC_DIM = 2
include("./source/constants.jl")
include("./source/helpers.jl")
include("./source/rgFlow.jl")
include("./source/probes.jl")
include("./source/plotting.jl")
const J_val = 0.1
const fractionBZ = 0.01
const size_BZ = 17
const omega_by_t = -2.0
const W_by_J_arr = -1.0 .* [0, 24, 25, 26.5, 27] ./ size_BZ
const orbitals = ("p", "p")
const savePaths = rgFlowData(size_BZ, omega_by_t, J_val, W_by_J_arr, orbitals)
const x_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
const cmap = :YlOrBr

function probe()
    collatedResults = []
    subFigTitles = []
    for (i, savePath) in enumerate(savePaths)
        jldopen(savePath, "r"; compress=true) do file
        kondoJArray = file["kondoJArray"]
        dispersion = file["dispersion"]
        W_val = file["W_val"]
        size_BZ = file["size_BZ"]
        orbitals = file["orbitals"]
        dispersion = file["dispersion"]
        push!(subFigTitles, L"W/J=%$(round(W_val, digits=3))")

        results_arr = scattProb(kondoJArray, size_BZ, dispersion, fractionBZ)
        push!(collatedResults, [results_arr[1], results_arr[2]])
        end
    end
    saveName = "scattprob-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4)).pdf"
    plotHeatmaps([x_arr, x_arr], [L"$ak_x/\pi$", L"$ak_y/\pi$"], [L"$\Gamma/\Gamma_0$", L"relevance of $\Gamma$"], subFigTitles, collatedResults, saveName)
end

function corr()

    collatedResultsSpin = []
    collatedResultsVne = []
    collatedResultsCharge = []
    collatedResultsChargeSpecific = []
    corrSpinFlip = i -> Dict(("+-+-", [2, 1, 2 * i + 1, 2 * i + 2]) => 1.0, ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1]) => 1.0)
    corrDoubOcc = i -> Dict(("nn", [2 * i + 1, 2 * i + 2]) => 1.0, ("hh", [2 * i + 1, 2 * i + 2]) => 1.0)
    corrimpKDoubOcc = i -> Dict(("nn", [1, 2 * i + 1]) => 1.0, 
                                      ("nn", [2, 2 * i + 1]) => 1.0, 
                                      ("nn", [1, 2 * i + 2]) => 1.0, 
                                      ("nn", [2, 2 * i + 2]) => 1.0, 
                                      ("hh", [1, 2 * i + 1]) => 1.0, 
                                      ("hh", [2, 2 * i + 1]) => 1.0, 
                                      ("hh", [1, 2 * i + 2]) => 1.0, 
                                      ("hh", [2, 2 * i + 2]) => 1.0,
                                     )
                           
    corrCharge = (x1, x2) -> Dict(("++--", [2 * x2 + 1, 2 * x2 + 2,  2 * x1 + 2, 2 * x1 + 1]) => 1.0, 
                                    ("--++", [2 * x1 + 2, 2 * x1 + 1,  2 * x2 + 1, 2 * x2 + 2]) => 1.0,
                                    ("++--", [2 * x1 + 1, 2 * x1 + 2,  2 * x2 + 2, 2 * x2 + 1]) => 1.0,
                                    ("--++", [2 * x2 + 2, 2 * x2 + 1,  2 * x1 + 1, 2 * x1 + 2]) => 1.0
                                   )
    subFigTitles = []
    @showprogress for (i, savePath) in collect(enumerate(savePaths))
        jldopen(savePath, "r"; compress=true) do file
        kondoJArray = file["kondoJArray"]
        dispersion = file["dispersion"]
        W_val = file["W_val"]
        size_BZ = file["size_BZ"]
        orbitals = file["orbitals"]
        dispersion = file["dispersion"]
        node = map2DTo1D(float(π)/2, float(π)/2, size_BZ)
        antinode = map2DTo1D(float(π), 0.0, size_BZ)
        genpoint = map2DTo1D(float(3 * π / 4), 0.0, size_BZ)

        push!(subFigTitles, L"W/J=%$(round(W_val, digits=3))")

        # set W_val to zero so that it does not interfere with the spin-flip fluctuations.
        @time spectrumFamily, numStatesFamily, activeStatesArr = getIterativeSpectrum(size_BZ, dispersion, kondoJArray, 0.0, orbitals, fractionBZ)
        groundstates = groundStates(spectrumFamily, numStatesFamily)
        resSpin, resSpinBool = correlationMap(groundstates, (corrSpinFlip, 1), Dict(activeStatesArr[end] .=> 1:length(activeStatesArr[end])))

        @time spectrumFamily, numStatesFamily, activeStatesArr = getIterativeSpectrum(size_BZ, dispersion, kondoJArray, W_val, orbitals, fractionBZ)
        groundstates = groundStates(spectrumFamily, numStatesFamily)
        # vne, mutInfo = entanglementMap(size_BZ, basis, dispersion, suitableIndices, uniqueSequences, gstatesSet, [node, antinode])
        # push!(collatedResultsVne, [vne, mutInfo[node], mutInfo[antinode]])
        resDoubleOcc, resDoubleOccBool = correlationMap(groundstates, (corrDoubOcc, 1), Dict(activeStatesArr[end] .=> 1:length(activeStatesArr[end])))
        # resCharge, resChargeBool = correlationMap(groundstates, (corrCharge, 2), Dict(activeStatesArr[end] .=> 1:length(activeStatesArr[end])))
        # resChargeN, _ = correlationMap(size_BZ, basis, dispersion, suitableIndices, uniqueSequences, gstatesSet, corrCharge; twoParticle=1, pivotIndex=node)
        # resChargeAN, _ = correlationMap(size_BZ, basis, dispersion, suitableIndices, uniqueSequences, gstatesSet, corrCharge; twoParticle=1, pivotIndex=antinode)
        push!(collatedResultsCharge, [resSpin, resDoubleOcc])
        # push!(collatedResultsChargeSpecific, [resChargeN, resChargeAN])
        end
    end

    saveName = "charge-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4)).pdf"
    plotHeatmaps([x_arr, x_arr], [L"ak_x/\pi", L"ak_y/\pi"], [L"\chi_s(d, \vec{k})", L"\chi_c(\vec{k})", L"n_{k\uparrow}n_{k\downarrow}"], subFigTitles, collatedResultsCharge, saveName)
    # saveName = "vne-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4)).pdf"
    # plotHeatmaps([x_arr, x_arr], [L"ak_x/\pi", L"ak_y/\pi"], [L"\text{S}_\text{EE}(k)", L"I_2(k_N:k)", L"I_2(k_\text{AN}:k)"], subFigTitles, collatedResultsVne, saveName)
    # saveName = "chargeSpec-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4)).pdf"
    # plotHeatmaps([x_arr, x_arr], [L"ak_x/\pi", L"ak_y/\pi"], [L"\chi_c(k_\text{N}, k)", L"\chi_c(k_\text{AN}, k)"], subFigTitles, collatedResultsChargeSpecific, saveName)
end

probe();
corr();
