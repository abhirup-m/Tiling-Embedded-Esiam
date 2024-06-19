using JLD2
using ProgressMeter

const TRUNC_DIM = 2
include("./source/constants.jl")
include("./source/rgFlow.jl")
include("./source/probes.jl")
include("./source/plotting.jl")
const J_val = 0.1
const BZfraction = 0.5
const size_BZ = 37
const omega_by_t = -2.0
const W_by_J_arr = -1.0 .* [59, 60, 61, 62, 63, 64] ./ size_BZ # time = 14.318 s
const orbitals = ("p", "p")
const savePaths = rgFlowData(size_BZ, omega_by_t, J_val, W_by_J_arr, orbitals)
const x_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi

function probe()
    collatedResults = []
    for (i, savePath) in enumerate(savePaths)
        jldopen(savePath, "r"; compress=true) do file
        kondoJArray = file["kondoJArray"]
        dispersion = file["dispersion"]
        W_val = file["W_val"]
        size_BZ = file["size_BZ"]
        orbitals = file["orbitals"]
        dispersion = file["dispersion"]

        results_arr = scattProb(kondoJArray, size_BZ, dispersion)
        push!(collatedResults, [log10.(results_arr[1]), results_arr[2]])
        end
    end
    saveName = "scattprob-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4)).pdf"
    plotHeatmaps([x_arr, x_arr], [L"$ak_x/\pi$", L"$ak_y/\pi$"], [L"$\Gamma/\Gamma_0$", L"relevance of $\Gamma$"], collatedResults, saveName)
end

function corr()
    collatedResultsSpin = []
    collatedResultsVne = []
    collatedResultsCharge = []
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
                           
    corrCharge = pair -> Dict(("++--", [2 * pair[2] + 1, 2 * pair[2] + 2,  2 * pair[1] + 2, 2 * pair[1] + 1]) => 1.0, 
                                    ("--++", [2 * pair[1] + 2, 2 * pair[1] + 1,  2 * pair[2] + 1, 2 * pair[2] + 2]) => 1.0,
                                    ("++--", [2 * pair[1] + 1, 2 * pair[1] + 2,  2 * pair[2] + 2, 2 * pair[2] + 1]) => 1.0,
                                    ("--++", [2 * pair[2] + 2, 2 * pair[2] + 1,  2 * pair[1] + 1, 2 * pair[1] + 2]) => 1.0
                                   )
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

        # set W_val to zero so that it does not interfere with the spin-flip fluctuations.
        basis, suitableIndices, uniqueSequences, gstatesSet = getBlockSpectrum(size_BZ, dispersion, kondoJArray, W_val, orbitals, BZfraction)
        resSpin, resSpinBool = correlationMap(size_BZ, basis, dispersion, suitableIndices, uniqueSequences, gstatesSet, corrSpinFlip)
        push!(collatedResultsSpin, [log10.(resSpin), resSpinBool])

        vne, vneBool = entanglementMap(size_BZ, basis, dispersion, suitableIndices, uniqueSequences, gstatesSet)
        push!(collatedResultsVne, [vne, vneBool])

        basis, suitableIndices, uniqueSequences, gstatesSet = getBlockSpectrum(size_BZ, dispersion, kondoJArray, W_val, orbitals, BZfraction)
        resDoubOcc, resDoubOccBool = correlationMap(size_BZ, basis, dispersion, suitableIndices, uniqueSequences, gstatesSet, corrDoubOcc)
        resCharge, resChargeBool = correlationMap(size_BZ, basis, dispersion, suitableIndices, uniqueSequences, gstatesSet, corrCharge; twoParticle=1)
        push!(collatedResultsCharge, [resCharge, resDoubOcc])
        end
    end

    saveName = "spin-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4)).pdf"
    plotHeatmaps([x_arr, x_arr], [L"$ak_x/\pi$", L"$ak_y/\pi$"], [L"$\chi_s(d, \vec k)$", L"relevance of $\chi_s(d, \vec k)$"], collatedResultsSpin, saveName)
    saveName = "charge-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4)).pdf"
    plotHeatmaps([x_arr, x_arr], [L"$ak_x/\pi$", L"$ak_y/\pi$"], ["off-diag ch. corr.", L"$n_{k\uparrow}n_{k\downarrow}$"], collatedResultsCharge, saveName)
    saveName = "vne-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4)).pdf"
    plotHeatmaps([x_arr, x_arr], [L"$ak_x/\pi$", L"$ak_y/\pi$"], ["off-diag ch. corr.", L"$n_{k\uparrow}n_{k\downarrow}$"], collatedResultsVne, saveName)
end

probe()
corr()
