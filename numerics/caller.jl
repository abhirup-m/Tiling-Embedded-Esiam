using JLD2
using ProgressMeter

const TRUNC_DIM = 2
include("./source/constants.jl")
include("./source/helpers.jl")
include("./source/rgFlow.jl")
include("./source/probes.jl")
include("./source/plotting.jl")
const J_val = 0.1
const omega_by_t = -2.0
const orbitals = ("p", "p")
const cmap = :cherry
fractionBZ = 0.1
size_BZ = 9
W_val_arr = -1.0 .* [0, 1] ./ size_BZ
x_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi

node = map2DTo1D(float(π)/2, float(π)/2, size_BZ)
antinode = map2DTo1D(float(π), 0.0, size_BZ)
genpoint = map2DTo1D(float(3 * π / 4), 0.0, size_BZ)

densityOfStates, dispersion = getDensityOfStates(tightBindDisp, size_BZ)
kondoJArrays = Dict()
for W_val in W_val_arr
    kondoJArray, _ = momentumSpaceRG(size_BZ, omega_by_t, J_val, W_val, orbitals)
    averageKondoScale = sum(abs.(kondoJArray[:, :, 1])) / length(kondoJArray[:, :, 1])
    @assert averageKondoScale > RG_RELEVANCE_TOL
    kondoJArray[:, :, end] .= ifelse.(abs.(kondoJArray[:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, kondoJArray[:, :, end], 0)
    kondoJArrays[W_val] = kondoJArray
end

function probe(kondoJArrays, dispersion)
    collatedResults = []
    subFigTitles = []
    @time for W_val in W_val_arr
        push!(subFigTitles, L"W/J=%$(round(W_val, digits=3))")
        results_arr = scattProb(kondoJArrays[W_val], size_BZ, dispersion, fractionBZ)
        push!(collatedResults, [results_arr[1], results_arr[2]])
    end
    saveName = "scattprob-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(J_val, digits=4)).pdf"
    plotHeatmaps([x_arr, x_arr], [L"$ak_x/\pi$", L"$ak_y/\pi$"], [L"$\Gamma/\Gamma_0$", L"relevance of $\Gamma$"], subFigTitles, collatedResults, saveName)
end


function corr(kondoJArrays, dispersion)
    collatedResultsSpin = []
    corrSpinFlip = i -> [("+-+-", [2, 1, 2 * i + 1, 2 * i + 2], 1.0), ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1], 1.0)]
    subFigTitles = LaTeXString[]
    @showprogress for W_val in W_val_arr
        push!(subFigTitles, L"W/J=%$(round(W_val, digits=3))")

        @time savePaths, activeStatesArr = getIterativeSpectrum(size_BZ, dispersion, kondoJArrays[W_val][:, :, end], 0 * W_val, orbitals, fractionBZ)
        resSpin, resSpinBool = correlationMap(savePaths[end], corrSpinFlip, 1, Dict(activeStatesArr[end] .=> 1:length(activeStatesArr[end])), size_BZ)

        push!(collatedResultsSpin, [resSpin, resSpinBool])
        display(maximum(abs.(resSpin)))
    end
    saveName = "charge-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_val_arr), digits=4))_$(round(maximum(W_val_arr), digits=4))_$(round(J_val, digits=4)).pdf"
    plotHeatmaps([x_arr, x_arr], [L"ak_x/\pi", L"ak_y/\pi"], [L"\chi_s(d, \vec{k})", L"\chi_c(\vec{k})", L"n_{k\uparrow}n_{k\downarrow}"], subFigTitles, collatedResultsSpin, saveName)
end

probe(kondoJArrays, dispersion);
corr(kondoJArrays, dispersion);
