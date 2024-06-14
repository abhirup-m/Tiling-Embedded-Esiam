const TRUNC_DIM = 2
include("./source/main.jl")
include("./source/plotting.jl")
J_val = 0.1
size_BZ = 33
omega_by_t = -2.0
# W_by_J_arr = -1.0 .* [0, 50, 56, 57, 58, 59, 60] ./ size_BZ
W_by_J_arr = -1.0 .* [0, 100, 200] ./ size_BZ # time = 14.318 s
orbitals = ("p", "p")
savePaths = rgFlowData(size_BZ, omega_by_t, J_val, W_by_J_arr, orbitals)
x_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi

# collatedResults = []
# for (i, savePath) in enumerate(savePaths)
#     jldopen(savePath, "r"; compress=true) do file
#     kondoJArray = file["kondoJArray"]
#     dispersion = file["dispersion"]
#     W_val = file["W_val"]
#     size_BZ = file["size_BZ"]
#     orbitals = file["orbitals"]
#     dispersion = file["dispersion"]
# 
#     results_arr = scattProb(kondoJArray, size_BZ, dispersion)
#     push!(collatedResults, [log10.(results_arr[1]), results_arr[2]])
#     end
# end
# saveName = "scattprob-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4)).pdf"
# plotHeatmaps([x_arr, x_arr], [L"$ak_x/\pi$", L"$ak_y/\pi$"], [L"$\Gamma/\Gamma_0$", L"relevance of $\Gamma$"], collatedResults, saveName)

collatedResultsSpin = []
collatedResultsCharge = []
corrSpinArray = [i -> Dict(("+-+-", [2, 1, 2 * i + 1, 2 * i + 2]) => 1.0, ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1]) => 1.0)]
corrChargeArray = [pair -> Dict(("++--", [2 * pair[2] + 1, 2 * pair[2] + 2,  2 * pair[1] + 2, 2 * pair[1] + 1]) => 1.0, ("++--", [2 * pair[1] + 1, 2 * pair[1] + 2,  2 * pair[2] + 2, 2 * pair[2] + 1]) => 1.0)]
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

    # set W_val to zero so that it does not interfere with the spin-flip fluctuations.
    resSpin = getCorrelations(size_BZ, dispersion, kondoJArray, 0.0, orbitals, corrSpinArray, 0.5)
    resCharge = getCorrelations(size_BZ, dispersion, kondoJArray, W_val, orbitals, corrChargeArray, 0.5; twoParticle=1)
    push!(collatedResultsSpin, [log10.(resSpin[1][1]), resSpin[1][2]])
    push!(collatedResultsCharge, [resCharge[1][1][node,:], resCharge[1][1][antinode,:]])
    end
end

saveName = "spin-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4)).pdf"
plotHeatmaps([x_arr, x_arr], [L"$ak_x/\pi$", L"$ak_y/\pi$"], [L"$\chi_s(d, \vec k)$", L"relevance of $\chi_s(d, \vec k)$"], collatedResultsSpin, saveName)
saveName = "charge-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4)).pdf"
plotHeatmaps([x_arr, x_arr], [L"$ak_x/\pi$", L"$ak_y/\pi$"], [L"$\chi_c(d, k_\text{N})$", L"$\chi_c(d, k_\text{AN})$"], collatedResultsCharge, saveName)
