include("./source/main.jl")
include("./source/plotting.jl")

J_val = 0.1
size_BZ = 33
omega_by_t = -2.0
W_by_J_arr = -1.0 .* [56, 57, 58, 59, 60] ./ size_BZ
orbitals = ("p", "p")
savePaths = rgFlowData(size_BZ, omega_by_t, J_val, W_by_J_arr, orbitals)
x_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi

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

collatedResultsPm = []
corrDefArray = [i -> Dict(("+-+-", [2, 1, 2 * i + 1, 2 * i + 2]) => 1.0, ("+-+-", [1, 2, 2 * i + 2, 2 * i + 1]) => 1.0)]
               
@showprogress for (i, savePath) in collect(enumerate(savePaths))
    jldopen(savePath, "r"; compress=true) do file
    kondoJArray = file["kondoJArray"]
    dispersion = file["dispersion"]
    W_val = file["W_val"]
    size_BZ = file["size_BZ"]
    orbitals = file["orbitals"]
    dispersion = file["dispersion"]

    # set W_val to zero so that it does not interfere with the spin-flip fluctuations.
    res_pm = getCorrelations(size_BZ, dispersion, kondoJArray, 0.0, orbitals, corrDefArray)
    push!(collatedResultsPm, [log10.(res_pm[1][1]), res_pm[1][2]])
    end
end

saveName = "spm-$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4)).pdf"
plotHeatmaps([x_arr, x_arr], [L"$ak_x/\pi$", L"$ak_y/\pi$"], [L"$\chi(d, \vec k)$", L"relevance of $\chi(d, \vec k)$"], collatedResultsPm, saveName)
