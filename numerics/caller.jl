include("./source/main.jl")
include("./source/plotting.jl")

J_val = 0.1
size_BZ = 13
omega_by_t = -2.0
W_by_J_arr = -1.0 .* [57.5, 59, 60, 62] ./ size_BZ
orbitals = ("p", "p")
savePaths = rgFlowData(size_BZ, omega_by_t, J_val, W_by_J_arr, orbitals)
x_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
collatedResults = []
saveName = "$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4))"

for (i, savePath) in enumerate(savePaths)
    jldopen(savePath, "r"; compress=true) do file
    kondoJArray = file["kondoJArray"]
    dispersion = file["dispersion"]
    W_val = file["W_val"]
    size_BZ = file["size_BZ"]
    orbitals = file["orbitals"]

    averageKondoScale = sum(abs.(kondoJArray[:, :, 1])) / length(kondoJArray[:, :, 1])
    @assert averageKondoScale > RG_RELEVANCE_TOL
    kondoJArray[:, :, end] .= ifelse.(abs.(kondoJArray[:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, kondoJArray[:, :, end], 0)
    dispersion = file["dispersion"]

    results_arr = scattProb(kondoJArray, size_BZ, dispersion)
    push!(collatedResults, [log10.(results_arr[1]), results_arr[2]])
    end
end

plotHeatmaps(x_arr, collatedResults, saveName)
