include("./source/main.jl")
include("./source/plotting.jl")

J_val = 0.1
size_BZ = 33
omega_by_t = -2.0
W_by_J_arr = -1.0 .* [57.5, 59, 60, 62] ./ size_BZ
orbitals = ("p", "p")
savePaths = rgFlowData(size_BZ, omega_by_t, J_val, W_by_J_arr, orbitals)

fig, axes = plt.subplots(length(savePaths), 2)
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

    # reshape to 1D array of size N^2 into 2D array of size NxN, so that we can plot it as kx vs ky.
    # kondoJArray = reshape(kondoJArray[:,:,[1,end]], (size_BZ^2, size_BZ^2, 2))

    results_arr = scattProb(kondoJArray, size_BZ, dispersion)
    results_arr[1] .= log10.(results_arr[1])
    x_arr = y_arr = range(K_MIN, stop=K_MAX, length=size_BZ) ./ pi
    subtitles = [L"\mathrm{rel(irrel)evance~of~}\Gamma(k)", L"\Gamma(k)/\Gamma^{(0)}(k)"]
    fig, axes[i-1, 0:1] = plotHeatmaps(axes[i-1, 0:1], results_arr, x_arr, y_arr)
    end
end
fig.show()
# fig = plot(figs..., size=(250 * 2, 250 * length(savePaths)), layout = grid(length(savePaths), 2))
saveName = "$(orbitals[1])-$(orbitals[2])_$(size_BZ)_$(omega_by_t)_$(round(minimum(W_by_J_arr), digits=4))_$(round(maximum(W_by_J_arr), digits=4))_$(round(J_val, digits=4))"
fig.savefig("scattProb-$(saveName).pdf", bbox_inches="tight")
