using ProgressMeter, Measures, CairoMakie, LaTeXStrings

const TRUNC_DIM = 2
include("./source/constants.jl")
include("./source/helpers.jl")
include("./source/rgFlow.jl")
include("./source/probes.jl")

const cmap = resample_cmap(:cherry, 20)[1:end-1]
const linecmap = :Dark2_3
const phaseMaps = Dict("FL" => 2, "PG" => 1, "MI" => 3)
const scatterLabels = ["N gaps", "AN gaps", "MP gaps"]
const colorSet = resample_cmap(cmap, length(phaseMaps))
const lineColorSet = resample_cmap(linecmap, 3)
const labelSet = LaTeXString.([sort(collect(phaseMaps), by=last) .|> first; scatterLabels])

const omega_by_t::Float64 = -2.0
const orbitals::Tuple{String, String} = ("p", "p")
const numPoints::Int64 = 150
const fractionBZ::Float64 = 0.3
const size_BZ::Int64 = 17

J_val_arr = collect(range(0.05, stop=0.6, length=numPoints))
W_val_arr = -1 .* collect(range(0.05, 0.2, length=numPoints))
# phaseDiagram, nodeGap, antiNodeGap, midPointGap = PhaseDiagram(J_val_arr, W_val_arr, numPoints, phaseMaps)
f = Figure(size=(400, 300), figure_padding=8, fontsize=14)
ax = Axis(f[1, 1], xlabel=L"Kondo Int. $(J)$", ylabel=L"Bath Int. $(|W|)$")
heatmap!(ax, J_val_arr, abs.(W_val_arr), phaseDiagram;
         colormap=cmap, legend=true,)

sc1 = lines!(ax, J_val_arr[nodeGap .!= nothing], abs.(nodeGap[nodeGap .!= nothing]),
       color=lineColorSet[1])
sc2 = lines!(ax, J_val_arr[antiNodeGap .!= nothing], abs.(antiNodeGap[antiNodeGap .!= nothing]),
             color=lineColorSet[2], linestyle=:dash)
sc3 = lines!(ax, J_val_arr[midPointGap .!= nothing], abs.(midPointGap[midPointGap .!= nothing]),
                    color=lineColorSet[3], linestyle=(:dashdotdot, :dense))
legnd = [[PolyElement(color=col) for col in colorSet]; [sc1, sc2, sc3]]
axislegend(ax, legnd, labelSet, markersize=2)
save("phaseDiagram.pdf", f, px_per_unit=4)
save("phaseDiagram.png", f, px_per_unit=8)
