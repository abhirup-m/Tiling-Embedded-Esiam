using ProgressMeter, Measures, CairoMakie

const TRUNC_DIM = 2
include("./source/constants.jl")
include("./source/helpers.jl")
include("./source/rgFlow.jl")
include("./source/probes.jl")

const colorSet = resample_cmap(:YlOrBr, 3)
function getPhase(J_val, W_val)
    kondoJArray, dispersion = momentumSpaceRG(size_BZ, omega_by_t, J_val, W_val, orbitals)
    averageKondoScale = sum(abs.(kondoJArray[:, :, 1])) / length(kondoJArray[:, :, 1])
    @assert averageKondoScale > RG_RELEVANCE_TOL
    kondoJArray[:, :, end] .= ifelse.(abs.(kondoJArray[:, :, end]) ./ averageKondoScale .> RG_RELEVANCE_TOL, kondoJArray[:, :, end], 0)
    scattProbBool = scattProb(kondoJArray, size_BZ, dispersion, fractionBZ)[2]
    if all(>(0), scattProbBool[fermiPoints])
        return 1
    elseif all(==(0), scattProbBool[fermiPoints])
        return 3
    else
        return 2
    end
end

const omega_by_t::Float64 = -2.0
const orbitals::Tuple{String, String} = ("p", "p")
const numPoints::Int64 = 100
const fractionBZ::Float64 = 0.3
const size_BZ::Int64 = 17

densityOfStates, dispersionArray = getDensityOfStates(tightBindDisp, size_BZ)
node = map2DTo1D(float(π)/2, float(π)/2, size_BZ)
antinode = map2DTo1D(float(π), 0.0, size_BZ)
fermiPoints = unique(getIsoEngCont(dispersionArray, 0.0))

W_val_arr = -1.0 * range(0.0, stop=0.3, length=numPoints)
J_val_arr = 1.0 * range(0.05, stop=0.6, length=numPoints)

@time phaseDiagram = [getPhase(J_val_arr[i], W_val_arr[j]) for (i, j) in Iterators.product(1:numPoints, 1:numPoints)]
println(unique(phaseDiagram))
f = Figure()
ax = Axis(f[1, 1])
heatmap!(ax, J_val_arr, -1 .* W_val_arr, phaseDiagram;
            xlabel="Kondo Int. \$J\$", ylabel="Bath Int. \$-W\$", colormap=:YlOrBr, 
            thickness_scaling=1.5)
display(f)
save("phaseDiagram.pdf", f)
