using Plots

function GreensFunction(
        freqValues::Vector{Float64},
        correlation::Number,
        epsilonValues::Vector{Float64},
        broadening::Number,
    )
    return [0.5 ./ (freqValues .- epsilon .+ correlation / 2 .+ 1im .* broadening) .+ 0.5 ./ (freqValues .- epsilon .- correlation / 2 .+ 1im .* broadening) for epsilon in epsilonValues]
end


W = 1
epsilonValues = range(-W/2, W/2, length=1000) |> collect
freqValues = collect(range(-2, 2, length=1000))
freqPlot = 0.5
xlims!(p, (-freqPlot, freqPlot))
lstyles = [:solid, :dash, :dashdot]
for (i, U_by_W) in enumerate([0.99, 1, 1.111])
    greensFunctionArr = GreensFunction(freqValues, U_by_W * W, epsilonValues, 0.001)
    specFunc = sum([-imag(g) for g in greensFunctionArr])
    plot!(p, specFunc, label="U/W=$(U_by_W*W)", linestyle=lstyles[i])
end
display(p)
