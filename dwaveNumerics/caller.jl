include("./source/main.jl")

manager(5, -2.0, 1.0, [0.0], ("p", "p"), ["scattProb", "spinFlipCorrMap"])

J_val = 0.1
size_BZ = 21
manager(size_BZ, -2.0, J_val, collect(-18:-0.1:-18.2) ./ size_BZ, ("p", "p"), ["scattProb", "spinFlipCorrMap"])
