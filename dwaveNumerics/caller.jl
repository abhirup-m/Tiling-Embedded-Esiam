include("./source/main.jl")

J_val = 0.1
size_BZ = 5
manager(size_BZ, -2.0, J_val, [-19, -19.5, -20.] ./ size_BZ, ("p", "p"), ["scattProb", "spinFlipCorrMap"])
# manager(size_BZ, -2.0, J_val, [0.0, -1.3, -2.3, -2.9, -3.9, -4.9, -28.0] ./ size_BZ, ("p", "p"), ["spinFlipCorrMap"])
