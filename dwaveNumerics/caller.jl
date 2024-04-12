include("./source/main.jl")

manager(13, -2., 0.1, [0, -0.815,-0.817, -0.819, -0.82, -0.9], ["p", "p"], ["scattProb", "kondoCoupNodeMap", "kondoCoupAntinodeMap", "kondoCoupOffNodeMap", "kondoCoupOffAntinodeMap"], figScale=4.)
