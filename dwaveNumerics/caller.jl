include("./source/main.jl")

manager(33, -2.0, 0.1, [0.0, 0.05], ["p", "p"], ["scattProb", "kondoCoupNodeMap", "kondoCoupAntinodeMap", "kondoCoupOffNodeMap", "kondoCoupOffAntinodeMap"], figScale=1.0)
# manager(73, -2., 0.1, [0, -0.815,-0.817, -0.819], ["p", "p"], ["scattProb", "kondoCoupNodeMap", "kondoCoupAntinodeMap", "kondoCoupOffNodeMap", "kondoCoupOffAntinodeMap"], figScale=1.)
