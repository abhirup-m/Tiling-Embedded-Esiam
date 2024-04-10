include("source/main.jl")

_ = manager(21, 0.2, collect(-0.397:-0.002:-0.41), ["p", "p"], ["scattProb"], figScale=1.0)
_ = manager(41, 0.2, collect(-0.397:-0.002:-0.41), ["p", "p"], ["scattProb"], figScale=1.0)
_ = manager(41, 0.2, collect(-0.397:-0.002:-0.41), ["p", "p"], ["scattProb"], figScale=1.0)
