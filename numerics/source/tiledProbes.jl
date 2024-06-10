##### Functions for calculating various probes          #####
##### (correlation functions, Greens functions, etc)    #####

using LinearAlgebra
using Combinatorics
using fermions
include("./probes.jl")

"""
Function to calculate the total Kondo scattering probability Γ(k) = ∑_q J(k,q)^2
at the RG fixed point.
"""

