# Manuscript and Numerical Codes For "Mott Criticality as the Confinement Transition of a Pseudogap-Mott Metal"

Preprint can be found at [arXiv:2507.17201](https://arxiv.org/abs/2507.17201).

## Dependencies

- Distributed
- Random
- ProgressMeter
- LsqFit
- LinearAlgebra
- CSV
- JLD2
- FileIO
- CodecZlib
- Distributed
- Combinatorics
- Serialization
- Fermions [[custom module](https://github.com/abhirup-m/Fermions.jl/tree/TilingSIAM)]
- JSON3

## Generating results

All modules are executed by running the script `caller.jl`:

```
julia -tauto -p auto caller.jl
```

## Description of Caller Modules


- `RGFlow(...)`: Generates renormalised low-energy Hamiltonian couplings
- `ScattProb(...)`: Calculates renormalised scattering probabilities for momentum states in the conduction bath
- `KondoCouplingMap(...)`: Calculates distribution of renormalised Kondo coupling strength as a function of momentum indices
- `Auxiliary Correlations(...)`: Computes $k-$space impurity-bath correlation functions (spin and charge correlations as well entanglement measures)
- `AuxiliaryRealSpaceEntanglement(...)`: Computes real-space impurity-bath correlation functions (spin correlations, entanglement measures and dynamical correlations such as spectral function and self-energy)
- `LatticeKspaceDOS(...)`: Periodises auxiliary model dynamical correlations into the tiled spectral function and self-energy of the lattice model
- `TiledSpinCorr(...)`: Periodises auxiliary model static correlations into the tiled correlations of the lattice model
- `TiledEntanglement(...)`: Periodises auxiliary model entanglement measures into the tiled measures of the lattice model
- `PhaseDiagram(...)`: Generates RG phase diagram as function of the Kondo coupling and bath correlation strength

