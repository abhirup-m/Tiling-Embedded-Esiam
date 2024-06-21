# Lattice-Embedded Impurity Model

## Implementation Algorithm
1. Solve the unitary RG equations on a discrete lattice in momentum space to obtain the fixed point values of the Kondo coupling matrix $J$.
2. Calculate Kondo scattering probability.
3. Create and diagonalise RG fixed-point Hamiltonians.
4. Calculate static correlations.
5. Calculate entanglement measures.

## Solving renormalisation group equations

**Preliminaries**: Decide on a Brillouin zone size $N_\text{BZ}\times N_\text{BZ}$. Choice of omega: $\omega = -2t$. This is guided by the fact that this value is half of $-D_0$, $D_0$ itself being the maximum possible kinetic energy. Choose RG scheme by selecting isoenergetic energy shells $E_n = 2t|cos(kx) + 1|$ with  $kx=0,\pi/N, 2\pi/N,\ldots,\pi$.

Solve the unitary RG equations on a discrete $N_\text{BZ}\times N_\text{BZ}$ lattice, in momentum space, to obtain the fixed point values of the Kondo coupling $J$. The object $J$ will either by a matrix of dimensions $N_\text{BZ}^2\times N_\text{BZ}^2$, or a rank four tensor of dimensions $N_\text{BZ}\times N_\text{BZ}\times N_\text{BZ}\times N_\text{BZ}$, depending on the implementation (it's simpler to just use the matrix representation). Each matrix element $J[k,q]$ represents a Kondo term $J_{k,q} \sum_{\alpha,\beta}\vec{S}_d\cdot\vec{\sigma}_{\alpha\beta}c^\dagger_{k\alpha}c_{q\beta}$.

**Computation step**: 
At each energy shell En, states on this shell as well as the particle-hole transformed shell $-E_n$ renormalise all shells _within_ them.
$$\Delta J_{k_1,k_2} = \sum_{q} \rho_q \left(J_{k_1,q} J_{q, k_2}/\mathcal{D} + \ldots\right).$$
More explicitly, the UV states $q$ are drawn from the set of states with energies in a small window around $|E_n|$ and $-|E_n|$:
$$|E_n| - \epsilon < |\varepsilon(q)| < |E_n| + \epsilon, \epsilon \to 0^+,$$
while the IR states $k_1$ and $k_2$ are from the states within these two shells:
$$|\varepsilon_{k_1}| < |E_n|~.$$

At the end of the RG, define an average Kondo scale $J_\text{avg} = \langle |J^{(0)}_{k_1, k_2} | \rangle$, and if a certain $J^*_{k_1, k_2} / J_\text{avg}$ is less than a predefined `TOLERANCE`, set that $J^*_{k_1, k_2}$ to zero.

**Calculation of the density of states**: The density of states $\rho_q$ is the momentum-resolved density of states for the 2D tight-binding lattice. It represents the number of states per unit energy around the momentum point $q$, and is calculated from the expression
$$\rho_q = \frac{dn(q)}{dq}\frac{dq}{dE(q)},$$
where $dq = (2\pi/L)^2$ is the area of a unit cell in momentum space, $dn(q)=1$ is the number of states within that unit cell, and $\frac{dE}{dq}$ is obtained from the dispersion relation. The density of states is finally normalised using the condition that the total integral (sum?) of the density of states should be equal to the total number of $k-$sites (filling all energy states in the 1-particle picture should be equivalent to occupying all momentum states).

The actual numerical expression used to determine the DOS at the momentum states $(k_x^n, k_y^n)$ is
$$4\left[\left(E(k_x^{n+1}, k_y^n) - E(k_x^{n-1}, k_y^n)\right)^2 + \left(E(k_x^{n}, k_y^{n+1}) - E(k_x^n, k_y^{n-1})\right)^2\right]^{-1/2}~.$$

**Note**: Make sure to round off the $J$ matrix at every step (including the zeroth step when the matrix is created from the p-wave definition).

## Calculating Kondo Scattering Probability

For all points except the four corner points and the center point, calculate 

$$\Gamma_k = \frac{\sum_{q,\varepsilon(q)<\varepsilon(k)} (J^*_{kq})^2}{\sum_{q,\varepsilon(q)<\varepsilon(k)} (J^{(0)}_{kq})^2}$$

## Diagonalisation of fixed-point Hamiltonian

Decide on a maximum energy cutoff `BZ_FRAC` till which we want to calculate objects (`BZ_FRAC ~ 0.25`). Also decide on the number `TRUNC_DIM` of $k-$states we want to keep for every diagonalisation (`TRUNC_DIM ~ 3`). Create all possible combinations of size `TRUNC_DIM` of the $k-$states within an energy `BZ_FRAC x 4t`. Each such combinations $\mathcal{C}$ corresponds to a Hamiltonian $H(\mathcal{C})$, formed by using the fixed point values of the couplings. Diagonalise each such Hamiltonian $H(\mathcal{C})$, and note the ground state.

Make sure to round off the Hamiltonian matrix to some precision like `1e-10` to remove stray values.


## Calculation of single momentum static correlations $\langle \mathcal{O}_d \mathcal{O}^\dagger_k\rangle$

**Preparation**: These are correlations that involve a single momentum label in the scattering vertex. The resultant correlation result $R$ will be a matrix of dimensions $N_\text{BZ}\times N_\text{BZ}$, or a vector of size $N_\text{BZ}^2$, depending on the implementation. Each matrix element `R[k_x, k_y]` represents the correlation value at momentum $(k_x, k_y)$. It is important to initialise this matrix to zeros.

**Steps**: The correlation values are calculated by performing the following steps for each combination $\mathcal{C}$:
1. Take the spectrum $\mathcal{S}(\mathcal{C})$ for the combination, and create `TRUNC_DIM` number of operators $O_{k_1}, O_{k_2},\ldots$ of the form $\mathcal{O}_d \mathcal{O}^\dagger_{k_1}, \mathcal{O}_d \mathcal{O}^\dagger_{k_2}, \ldots$ where $\{k_i\}$ are drawn from the combination $\mathcal{C}$ (recall that $\mathcal{C}$ is simply a collection (of size `TRUNC_DIM`) of momentum states drawn from the full set of $k-$ states).
2. Calculate the expectation value of each such operator $O_k$ using the ground states of the spectrum $\mathcal{S}(\mathcal{C})$, and _add_ this value to the matrix element `R[k_x, k_y]`, subject to the condition in point 3.
3. **Make the change in point 2 only if the energy of this $k-$state is the largest in magnitude among the set of $k-$states in the sequence $\mathcal{C}$.** That is, if the sequence has three states $k_1, k_2$ and $k_3$ with energy magnitudes 0, 0.5 and 1 respectively, only `R[k3]` will be modified by this sequence. This is done to ensure that each $k-$state only picks up correlations from its final emergent window in terms of the RG flow.

These three steps need to be carried out for all the combinations. 

We also need to keep a count of the number of combinations each $k-$state has received support from, in order to finally average over all combinations. This can be done by initialising a zero vector of the size of the total number of $k-$states, and incrementing its $m^\text{th}$ element by 1 every time the $m^\text{th}$ $k-$state receives contribution from any combination. Zero contributions will also have to be counted, of course. Once all the combinations have been worked through, divide each correlation matrix element  by the number of contributions it received, in order to average over all combinations. 

**Summary**: The essence of the previous points can be captured through the following expression:
$$
R[k] = \frac{1}{\sum_{\mathcal{C}}\delta_{\varepsilon(k), \text{max}\{\varepsilon(\mathcal{C})\}}}\sum_{\mathcal{C}}\delta_{\varepsilon(k), \text{max}\{\varepsilon(\mathcal{C})\}}\sum_{n \in \text{GS}(\mathcal{C})}\frac{|\langle\psi_n | \mathcal{O}_d \mathcal{O}^\dagger_{k} | \psi_n\rangle|}{d_\text{GS}(\mathcal{C})}
$$

## Calculation of 2-momentum static correlations $\langle \mathcal{O}_{k_1} \mathcal{O}^\dagger_{k_2}\rangle$

The approach for computing these correlations are very similar to the previous section. We just highlight any differences here.
- In practise, we will be interested in computing these 2-momentum correlations where one of the momenta is fixed at a specific state $k^*$ (for eg., the node, the antinode or some offset point). It therefore doesn't make sense to calculate the full $N_\text{BZ}^2 \times N_\text{BZ}^2$ matrix, and instead calculate, say, three matrices for the correlations $\langle \mathcal{O}_{k^*} \mathcal{O}^\dagger_{k}\rangle$, where $k^*$ is the $k-$state we are interested in (as mentioned before, physically motivated choices can be the nodal state or the antinodal state) and it remains fixed during the process, while $k$ is the usual momentum state that varies over the Brillouin zone (as in the previous section).
- With this choice, the calculation reduces to that of the previous section, with the change $\mathcal{O}_d \to \mathcal{O}_{k^*}$. The succint expression that captures all the details is
$$
R_{k^*}[\vec k] = \frac{\sum_{\mathcal{C}}\delta_{\varepsilon(k), \text{max}\{\varepsilon(\mathcal{C})\}}\delta_{\varepsilon(k^*), \text{max}\{\varepsilon(\mathcal{C})\}}\sum_{n \in \text{GS}(\mathcal{C})}\frac{|\langle\psi_n | \mathcal{O}_{k^*} \mathcal{O}^\dagger_{k} | \psi_n\rangle|}{d_\text{GS}(\mathcal{C})}}{\sum_{\mathcal{C}}\delta_{\varepsilon(k), \text{max}\{\varepsilon(\mathcal{C})\}\delta_{\varepsilon(k^*), \text{max}\{\varepsilon(\mathcal{C})\}}}}~.
$$
Note that similar to point 3 of the previous section, we only consider contributions to $k$ from those sequences where both $k$ and $k^*$ are the states with the maximum kinetic energy.

## Calculation of entanglement measures

This proceeds similar to the previous two sections, depending on whether we are calculating single momentum measure like $S_\text{EE}(k)$ or a two momentum measure like $I_2(k_1:k_2)$. We just replace the expectation value calculation step with the entanglement calculation step:
$$
S_\text{EE}(k) = \frac{\sum_{\mathcal{C}}\delta_{\varepsilon_{k}, \text{max}\{\varepsilon(\mathcal{C})\}}\sum_{n \in \text{GS}(\mathcal{C})}\frac{S_\text{EE}(\psi_n; k)}{d_\text{GS}(\mathcal{C})}}{\sum_{\mathcal{C}}\delta_{\varepsilon(k), \text{max}\{\varepsilon(\mathcal{C})\}}}~,\\
I_2[k_1, k_2] = \frac{\sum_{\mathcal{C}}\delta_{\varepsilon_{k_1}, \text{max}\{\varepsilon(\mathcal{C})\}}\delta_{\varepsilon_{k_2}, \text{max}\{\varepsilon(\mathcal{C})\}}\sum_{n \in \text{GS}(\mathcal{C})}\frac{I_2(\psi_n; k_1: k_2)}{d_\text{GS}(\mathcal{C})}}{\sum_{\mathcal{C}}\delta_{\varepsilon(k_1), \text{max}\{\varepsilon(\mathcal{C})\}\delta_{\varepsilon(k_2), \text{max}\{\varepsilon(\mathcal{C})\}}}}~.
$$

## Useful optimisations

- The RG equation solving process is in general numerically quite intensive because of the large number of momentum pairs participating in the process, but it can be optimised by calculating the renormalisation only for $k_1,k_2$ in the left half of the Brillouin zone, and for $k_1\leq k_2$. The other points are obtained through the fact that (i) the RG equations are invariant under $(\vec k_1, \vec k_2) \to (\vec k_2, \vec k_1)$, and it flips sign under $\vec k_1 \to \vec k_1 + \vec \pi$ or $\vec k_2 \to \vec k_2 + \vec \pi$.

- The Hamiltonian diagonalisation process is also made intensive by the fact that there will typically be a very large number of Hamiltonians that need to be diagonalised arising from the various combinations of $k-$states. This process can be optimised by noting that for many of the combinations, the Hamiltonians will be identical up to a relabelling of $k-$states (because of symmetries in $k-$space), and it suffices to diagonalise only one member of such groups of identical Hamiltonians, and obtain the values for all the other Hamiltonians simply by relabelling the final results.

- The correlation function calculation can be optimised by computing the correlations only in the one octant of the Brillouin zone, and then setting the values at the other octants by using the $C_4$ symmetry of the Brillouin zone. This should also be incorporated in the previous point to determine which combinations need to be diagonalised. It suffices to only form combinations out of the $k-$ states within a single octant.
