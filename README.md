# Lattice-Embedded Impurity Model

## Implementation Algorithm
- Solve RG equations on a discrete N x N lattice, in momentum space, to obtain the fixed point values of the Kondo coupling matrix J[k,q].
- Calculate Kondo scattering probability
- Create and diagonalise RG fixed-point Hamiltonians
- Calculate static correlations
- Calculate entanglement measures

## Solving renormalisation group equations

- Choice of omega: $\omega = -2t (\sim -D_0/2)$. Choose RG scheme by selecting isoenergetic energy shells $E_n = 2t|cos(kx) + 1|$ with  $kx=0,\pi/N, 2\pi/N,\ldots,\pi$.

- At each energy shell En, states on this shell renormalise all shells _within_ it.
$$\Delta J_{k_1,k_2} = \sum_{q} \rho_q \left(J_{k_1,q} J_{q, k_2}/\mathcal{D} + \ldots\right)$$

- The density of states $\rho$ is of course for the 2D tight-binding lattice.

- In order to ease the complexity, calculate only the renormalisation only for $k_1,k_2$ in the left half of the Brillouin zone, and for $k_1\neq k_2$. The other points are obtained through the fact that the RG equations are invariant under $\vec k_1, \vec k_2 \to \vec k_2, \vec k_1$, and it flips sign under $\vec k_1 \to \vec k_1 + \vec \pi$ or $\vec k_2 \to \vec k_2 + \vec \pi$.

- Make sure to round off the J_kq matrix at every step (including the zeroth step when the matrix is created from the p-wave definition).

- Finally define an average Kondo scale $\bar J = \langle |J^{(0)}_{k_1, k_2} | \rangle$, and if a certain $J^*_{k_1, k_2} / \bar J$ is less than a predefined TOLERANCE, set that $J^*_{k_1, k_2}$ to zero.

## Calculating Kondo Scattering Probability

- For all points except the four corner points and the center point, calculate 

$$\Gamma_k = \frac{\sum_{q,\varepsilon_q<\varepsilon_q<k} (J^*_{kq})^2}{\sum_{q,\varepsilon_q<\varepsilon_q<k} (J^{(0)}_{kq})^2}$$

## Diagonalisation of fixed-point Hamiltonian

- Decide on a maximum energy cutoff till which we want to calculate objects (something like 25% of the bandwidth). Also decide on the number TRUNC_DIM of $k-$states we want to keep for every diagonalisation. This should be small like 3 or 4.

- Create all possible combinations of size TRUNC_DIM of the $k-$states within the selected fraction of the Brillouin zone. Each such combinations $\mathcal{C}$ corresponds to a Hamiltonian $H(\mathcal{C})$, formed by using the fixed point values of the couplings.

- Diagonalise each such Hamiltonian $H(\mathcal{C})$, and note the ground state. Make sure to round off the Hamiltonian matrix to some precision like `1e-10` to remove stray values.
