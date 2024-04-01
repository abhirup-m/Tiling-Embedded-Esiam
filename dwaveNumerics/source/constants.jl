# define the edges of the Brillouin zone
const K_MIN = -pi
const K_MAX = pi

# tolerance for identifying energy contours
const TOLERANCE = 1e-10

# overlap integral 't' set to 1
const HOP_T = 1.0

# for testing. 
const SIZE_BZ = [5, 101, 1001]
const KX_VALUES = [K_MIN, K_MAX, K_MIN, K_MAX, 0]
const KY_VALUES = [K_MIN, K_MIN, K_MAX, K_MAX, 0]
const POINTS(num_kspace) = [
    1,
    num_kspace,
    num_kspace^2 - num_kspace + 1,
    num_kspace^2,
    Int(0.5 * num_kspace^2 + 0.5),
]

