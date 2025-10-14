import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, kv, jvp, kvp
from scipy.optimize import root_scalar

# jv = Bessel function 1st kind
# jvp = First derivative of Bessel function 1st kind
# kv = Bessel function 2nd kind
# kvp = First derivative of Bessel function 2nd kind

# -----------------------------------------------------------
# Characteristic equation for step-index fibre (scalar form)
# -----------------------------------------------------------
def characteristic_eq(u, v_squared, l):
    """Characteristic equation F(u)=0 for LP_lm modes."""
    # Ensure valid region
    if u <= 0 or u**2 >= v_squared:
        return np.nan

    w = np.sqrt(v_squared - u**2)

    # conventional dispersion relation
    lhs = jvp(l, u) / (u * jv(l, u))
    rhs = -kvp(l, w) / (w * kv(l, w))

    # calculated dispersion relation
    lhs = (u * jvp(l, u)) / jv(l, u)
    rhs = (w * kvp(l, w)) / kv(l, w)

    return lhs - rhs


# -----------------------------------------------------------
# Find roots u_lm(V) using scanning + root_scalar
# -----------------------------------------------------------
def find_modes(v_squared, l):
    """Find all roots u for given V and mode order l."""
    # create array of u values (like an x-axis)
    us = np.linspace(1e-6, v_squared - 1e-6, 1000)
    # calculate the function values for each of these u values
    F = np.array([characteristic_eq(u, v_squared, l) for u in us])
    roots = []
    # loop through the function values in pairs
    for i in range(len(us) - 1):
        # check that the function is defined
        if np.isnan(F[i]) or np.isnan(F[i + 1]):
            continue
        # if the function changes sign, there is a root between the two u values
        if F[i] * F[i + 1] < 0:
            sol = root_scalar(characteristic_eq, args=(v_squared, l),
                              bracket=[us[i], us[i + 1]])
            if sol.converged:
                roots.append(sol.root)
    return roots


def plot_radial_fields(v_squared, a, l, roots):
    r = np.linspace(0, 2.5 * a, 1000)  # up to 2.5x core radius
    plt.figure(figsize=(8, 5))

    for m, u in enumerate(roots, 1):
        w = np.sqrt(v_squared - u ** 2)
        R = np.zeros_like(r)

        # Inside core (r<a)
        inside = r < a
        R[inside] = jv(l, u * r[inside] / a)

        # Outside core (r>a)
        outside = r >= a
        R[outside] = jv(l, u) / kv(l, w) * kv(l, w * r[outside] / a)

        # Normalise peak amplitude
        R /= np.max(np.abs(R))

        plt.plot(r / a, R, label=f"LP{l}{m}")

    plt.axvline(1, color='k', linestyle='--', linewidth=1, label='Core boundary')
    plt.title(f"Radial field distributions for LPₗₘ modes (l={l})")
    plt.xlabel("r / a (normalised radius)")
    plt.ylabel("Normalised field amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def solve_step_index(lambda0, core_diam, core_n, cladding_n):
    # Parameter definitions

    # Fibre geometry and refractive indices
    a = core_diam / 2  # core radius [m]
    n1 = core_n
    n2 = cladding_n
    wavelength = lambda0  # operating wavelength [m]
    k0 = 2 * np.pi / wavelength

    # Choose azimuthal mode order l (0 = LP0m modes, 1 = LP1m, etc.)
    l = 0

    # Normalised frequency range (V-number)
    v_squared = k0**2 * a**2 * (n1**2 + n2**2)
    # v_vals = np.linspace(0.1, 8, 400)
    # v_squared_vals = v_vals ** 2
    v_squared_vals = [v_squared]

    # -----------------------------------------------------------
    # Compute propagation constants for first few modes
    # -----------------------------------------------------------
    modes_data = {}  # dict: mode_index -> (V_list, beta_list)

    for v_sq in v_squared_vals:
        u_roots = find_modes(v_sq, l)
        for m, u in enumerate(u_roots):
            w = np.sqrt(v_sq - u**2)
            beta = np.sqrt((k0 * n1)**2 - (u / a)**2)
            if np.isnan(beta):
                continue

            if m not in modes_data:
                modes_data[m] = {"V": [], "beta": []}
            modes_data[m]["V"].append(np.sqrt(v_sq))
            modes_data[m]["beta"].append(beta)

    # Plot results
    plt.figure(figsize=(7, 5))
    for m, data in modes_data.items():
        plt.plot(data["V"], np.array(data["beta"]) / k0, label=f"LP{l}{m+1}", marker="o")

    plt.title(f"Step-index fibre modes (l={l})")
    plt.xlabel("V-number")
    # plt.ylabel("Propagation constant β [1/m]")
    plt.ylabel("Effective Refractive Index n_eff")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plot_radial_fields(v_squared, a, l, u_roots)


def eigenstate_func(x, y, m, n):
    return np.multiply(np.sin(m * np.pi * x / 1.), np.cos(n * np.pi * y / 1.))


def analytical_Laplace_square_plot(modes: tuple, absolute_field: bool = False):
    # eigenvalues = (n**2 + m**2) pi**2 / L**2
    # eigenfunctions = A sin(m*pi*x/L) sin(n*pi*y/L)

    # plot the radial solutions
    nm_lst = [(n, m) for n in range(1, max(modes)) for m in range(1, max(modes))]
    nm_lst.sort(key=lambda a: a[0]**2 + a[1]**2)
    nm = np.array(nm_lst)
    eigenvalues = np.pi ** 2 / 1.**2 * np.sum(nm ** 2, axis=1)

    x_line = np.linspace(0.5, 1., 1000)
    y_line = 0 * (x_line) + .5

    for mode in modes:
        m, n = nm[mode]

        field_line = eigenstate_func(x_line, y_line, m, n)

        if absolute_field:
            field_line = np.abs(field_line)

        # normalise the field lines
        field_line /= np.max(np.abs(field_line))

        plt.plot(x_line - .5, field_line, label=f"mode {mode}")

    plt.legend()
    plt.show()
    print("analytical eigenvalues:", eigenvalues)


if __name__ == "__main__":
    solve_step_index(lambda0=.5e-6, core_diam=1.5e-6, core_n=1.46, cladding_n=1.)