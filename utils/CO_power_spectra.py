import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from colossus.cosmology import cosmology
from colossus.lss import bias, mass_function


def IR_lumi_func(L_IR, z):
    if z <= 1.1:
        phi_star = 5.7*10**-3*(1+z)**-0.57
    else:
        phi_star = 6.81*10**-2*(1+z)**-3.92

    if z <= 1.85:
        L_IR_star = 7.68*10**9*(1+z)**3.55
    else:
        L_IR_star = 5.80*10**10*(1+z)**1.62

    if z <= 0.3:
        alpha = 1.15
    else:
        alpha = 1.2

    if z <= 0.3:
        sigma = 0.52
    else:
        sigma = 0.5

    phi_L = phi_star * (L_IR/L_IR_star)**(1 - alpha) * \
        np.exp(-1 / (2*sigma**2)*(np.log10(1 + L_IR/L_IR_star))**2)

    return phi_L


def IR_lumi_func_integral(L_IR, z):
    ir = 10**np.arange(5, 16, 0.01)
    y = IR_lumi_func(ir, z)
    idx = np.where(ir >= L_IR)[0]
    integral = np.trapz(y[idx], np.log10(ir[idx]))
    return integral


cosmo_params = {'flat': True, 'H0': 67.77, 'Om0': 0.307115,
                'Ob0': 0.048206, 'sigma8': 0.8228, 'ns': 0.96,
                'de_model': 'lambda', 'w0': -1}


# its_params = {'alpha': 1.24, 'beta': 2.39}
its_params = {'alpha': 1.17, 'beta': 0.28}

Mpc = 3.0856776e19  # km
Lsun = 3.839e26  # unit W
Jy = 1e-26  # W/m^2/Hz
GHz = 1e9  # Hz
c_light = 3e8  # m/s


def theoretical_power_spectrum(k, z, cosmo_params=cosmo_params,
                               its_params=its_params):

    cosmo = cosmology.setCosmology('myCosmo', params=cosmo_params)

    M = 10**np.arange(8, 17, 0.01)
    dndm_integrand = mass_function.massFunction(M, z=z, q_out='dndlnM',
                                                model='sheth99') / M  # (Mpc/h)^-3

    d_L = cosmo.luminosityDistance(z=z)  # Mpc/h
    d_A = cosmo.angularDiameterDistance(z=z) * (1 + z)  # Mpc/h
    hubble = cosmo.Hz(z=z) / Mpc  # s^-1 or Hz
    lam_rest = c_light / (115.27 * GHz)  # m
    y_z = lam_rest * (1 + z)**2 / hubble  # m/Hz
    h = cosmo.Hz(z=0) / 100.  # no unit
    Jy_to_uK = 32.6 / (115.27/(1 + z))**2

    def mass_function_integral(M_h):
        idx = np.where(M >= M_h)[0]
        integral = np.trapz(dndm_integrand[idx], M[idx])
        return integral

    mass_func = np.vectorize(mass_function_integral)
    lumi_func = np.vectorize(IR_lumi_func_integral)

    mass_range = 10**np.arange(10, 15.1, 0.1)
    n_mass = mass_func(M)

    fit_n = interp1d(n_mass[::-1], M[::-1],
                     kind='linear', fill_value='extrapolate')

    IR_lumi = 10**np.arange(5, 15.1, 0.1)
    num = lumi_func(IR_lumi, z=z) * h**-3
    Halo_mass = fit_n(num)  # M_sun/h

    alpha = its_params['alpha']
    beta = its_params['beta']

    CO_prime = 10**((np.log10(IR_lumi) - beta) / alpha)
    CO_lumi = 4.9 * 10**-5 * CO_prime  # L_sun

    lumi_mass_relation = interp1d(Halo_mass, CO_lumi,
                                  kind='linear', fill_value='extrapolate')

    #halo_bias = bias.haloBias(mass_range, model='sheth01', z=z)
    halo_bias = bias.haloBias(mass_range, model='tinker10', z=z, mdef='vir')
    Lco = lumi_mass_relation(mass_range)

    dndm = mass_function.massFunction(mass_range, z=z, q_out='dndlnM',
                                      model='sheth99') / mass_range

    galaxy_bias = np.trapz(dndm * Lco * halo_bias, mass_range)
    weights = np.trapz(dndm * Lco, mass_range)

    weighted_bias = galaxy_bias / weights

    Lco_unit = Lco * Lsun  # W
    dndm_unit = dndm * (Mpc/h * 10**3)**-3  # m^-3

    mean_intensity = np.trapz(
        dndm_unit * Lco_unit * y_z * d_A**2 / Jy * Jy_to_uK / (4 * np.pi * d_L**2), mass_range)

    matter_power = cosmo.matterPowerSpectrum(k=k, z=z)
    clustering_ps = weighted_bias**2 * mean_intensity**2 * matter_power

    shot_noise = np.trapz(
        dndm_unit * (Lco_unit * y_z * d_A**2 / Jy * Jy_to_uK / (4 * np.pi * d_L**2))**2, mass_range) * (1 / (Mpc * 10**3))**3 * h**3

    total_power = clustering_ps + shot_noise

    return k, total_power, clustering_ps, Halo_mass, CO_lumi
