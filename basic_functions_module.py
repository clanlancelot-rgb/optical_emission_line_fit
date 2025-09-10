import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import dill as pickle  # replaces the standard pickle module
import pandas as pd
from scipy.special import wofz
from scipy.ndimage import gaussian_filter1d


# Physical constants from astropy
from astropy import constants as const
import astropy.units as u
c_cms = const.c.to('cm/s').value
#c_kms = const.c.to('km/s').value
c_kms = const.c.to('km/s').value   # speed of light in km/s
c_cgs = const.c.cgs
m_e = const.m_e.cgs
e_cgs = const.e.esu
pi = np.pi


from matplotlib.ticker import AutoMinorLocator
import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 2})
import matplotlib as mpl
mpl.rcParams.update({'font.size': 10})

import continuum_function_module as cont_func_mod

import os
current_working_directory = os.getcwd()
code_working_directory = os.path.dirname(os.path.abspath(__file__))

class base_config:
    def __init__(self):
        self.cwd = current_working_directory + '/'
        self.codewd = code_working_directory + '/'
        self.filename = 'config.dat'
        self.print = True
        pass
    # Function to parse the .dat file
    def load_config(self):
        tmp_filename = self.filename
        tmp_print = self.print
        filename = str(self.cwd) + str(tmp_filename)
        assert os.path.exists(filename), f"File: {filename} does not exist!"
        config = {}
        with open(filename, "r") as file:
            for line in file:
                if '#' not in line:
                    key, value = line.strip().split(" = ")
                    # Convert lists
                    if "," in value:
                        values = value.split(", ")
                        try:
                            # Convert to float or keep as string
                            config[key] = np.array([float(v) if "." in v or "e" in v else v for v in values])
                        except ValueError:
                            config[key] = np.array(values)
                    else:
                        # Convert numerical values appropriately
                        try:
                            config[key] = float(value) if "." in value or "e" in value else int(value)
                        except ValueError:
                            config[key] = value  # Keep as string if not a number
        if (tmp_print):
            # Print all keys
            print ('\n')
            print ('Configuration:')
            for key, value in config.items():
                print(f"{key}: {value}")
            print ('\n')
        return config

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def clean_spectrum(wavelength_init, flux_init, flux_init_unc, flux_cont, flux_prof):
    """
    Cleans a spectrum by:
    1. Removing NaN flux values.
    2. Sorting wavelengths in ascending order (adjusting flux accordingly).
    3. Removing duplicate wavelengths (keeping the first occurrence).

    Parameters:
        wavelength (array-like): Array of wavelength values.
        flux (array-like): Array of flux values corresponding to each wavelength.

    Returns:
        tuple: Cleaned wavelength and flux arrays.
    """
    # Convert to numpy arrays
    wavelength1 = np.array(wavelength_init)
    flux1 = np.array(flux_init)
    flux1_unc = np.array(flux_init_unc)
    cont1 = np.array(flux_cont)
    prof1 = np.array(flux_prof)

    # Step 1: Remove NaN flux values
    valid_idx = ~np.isnan(flux1)
    wavelength2, flux2, flux2_unc, cont2, prof2 = wavelength1[valid_idx], flux1[valid_idx], flux1_unc[valid_idx], cont1[valid_idx], prof1[valid_idx]

    # Step 2: Sort by wavelength (monotonically increasing)
    sort_idx = np.argsort(wavelength2)
    wavelength3, flux3, flux3_unc, cont3, prof3 = wavelength2[sort_idx], flux2[sort_idx], flux2_unc[sort_idx], cont2[sort_idx], prof2[sort_idx]

    # Step 3: Remove duplicate wavelengths (keep first occurrence)
    unique_idx = np.unique(wavelength3, return_index=True)[1]
    wavelength4, flux4, flux4_unc, cont4, prof4 = wavelength3[unique_idx], flux3[unique_idx], flux3_unc[unique_idx], cont3[unique_idx], prof3[unique_idx]

    #self.wave, self.flux, self.err, self.cont, self.prof = wavelength4, flux4, flux4_unc, cont4, prof4
    return wavelength4, flux4, flux4_unc, cont4, prof4

#wavelength4, flux4, flux4_unc, cont4, prof4 = clean_spectrum(wavelength_init, flux_init, flux_init_unc, flux_cont=np.ones_like(flux_init), flux_prof=np.zeros_like(flux_init))

def find_nearest(array, value, return_idx=True):
    """
    Find the index of the element in an array that is closest to a given value.

    Parameters:
    array (numpy array or list): The array to search within.
    value (float or int): The target value to find the closest match for.

    Returns:
    int: The index of the nearest element in the array.
    """
    idx1 = int((np.abs(array - value)).argmin())  # Compute absolute differences and find index of minimum difference
    if return_idx:
        return idx1  # Ensure the index is returned as an integer
    else:
        return float(array[idx1])  # Ensure the value is returned as an integer


def get_automated_continuum(orig_wave, orig_flux_noisy):
    cont_func = cont_func_mod.continuum_fitClass()
    cont_func.flux = orig_flux_noisy
    cont_func.default_smooth=50
    cont_func.default_order=30
    cont_func.default_poly_order=3
    cont_func.default_window_size_default=100
    cont_func.default_allowed_percentile=95
    cont_func.default_filter_points_len=100
    cont_func.data_height_upscale=2
    cont_func.default_fwhm_galaxy_min=10
    cont_func.default_noise_level_sigma=10
    cont_func.default_fwhm_ratio_upscale=30
    #cont_func.gaussian_cont_fit = 200
    #cont_func.pca_component_number = 1
    #cont_func.lowess_cont_frac = 0.05
    #cont_func.peak_prominence = 0.05
    #cont_func.peak_width = 500
    #cont_func.median_filter_window = 201
    orig_continuum = cont_func.continuum_finder_flux(orig_wave)
    return (orig_continuum)

#orig_continuum = get_automated_continuum(orig_wave, orig_flux_noisy)

def get_automated_fit(orig_wave, orig_flux_noisy, tot_line_wave, number_of_components_tmp = 1, vel_window_tmp = 2000.):
    tmp_v_shifts = np.array([float(0.0)]*number_of_components_tmp)  # Wind velocity in km/s
    tmp_sig_shifts = np.array([float(100.0)]*number_of_components_tmp)  # Wind velocity in km/s
    tmp_amps = np.array([float(np.nanmedian(orig_flux_noisy)*0.5)]*(number_of_components_tmp*len(tot_line_wave)))
    tmp_prof_to_use = model_amplitudes_only(orig_wave, *tmp_amps, line_wave=tot_line_wave, v_shifts=tmp_v_shifts, sigma_vs=tmp_sig_shifts)
    return (tmp_prof_to_use)

#tmp_prof_to_use = get_automated_fit(orig_wave, orig_flux_noisy, number_of_components_tmp = 1, vel_window_tmp = 2000.)


def save_pickle(filename, *objects):
    """Save any number of Python objects to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(objects, f)
    print(f"Saved to '{filename}'.")

def load_pickle(filename):
    """Load objects from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def ceil_perfect_square(n):
    """
    Returns the smallest perfect square greater than or equal to n.
    """
    if n < 0:
        raise ValueError("Input must be non-negative")
    
    root = np.ceil(np.sqrt(n))
    return int(root ** 1)

def log_to_real_err(log_array, log_error):
    linear_array = 10**log_array
    linear_err = np.abs(linear_array * (np.log(10) * log_error))
    return linear_err

def real_to_log_err(linear_array, linear_error):
    log_array = np.log10(linear_array)
    log_err = np.abs(linear_error / (linear_array * np.log(10)))
    return log_err

def obtain_lines(wave, line_list_file):
    tmp_mask = (np.nanmin(wave)<np.array(line_list_file['wave']).astype(np.float32)) & (np.nanmax(wave)>np.array(line_list_file['wave']).astype(np.float32))
    tmp_line_id = []
    tmp_line_wave = []
    tmp_line_id_rev = []
    for n in range(len(line_list_file['Ion'][tmp_mask])):
        tmp_line_id.append(str(line_list_file['Ion'][tmp_mask][n]).replace(' ', '_'))
        tmp_line_wave.append(float(line_list_file['wave'][tmp_mask][n]))
        tmp_line_id_rev.append(str(line_list_file['Ion'][tmp_mask][n]).replace(' ', '_')+str(int(float(line_list_file['wave'][tmp_mask][n]))))
    return (np.array(tmp_line_id), np.array(tmp_line_wave), np.array(tmp_line_id_rev))

def vel_prof(x, centre):
    xnew = c_kms * ((x-centre)/x)
    return (xnew)

def wavelength_shifted(lambda0, v_shift):
    return lambda0 * (1 + v_shift / c_kms)

def sigma_AA(lambda0, sigma_v):
    return lambda0 * sigma_v / c_kms

def gaussian(wave, amp, mean, stddev):
    return amp * np.exp(-0.5 * ((wave - mean) / stddev)**2)

def wavelength_shifted(lambda0, v_shift):
    return lambda0 * (1 + v_shift / c_kms)

def sigma_AA(lambda0, sigma_v):
    return lambda0 * sigma_v / c_kms

def gaussian(wave, amp, mean, stddev):
    return amp * np.exp(-0.5 * ((wave - mean) / stddev)**2)

def model_curvefit(wave, *theta, line_wave=['6562.819'], magic_number=3):
    N_components = len(theta) // magic_number
    flux_model = np.zeros_like(wave)
    tmp_amp = np.zeros([int(len(line_wave)), magic_number])
    tmp_mean = np.zeros([int(len(line_wave))])
    tmp_std = np.zeros([int(len(line_wave))])
    for o in range(N_components):
        base = o * magic_number
        count=0
        for p in range(len(line_wave)):
            tmp_amp[p,o] = theta[base+p]
            count+=1
        v_shift = theta[base+count]
        sigma_v = theta[base+count+1]
        for q in range(len(line_wave)):
            tmp_mean = wavelength_shifted(line_wave[q], v_shift)
            tmp_std = sigma_AA(line_wave[q], sigma_v)
            flux_model += (gaussian(wave, tmp_amp[q,o], tmp_mean, tmp_std))
    return flux_model

# New model function: only amp is fitted; mean and stddev are passed in
def model_amplitudes_only(wave, *amps, line_wave=None, v_shifts=None, sigma_vs=None):
    """
    Args:
        wave: wavelength array
        amps: amplitudes (length = N_components * N_lines)
        line_wave: list of rest-frame line centers (e.g., [Hα, [NII]1, [NII]2])
        v_shifts: list of velocity shifts (km/s), length = N_components
        sigma_vs: list of velocity dispersions (km/s), length = N_components
    """
    assert line_wave is not None and v_shifts is not None and sigma_vs is not None
    N_components = len(v_shifts)
    N_lines = len(line_wave)
    #print (len(amps), N_components * N_lines)
    assert len(amps) == N_components * N_lines

    flux_model = np.zeros_like(wave)

    for i in range(N_components):
        for j in range(N_lines):
            idx = i * N_lines + j
            amp = amps[idx]
            mu = wavelength_shifted(line_wave[j], v_shifts[i])
            std = sigma_AA(line_wave[j], sigma_vs[i])
            flux_model += gaussian(wave, amp, mu, std)

    return flux_model



def delta_lambda_kms(lambda0, delta_v_kms):
    return lambda0 * delta_v_kms / c_kms

def integrated_flux(wave, flux_noisy, flux_err, lambda_Ha, delta_v_kms = 400):
    cust_vel_prof = vel_prof(wave, lambda_Ha)
    mask_Ha = (cust_vel_prof >= -delta_v_kms) & (cust_vel_prof <= delta_v_kms)
    mask_Ha2 = (cust_vel_prof >= 500) & (cust_vel_prof <= 700)

    #delta_wave = np.diff(cust_vel_prof).mean()  # assume constant pixel size
    delta_wave = np.diff(wave[mask_Ha]).mean()  # assume constant pixel size
    #flux_integrated_Ha = np.trapz(flux_noisy[mask_Ha] - np.median(flux_noisy[~mask_Ha]), cust_vel_prof[mask_Ha])
    #print (np.array(wave)[mask_Ha])
    flux_integrated_Ha = np.nansum(flux_noisy[mask_Ha] - np.nanmedian(flux_noisy[mask_Ha2])) * delta_wave
    flux_unc_integrated_Ha = np.sqrt(np.nansum(flux_err[mask_Ha]**2)) * delta_wave
    
    return (flux_integrated_Ha, flux_unc_integrated_Ha)


def integrated_flux_fit(wave, amplitude, amplitude_unc, sigma_kms, sigma_kms_unc, lambda_Ha, delta_v_kms = 400):
    cust_vel_prof = vel_prof(wave, lambda_Ha)
    mask_Ha = (cust_vel_prof >= -delta_v_kms) & (cust_vel_prof <= delta_v_kms)
    wave = wave[mask_Ha]

    delta_lambda = (sigma_kms/(c_kms))*lambda_Ha
    delta_lambda_unc = (sigma_kms_unc/(c_kms))*lambda_Ha
    delta_wave = np.diff(wave).mean()  # assume constant pixel size
    #flux_integrated_Ha = np.nansum(gaussian(wave, amplitude, lambda_Ha, delta_lambda))*delta_wave
    tmp_ar_flux_integrated_Ha = []
    for i in range(100):
        ar_amp = np.random.normal(amplitude, amplitude_unc)
        ar_del_lambda = np.random.normal(delta_lambda, delta_lambda_unc)
        tmp_ar_flux_integrated_Ha.append(np.nansum(gaussian(wave, ar_amp, lambda_Ha, ar_del_lambda))*delta_wave)
    #ar_flux_integrated_Ha =
    flux_integrated_Ha = np.nanmean(tmp_ar_flux_integrated_Ha, axis=0)
    flux_unc_integrated_Ha = np.nanstd(tmp_ar_flux_integrated_Ha, axis=0)
    return (flux_integrated_Ha, flux_unc_integrated_Ha)



def print_results(component_info, best_N, tot_original_flux=None, line_id=['Ha'], line_wave=['6562.819']):
    for z in range(best_N):
        print(f"Component {z+1}:")
        print (f"Velocity = {component_info[f'vel_{z}']:.3e} +/- {component_info[f'vel_unc_{z}']:.3e}")
        print (f"Dispersion = {component_info[f'sigma_{z}']:.3e} +/- {component_info[f'sigma_unc_{z}']:.3e}")
        for a in range(len(line_id)):
            line_id_rev = str(line_id[a])+str(int(line_wave[a]))
            tmp_line_flux, tmp_line_flux_unc = component_info[f'flux_{line_id_rev}_{z}'], component_info[f'flux_unc_{line_id_rev}_{z}']
            print(f"{line_id_rev} flux = {tmp_line_flux:.3e} +/- {tmp_line_flux_unc:.3e}")

    for b in range(len(line_id)):
        line_id_rev = str(line_id[b])+str(int(line_wave[b]))
        tmp_tot_flux, tmp_tot_unc_flux = component_info[f'tot_flux_{line_id_rev}'], component_info[f'tot_flux_unc_{line_id_rev}']
        if tot_original_flux is not None:
            print(f"Total Original Flux ({line_id_rev}) =  {tot_original_flux[b]:.3e}")
        print(f"Total Fitted Flux ({line_id_rev}) =  {tmp_tot_flux:.3e} +/- {tmp_tot_unc_flux:.3e}")
        tmp_integrated_flux, tmp_integrated_unc_flux = component_info[f'integrated_flux_{line_id_rev}'], component_info[f'integrated_flux_unc_{line_id_rev}']
        print(f"Total Integrated Flux ({line_id_rev}) =  {tmp_integrated_flux:.3e} +/- {tmp_integrated_unc_flux:.3e}")
    return None

#_ = print_results(component_info, best_N, line_id, line_wave)

def print_results_rev(component_info, best_N, tot_original_flux=None, line_id=['Ha'], line_wave=['6562.819']):
    import numpy as np

    n_comp = best_N

    # Clean line names (remove underscore)
    line_names = [f"{lid.replace('_', '')}{int(lw)}" for lid, lw in zip(line_id, line_wave)]
    original_keys = [f"{lid}{int(lw)}" for lid, lw in zip(line_id, line_wave)]

    def format_flux(val, unc, threshold=1e-3):
        """Format flux with cutoff for very low values"""
        if abs(val) < threshold and abs(unc) < threshold:
            return "0.0"
        else:
            return f"${val:.2e} \\pm {unc:.2e}$"

    # Prepare two tables: first 3 comps, last 2 comps + total
    rows_1 = []
    rows_2 = []

    # Velocity row
    vel_row_1 = ["Velocity (km/s)"]
    vel_row_2 = ["Velocity (km/s)"]
    for z in range(min(3, n_comp)):
        v = component_info[f"vel_{z}"]
        v_unc = component_info[f"vel_unc_{z}"]
        vel_row_1.append(f"${v:.1f} \\pm {v_unc:.1f}$")
    for z in range(3, n_comp):
        v = component_info[f"vel_{z}"]
        v_unc = component_info[f"vel_unc_{z}"]
        vel_row_2.append(f"${v:.1f} \\pm {v_unc:.1f}$")
    vel_row_2.append("")  # No total for velocity
    rows_1.append(vel_row_1)
    rows_2.append(vel_row_2)

    # Dispersion row
    sigma_row_1 = ["Dispersion (km/s)"]
    sigma_row_2 = ["Dispersion (km/s)"]
    for z in range(min(3, n_comp)):
        s = component_info[f"sigma_{z}"]
        s_unc = component_info[f"sigma_unc_{z}"]
        sigma_row_1.append(f"${s:.1f} \\pm {s_unc:.1f}$")
    for z in range(3, n_comp):
        s = component_info[f"sigma_{z}"]
        s_unc = component_info[f"sigma_unc_{z}"]
        sigma_row_2.append(f"${s:.1f} \\pm {s_unc:.1f}$")
    sigma_row_2.append("")
    rows_1.append(sigma_row_1)
    rows_2.append(sigma_row_2)

    # Flux rows
    for line_clean, line_orig in zip(line_names, original_keys):
        flux_row_1 = [f"F\_{line_clean}"]
        flux_row_2 = [f"F\_{line_clean}"]
        for z in range(min(3, n_comp)):
            f = component_info.get(f"flux_{line_orig}_{z}", np.nan)
            f_unc = component_info.get(f"flux_unc_{line_orig}_{z}", np.nan)
            flux_row_1.append(format_flux(f, f_unc))
        for z in range(3, n_comp):
            f = component_info.get(f"flux_{line_orig}_{z}", np.nan)
            f_unc = component_info.get(f"flux_unc_{line_orig}_{z}", np.nan)
            flux_row_2.append(format_flux(f, f_unc))
        total_flux = component_info.get(f"tot_flux_{line_orig}", np.nan)
        total_unc = component_info.get(f"tot_flux_unc_{line_orig}", np.nan)
        flux_row_2.append(format_flux(total_flux, total_unc))
        rows_1.append(flux_row_1)
        rows_2.append(flux_row_2)

    # Generate LaTeX Table 1
    latex1 = "\\begin{table*}[ht]\n\\centering\n"
    latex1 += "\\begin{tabular}{l" + "c" * min(3, n_comp) + "}\n"
    header_1 = ["Line/Param"] + [f"Comp {i+1}" for i in range(min(3, n_comp))]
    latex1 += " & ".join(header_1) + " \\\\\n\\hline\n"
    for row in rows_1:
        latex1 += " & ".join(row) + " \\\\\n"
    latex1 += "\\end{tabular}\n"
    #latex1 += "\\caption{Fitted Parameters and Fluxes (Components 1--3)}\n"
    latex1 += "\caption{Component-wise (Components 1--3) emission line fitted results for ionized (HII region) gas around the star-cluster, cluster\_1, in M83. First two rows show velocity, v and velocity dispersion, sigma, followed by component wise flux (in the units of 1e-16 erg/s/cm2) for each observed line.}\n"
    latex1 += "\\end{table*}\n"

    # Generate LaTeX Table 2
    latex2 = "\\begin{table*}[ht]\n\\centering\n"
    latex2 += "\\begin{tabular}{l" + "c" * (max(0, n_comp - 3) + 1) + "}\n"
    header_2 = ["Line/Param"] + [f"Comp {i+1}" for i in range(3, n_comp)] + ["Total Flux"]
    latex2 += " & ".join(header_2) + " \\\\\n\\hline\n"
    for row in rows_2:
        latex2 += " & ".join(row) + " \\\\\n"
    latex2 += "\\end{tabular}\n"
    #latex2 += "\\caption{Fitted Parameters and Fluxes (Components 4+, Total)}\n"
    latex2 += "\caption{Component-wise (Components 3--total) emission line fitted results for ionized (HII region) gas around the star-cluster, cluster\_1, in M83. First two rows show velocity, v and velocity dispersion, sigma, followed by component wise flux (in the units of 1e-16 erg/s/cm2) for each observed line.}\n"
    latex2 += "\\end{table*}"

    print ('\n\n')
    print(latex1)
    print("\n" + "="*80 + "\n")
    print(latex2)
    print ('\n\n')

    return latex1 + "\n\n" + latex2



def print_results_rev_components_only_single_table(component_info, best_N, line_id=['Ha'], line_wave=['6562.819']):
    import numpy as np

    n_comp = best_N

    # Clean line names (remove underscores)
    line_names = [f"{lid.replace('_', '')}{int(lw)}" for lid, lw in zip(line_id, line_wave)]
    original_keys = [f"{lid}{int(lw)}" for lid, lw in zip(line_id, line_wave)]

    def format_flux(val, unc, threshold=1e-3):
        if abs(val) < threshold and abs(unc) < threshold:
            return "$0.0 \\pm 0.0$"
        else:
            return f"${val:.2e} \\pm {unc:.2e}$"

    rows = []

    # Velocity row
    vel_row = ["Velocity (km/s)"]
    for z in range(n_comp):
        v = component_info[f"vel_{z}"]
        v_unc = component_info[f"vel_unc_{z}"]
        vel_row.append(f"${v:.1f} \\pm {v_unc:.1f}$")
    rows.append(vel_row)

    # Dispersion row
    sigma_row = ["Dispersion (km/s)"]
    for z in range(n_comp):
        s = component_info[f"sigma_{z}"]
        s_unc = component_info[f"sigma_unc_{z}"]
        sigma_row.append(f"${s:.1f} \\pm {s_unc:.1f}$")
    rows.append(sigma_row)

    # Flux rows
    for line_clean, line_orig in zip(line_names, original_keys):
        flux_row = [f"Flux ({line_clean})"]
        for z in range(n_comp):
            f = component_info.get(f"flux_{line_orig}_{z}", np.nan)
            f_unc = component_info.get(f"flux_unc_{line_orig}_{z}", np.nan)
            flux_row.append(format_flux(f, f_unc))
        rows.append(flux_row)

    # Generate LaTeX table
    latex = "\\begin{table}[ht]\n\\centering\n"
    latex += "\\begin{tabular}{l" + "c" * n_comp + "}\n"
    header = ["Line/Param"] + [f"Comp {i+1}" for i in range(n_comp)]
    latex += " & ".join(header) + " \\\\\n\\hline\n"
    for row in rows:
        latex += " & ".join(row) + " \\\\\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Fitted Parameters and Fluxes (Components only)}\n"
    latex += "\\end{table}"

    print(latex)
    return latex



def checking_results_plot(component_info, tot_original_flux=None, line_id=['Ha'], line_wave=['6562.819'], strong_line_id=['Ha'], figname='checking_results.pdf'):
    fig3 = plt.figure(figsize=(10,8))
    x_line = []
    #tot_original_flux,
    for c in range(len(line_id)):
        if tot_original_flux is not None:
            x_line.append(tot_original_flux[c])
            line_id_rev = str(line_id[c])+str(int(line_wave[c]))
            plt.errorbar(tot_original_flux[c], component_info[f'tot_flux_{line_id_rev}'], xerr=5e-16, yerr=component_info[f'tot_flux_unc_{line_id_rev}'], fmt='*', label='Fit', zorder=2)
            plt.errorbar(tot_original_flux[c], component_info[f'integrated_flux_{line_id_rev}'], xerr=5e-16, yerr=component_info[f'integrated_flux_unc_{line_id_rev}'], fmt='o', label='Integrated', zorder=1)
            plt.xlabel('Original Flux')
            plt.ylabel('Derived Flux')

        else:
            line_id_rev = str(line_id[c])+str(int(line_wave[c]))
            x_line.append(float(component_info[f'tot_flux_{line_id_rev}']))
            if (line_id[c] in strong_line_id):
                plt.errorbar(component_info[f'tot_flux_{line_id_rev}'], component_info[f'integrated_flux_{line_id_rev}'], xerr=component_info[f'tot_flux_unc_{line_id_rev}'], yerr=component_info[f'integrated_flux_unc_{line_id_rev}'], fmt='*', label=f'{line_id_rev}', zorder=4)
            else:
                plt.errorbar(component_info[f'tot_flux_{line_id_rev}'], component_info[f'integrated_flux_{line_id_rev}'], xerr=component_info[f'tot_flux_unc_{line_id_rev}'], yerr=component_info[f'integrated_flux_unc_{line_id_rev}'], fmt='o', label=f'{line_id_rev}', zorder=3)

            plt.xlabel('Fitted Flux')
            plt.ylabel('Integrated Flux')

    line = np.linspace(np.nanmin(x_line)-10, np.nanmax(x_line)+10, 100)
    plt.plot(line, line, label='1:1', color='green',ls='dashed', zorder=2)
    leg = plt.legend(loc='best', ncol=5)
    leg.set_zorder(0)
    #plt.xlim(np.nanmin(line), np.nanmax(line))
    #plt.ylim(np.nanmin(line), np.nanmax(line))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.show()
    plt.savefig(figname, dpi=100)
    print (f"{figname} saved")
    plt.close(fig3)
    #for all_figs in plt.get_fignums():
    #    plt.close(all_figs)
    return None

#_ = checking_results_plot(component_info, tot_original_flux=None, line_id=['Ha'], line_wave=['6562.819'])

def obtain_fittable_area(xdata, ydata, ydata_err, contdata, species_wave_array, allowable_velocity_range=1000.0):
    fittable_wave = np.array([])
    fittable_flux = np.array([])
    fittable_flux_err = np.array([])
    fittable_cont = np.array([])
    for i in range(len(species_wave_array)):
        tmp_vel_array = vel_prof(xdata, species_wave_array[i])
        tmp_mask = (-allowable_velocity_range < tmp_vel_array) & (tmp_vel_array < allowable_velocity_range)
        fittable_wave = np.append(fittable_wave, xdata[tmp_mask])
        fittable_flux = np.append(fittable_flux, ydata[tmp_mask])
        fittable_flux_err = np.append(fittable_flux_err, ydata_err[tmp_mask])
        fittable_cont = np.append(fittable_cont, contdata[tmp_mask])

    # Combine arrays and sort by wavelength
    sorted_indices = np.argsort(fittable_wave)
    wave_sorted = fittable_wave[sorted_indices]
    flux_sorted = fittable_flux[sorted_indices]
    flux_unc_sorted = fittable_flux_err[sorted_indices]
    cont_sorted = fittable_cont[sorted_indices]

    # Remove duplicates based on wavelength
    unique_wave, unique_indices = np.unique(wave_sorted, return_index=True)
    wave_final = wave_sorted[unique_indices]
    flux_final = flux_sorted[unique_indices]
    flux_unc_final = flux_unc_sorted[unique_indices]
    cont_final = cont_sorted[unique_indices]

    return (wave_final, flux_final, flux_unc_final, cont_final)

#wave_final, flux_final, flux_unc_final, cont_final = obtain_fittable_area(xdata, ydata, ydata_err, contdata, species_wave_array, allowable_velocity_range=1000.0)













####ABSORPTION LINE RELATED FUNCTIONS

def get_abs_atomic_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip comments and empty lines
            tokens = line.split()
            # Find index of ion_wave (first numeric token)
            wave_idx = None
            for i, tok in enumerate(tokens):
                try:
                    float(tok.replace("E", "e"))
                    wave_idx = i
                    break
                except ValueError:
                    continue
            if wave_idx is None or len(tokens) < wave_idx + 3:
                continue  # skip malformed lines
            ion_name    = " ".join(tokens[:wave_idx])
            ion_wave    = tokens[wave_idx]
            ion_osc_str = tokens[wave_idx + 1]
            ion_tau     = tokens[wave_idx + 2]
            comments    = " ".join(tokens[wave_idx + 3:]) if len(tokens) > wave_idx + 3 else ""
            try:
                ion_id = f"{ion_name.replace(' ', '')}{int(float(ion_wave))}"
            except ValueError:
                ion_id = f"{ion_name} NA"
            data.append({
                "ion_id": ion_id,
                "ion_name": ion_name.replace(' ', ''),
                "ion_wave": ion_wave,
                "ion_osc_str": ion_osc_str,
                "ion_tau": ion_tau,
                "comments": comments
            })
    df = pd.DataFrame(data)
    #print(df.head())
    #print (df['ion_id'])
    ion_id, ion_name, ion_wave, ion_osc_str, ion_tau, ion_comments = df['ion_id'].to_numpy().astype(np.str_), df['ion_name'].to_numpy().astype(np.str_), df['ion_wave'].to_numpy().astype(np.float32), df['ion_osc_str'].to_numpy().astype(np.float32), df['ion_tau'].to_numpy().astype(np.float32), df['comments'].to_numpy().astype(np.str_)
    #print (ion_id)
    tmp_unique_ions = np.unique(ion_name, return_index=True)
    unique_ions = tmp_unique_ions[0][np.argsort(tmp_unique_ions[1])]
    return (ion_id, ion_name, ion_wave, ion_osc_str, ion_tau, ion_comments, unique_ions)


#ion_id, ion_name, ion_wave, ion_osc_str, ion_tau, ion_comments = get_abs_atomic_data(file_path)

def obtain_lines_abs(wave, ion_id, ion_name, ion_wave, ion_osc_str, ion_tau, unique_ions, species_list=None):
    tmp_mask = (np.nanmin(wave)<np.array(ion_wave).astype(np.float32)) & (np.nanmax(wave)>np.array(ion_wave).astype(np.float32))
 
    tmp_line_id1 = np.array(ion_name)[tmp_mask]
    tmp_line_wave1 = np.array(ion_wave)[tmp_mask]
    tmp_line_osc_str1 = np.array(ion_osc_str)[tmp_mask]
    tmp_line_tau1 = np.array(ion_tau)[tmp_mask]
    tmp_line_id_rev1 = np.array(ion_id)[tmp_mask]
    tmp_unique_ions1 = np.array(unique_ions)

    if species_list is not None:
        tmp_mask2 = np.zeros(len(tmp_line_id1), dtype=bool)
        tmp_mask3 = np.zeros(len(tmp_unique_ions1), dtype=bool)
        for ion in species_list:
            tmp_mask2 |= np.char.startswith(tmp_line_id1, ion)
            tmp_mask3 |= np.char.startswith(tmp_unique_ions1, ion)

    else:
        tmp_mask2 = np.ones(len(tmp_line_id1), dtype=bool)
        tmp_mask3 = np.ones(len(tmp_unique_ions1), dtype=bool)

    tmp_line_id2 = tmp_line_id1[tmp_mask2]
    tmp_line_wave2 = tmp_line_wave1[tmp_mask2]
    tmp_line_osc_str2 = tmp_line_osc_str1[tmp_mask2]
    tmp_line_tau2 = tmp_line_tau1[tmp_mask2]
    tmp_line_id_rev2 = tmp_line_id_rev1[tmp_mask2]
    tmp_unique_ions3 = tmp_unique_ions1[tmp_mask3]

    return (tmp_line_id2, tmp_line_wave2, tmp_line_id_rev2, tmp_line_osc_str2, tmp_line_tau2, tmp_unique_ions3)

#orig_line_id, orig_line_wave, orig_line_id_rev, orig_line_osc_str, orig_line_tau = obtain_lines_abs(wave, ion_id, ion_name, ion_wave, ion_osc_str, ion_tau)



# Voigt absorption profile

def H(a, x):
    P = x**2
    H0 = np.exp(-x**2)
    Q = 1.5/x**2
    return H0 - a/np.sqrt(np.pi)/P * (H0*H0*(4.*P*P + 7.*P + 4. + Q) - Q - 1)

def Voigt3_H2(l, l0, f, N, b, gam):
    """Calculate the Voigt profile of transition with
    rest frame transition wavelength: 'l0'
    oscillator strength: 'f'
    column density: N  cm^-2
    velocity width: b  cm/s
    """
    # ==== PARAMETERS ==================
    c = 2.99792e10        # cm/s
    m_e = 9.1095e-28       # g
    e = 4.8032e-10        # cgs units
    # ==================================
    # Calculate Profile
    C_a = np.sqrt(np.pi)*e**2*f*l0*1.e-8/m_e/c/b
    a = l0*1.e-8*gam/(4.*np.pi*b)
    dl_D = b/c*l0
    #l = l/(z+1.)
    x = (l - l0)/dl_D+0.0001
    tau = np.float64(C_a) * N * H(a, x)
    return (tau)

#tau = Voigt3_H2(l, l0, f, N, b, gam)

def group_voigt_arrays(wave_array, ions, unique_ions, transitions, l0_array, f_array, gam_array, col_den, b_val, vel, fwhm_kms=50):

    # Force to NumPy arrays to avoid boolean indexing errors
    ions = np.array(ions)
    l0_array = np.array(l0_array, dtype=float)
    f_array = np.array(f_array, dtype=float)
    gam_array = np.array(gam_array, dtype=float)
    col_den = np.array(col_den, dtype=float)
    b_val = np.array(b_val, dtype=float)
    vel = np.array(vel, dtype=float)
    tau_total = np.zeros_like(wave_array, dtype=float)
    n_comp = len(b_val)
    n_ions = len(unique_ions)

    count=0
    for i in range(len(unique_ions)):
        tmp_mask = (unique_ions[i]==ions)
        for j in range(len(b_val)):
            for k in range(len(ions[tmp_mask])):
                l0_shifted = l0_array[tmp_mask][k] * (1. + vel[j] / c_kms)
                tmp_col_den = 10**(col_den[count])
                tau_total += Voigt3_H2(wave_array, l0_shifted, f_array[tmp_mask][k], tmp_col_den, b_val[j]*1e5, gam_array[tmp_mask][k])
            count+=1

    # Convert tau to flux
    flux = np.exp(-tau_total)
    # Convolve with Gaussian FWHM in km/s
    fwhm_lambda = (fwhm_kms / c_kms) * np.median(wave_array)
    sigma_lambda = fwhm_lambda / (2 * np.sqrt(2 * np.log(2)))
    sigma_pix = sigma_lambda / np.median(np.diff(wave_array))
    flux_conv = gaussian_filter1d(flux, sigma_pix, mode='nearest')
    return flux_conv

#flux_convolved = group_voigt_arrays(wave_array, ions, unique_ions, transitions, l0_array, f_array, gam_array, col_den, b_val, vel, fwhm_kms=50)


def abs_model_for_curvefit(wave, *theta, ions=['HI'], unique_ions=['HI'], transitions=['HI1215'], l0_array=[1215.6701], f_array=[0.416400], gam_array=[6.265E8], ncomp=3, fwhm_kms=50):
    b_val = list(theta[0:ncomp])
    vel = list(theta[ncomp:2*ncomp])
    col_den = list(theta[2*ncomp:])
    prof_conv = group_voigt_arrays(wave, ions, unique_ions, transitions, l0_array, f_array, gam_array, col_den, b_val, vel, fwhm_kms=fwhm_kms)
    return prof_conv



def abs_model_for_curvefit_fixed(wave, *theta, ions=['HI'], unique_ions=['HI'], transitions=['HI1215'], l0_array=[1215.6701], f_array=[0.416400], gam_array=[6.265E8], ncomp=3, fwhm_kms=50, b_val=[20.0, 20.0, 20.0], vel=[0.0, 0.0, 0.0]):
    #b_val = list(theta[0:ncomp])
    #vel = list(theta[ncomp:2*ncomp])
    col_den = list(theta)
    prof_conv = group_voigt_arrays(wave, ions, unique_ions, transitions, l0_array, f_array, gam_array, col_den, b_val, vel, fwhm_kms=fwhm_kms)
    return prof_conv






# Saving absorption data

def total_logN(logN_array, logN_unc_array, rounding_val=2):
    """
    Combine multiple log10 column densities with uncertainties.

    Parameters
    ----------
    logN_array : array-like
        Column densities in log10 space (dex).
    logN_unc_array : array-like
        1-sigma uncertainties in dex.

    Returns
    -------
    total_logN : float
        Total column density in log10 space (dex).
    total_logN_unc : float
        1-sigma uncertainty of total column density in dex.
    """
    logN_array = np.asarray(logN_array, dtype=float)
    logN_unc_array = np.asarray(logN_unc_array, dtype=float)

    # Convert to linear scale
    N_linear = 10**logN_array
    N_linear_unc = N_linear * np.log(10) * logN_unc_array

    # Total in linear space
    N_total_linear = np.nansum(N_linear)
    N_total_linear_unc = np.sqrt(np.nansum(N_linear_unc**2))

    # Convert back to log space
    total_logN = np.log10(N_total_linear)
    total_logN_unc = N_total_linear_unc / (N_total_linear * np.log(10))

    return np.round(total_logN, rounding_val), np.round(total_logN_unc, rounding_val)

#total_log_N, total_log_N_unc = total_logN(col_den_rev, np.zeros_like(col_den_rev))

def save_information(col_den, unique_ions, b_val, vel, col_den_unc, b_val_unc, vel_unc, tmp_bic_file='abs_result.pkl'):
    tmp_objects = [col_den, unique_ions, b_val, vel, col_den_unc, b_val_unc, vel_unc]
    save_pickle(tmp_bic_file, *tmp_objects)
    print (f"{tmp_bic_file} saved")
    return None

#_ = save_information(col_den, unique_ions, b_val, vel, tmp_bic_file='test.pkl')


def retrieve_information(col_den, unique_ions, b_val, vel, tmp_bic_file='abs_result.pkl', print_bool=True, col_den_unc=None, b_val_unc=None, vel_unc=None):
    if os.path.exists(tmp_bic_file):
        user_input = input(f"'{tmp_bic_file}' already exists. Load previous results? (y/n): ").strip().lower()
        if user_input == 'y':
            col_den, unique_ions, b_val, vel, col_den_unc, b_val_unc, vel_unc = load_pickle(tmp_bic_file)
        else:
            col_den, unique_ions, b_val, vel, col_den_unc, b_val_unc, vel_unc = col_den, unique_ions, b_val, vel, col_den_unc, b_val_unc, vel_unc
    else:
        col_den, unique_ions, b_val, vel, col_den_unc, b_val_unc, vel_unc = col_den, unique_ions, b_val, vel, col_den_unc, b_val_unc, vel_unc

    col_den_rev = np.reshape(col_den, (len(unique_ions), len(b_val)))
    if (col_den_unc is not None):
        col_den_unc_rev = np.reshape(col_den_unc, (len(unique_ions), len(b_val)))

    for j in range(len(b_val)):
        print (f"Component:{j+1}")
        print (f"Relative Velocity:{np.round(vel[j], 2)}+/-{np.round(vel_unc[j], 2)}")
        print (f"Velocity dispersion:{np.round(b_val[j], 2)}+/-{np.round(b_val_unc[j], 2)}")
        for i in range(len(unique_ions)):
            print (f"{unique_ions[i]}: {np.round(col_den_rev[i,j], 2)}+/-{np.round(col_den_rev[i,j], 2)}")

        tmp_h2, tmp_h2_unc = [], []
        print ("Total column densities:")
        for i in range(len(unique_ions)):
            if (col_den_unc is not None):
                total_log_N, total_log_N_unc = total_logN(col_den_rev[i,:], col_den_unc_rev[i,:])
            else:
                total_log_N, total_log_N_unc = total_logN(col_den_rev[i,:], np.zeros_like(col_den_rev[i,:]))
            print (f"{unique_ions[i]}: {np.round(total_log_N, 2)}+/-{np.round(total_log_N_unc, 2)}")
            tmp_h2.append(total_log_N)
            tmp_h2_unc.append(total_log_N_unc)

        print (tmp_h2, tmp_h2_unc)
        total_h2_log_N, total_h2_log_N_unc = total_logN(tmp_h2, tmp_h2_unc)
        print (f"Total H2 column densities:{np.round(total_h2_log_N, 2)}+/-{np.round(total_h2_log_N_unc, 2)}")
        
    '''
    user_input2 = input(f"Would you like to save results? (y/n):")
    if user_input2 == 'y':
        save_information(col_den, unique_ions, b_val, vel, col_den_unc, b_val_unc, vel_unc, tmp_bic_file=tmp_bic_file)
    '''
    return None

#_ = retrieve_information(col_den, unique_ions, b_val, vel, tmp_bic_file='test.pkl', print_bool=True)



