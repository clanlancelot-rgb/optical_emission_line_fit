import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from astropy import constants as const
c_cms = const.c.to('cm/s').value
c_kms = const.c.to('km/s').value

from matplotlib.ticker import AutoMinorLocator
import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 2})
import matplotlib as mpl
mpl.rcParams.update({'font.size': 10})

import basic_functions_module as base_func

def curvefit_method(wave, flux_noisy, flux_err, n_comp, line_wave=['6562.819'], v_init=0.0, v_lower=-250.0, v_upper=250.0, sigma_init=80.0, sigma_lower=10.0, sigma_upper=1500.0):

    magic_number = len(line_wave)+2

    # Wrap the function
    wrapped_model = lambda wave, *theta: base_func.model_curvefit(
        wave, *theta, line_wave=line_wave, magic_number=magic_number)

    print(f"\n=== Trying N_components = {n_comp} ===")
    ndim = magic_number * n_comp
    # Initial guess
    p0 = []
    lower_bounds = []
    upper_bounds = []
    for w in range(n_comp):
        for x in range(len(line_wave)):
            p0 += [0.0]
            lower_bounds += [0.0]
            upper_bounds += [np.nanmax(flux_noisy)]
        p0 += [v_init, sigma_init]
        lower_bounds += [v_lower, sigma_lower]
        upper_bounds += [v_upper, sigma_upper]
    try:
        popt, pcov = curve_fit(wrapped_model, wave, flux_noisy, sigma=flux_err, absolute_sigma=True, p0=p0, bounds=(lower_bounds, upper_bounds))
        perr = np.sqrt(np.diag(pcov))
        model_flux = wrapped_model(wave, *popt)
        residuals = flux_noisy - wrapped_model(wave, *popt)
        chi2 = np.sum((residuals / flux_err) ** 2)
        print(f"Chi² = {chi2:.2f}")
        return (popt, perr)
    except Exception as e:
        print(f"Fit failed for N_components = {n_comp}: {e}")
        return (None, None)

#popt, perr = curvefit_method(wave, flux_noisy, flux_err, n_comp, line_wave=['6562.819'], v_init=0.0, v_lower=-250.0, v_upper=250.0, sigma_init=80.0, sigma_lower=10.0, sigma_upper=1500.0)

def curvefit_method_abs(wave, flux, flux_err, ions=['HI'], unique_ions=['HI'], transitions=['HI1215'], l0_array=[1215.6701], f_array=[0.416400], gam_array=[6.265E8], ncomp=1, fwhm_kms=50, col_den_lower=10.0, col_den_upper=20.0, col_den_init=14.0, vel_init=0.0, sigma_init=30.0, v_lower=-500.0, v_upper=500.0, sigma_lower=1.0, sigma_upper=500.0):

    # Wrap the function
    wrapped_model_abs = lambda wave, *theta: base_func.abs_model_for_curvefit(wave, *theta, ions=ions, unique_ions=unique_ions, transitions=transitions, l0_array=l0_array, f_array=f_array, gam_array=gam_array, ncomp=ncomp, fwhm_kms=fwhm_kms)

    print(f"\n=== Trying N_components = {ncomp} ===")

    v_init_list = list(np.full([ncomp], fill_value=vel_init))
    v_lower_list = list(np.full([ncomp], fill_value=v_lower))
    v_upper_list = list(np.full([ncomp], fill_value=v_upper))

    sigma_init_list = list(np.full([ncomp], fill_value=sigma_init))
    sigma_lower_list = list(np.full([ncomp], fill_value=sigma_lower))
    sigma_upper_list = list(np.full([ncomp], fill_value=sigma_upper))

    p0.append(v_init_list)
    p0.append(sigma_init_list)
    lower_bounds.append(v_lower_list)
    lower_bounds.append(sigma_lower_list)
    upper_bounds.append(v_upper_list)
    upper_bounds.append(sigma_upper_list)

    col_den_init_list = list(np.full([ncomp*len(unique_ions)], fill_value=col_den_init))
    col_den_lower_list = list(np.full([ncomp*len(unique_ions)], fill_value=col_den_lower))
    col_den_upper_list = list(np.full([ncomp*len(unique_ions)], fill_value=col_den_upper))

    p0.append(col_den_init_list)
    lower_bounds.append(col_den_lower_list)
    upper_bounds.append(col_den_upper_list)

    try:
        popt, pcov = curve_fit(wrapped_model_abs, wave, flux, sigma=flux_err, absolute_sigma=True, p0=p0, bounds=(lower_bounds, upper_bounds))
        perr = np.sqrt(np.diag(pcov))
        model_flux = wrapped_model_abs(wave, *popt)
        residuals = flux - wrapped_model_abs(wave, *popt)
        chi2 = np.sum((residuals / flux_err) ** 2)
        print(f"Chi² = {chi2:.2f}")
        return (popt, perr)
    except Exception as e:
        print(f"Fit failed for N_components = {n_comp}: {e}")
        return (None, None)

#popt, perr = curvefit_method_abs(wave, flux, flux_err, ions=['HI'], unique_ions=['HI'], transitions=['HI1215'], l0_array=[1215.6701], f_array=[0.416400], gam_array=[6.265E8], ncomp=1, fwhm_kms=50, col_den_lower=10.0, col_den_upper=20.0, col_den_init=14.0, vel_init=0.0, sigma_init=30.0, v_lower=-500.0, v_upper=500.0, sigma_lower=1.0, sigma_upper=500.0)



def get_vel_sigma_from_popt(theta, n_comp, line_wave):
    tmp_vel, tmp_sigma = [], []
    magic_number = len(line_wave)+2
    for o in range(n_comp):
        base = o * magic_number
        count=0
        for p in range(len(line_wave)):
            count+=1
        tmp_vel.append(float(theta[base+count]))
        tmp_sigma.append(float(theta[base+count+1]))
    return (tmp_vel, tmp_sigma)



def curvefit_amp_method(orig_wave, orig_flux_noisy, orig_flux_err_noisy, tmp_v_shifts, tmp_sig_shifts, weak_line_wave, number_of_components_tmp, amp_val_init=0.0, amp_val_min=0.0, amp_val_max=100.0):
    p0 = np.array([float(amp_val_init)]*(number_of_components_tmp*len(weak_line_wave)))
    lower_bounds = np.full_like(p0, fill_value=amp_val_min)
    upper_bounds = np.full_like(p0, fill_value=amp_val_max)
    # Wrapped model for curve_fit
    wrapped_model_amplitudes_curvefit = lambda wave, *amp: base_func.model_amplitudes_only(wave, *amp, line_wave=weak_line_wave, v_shifts=tmp_v_shifts, sigma_vs=tmp_sig_shifts)
    try:
        popt, pcov = curve_fit(wrapped_model_amplitudes_curvefit, orig_wave, orig_flux_noisy, sigma=orig_flux_err_noisy, absolute_sigma=True, p0=p0, bounds=(lower_bounds, upper_bounds))
        perr = np.sqrt(np.diag(pcov))
        model_flux = wrapped_model_amplitudes_curvefit(orig_wave, *popt)
        residuals = orig_flux_noisy - wrapped_model_amplitudes_curvefit(orig_wave, *popt)
        chi2 = np.sum((residuals / orig_flux_err_noisy) ** 2)
        print(f"Chi² = {chi2:.2f}")
        return (popt, perr)
    except Exception as e:
        print(f"Fit failed for N_components = {number_of_components_tmp}: {e}")
        return (None, None)

#popt_amp, perr_amp = curvefit_amp_method(orig_wave, orig_flux_noisy, orig_flux_err_noisy, tmp_v_shifts, tmp_sig_shifts, weak_line_wave, number_of_components_tmp, amp_val_init=0.0, amp_val_min=0.0, amp_val_max=100.0)



def BIC_fitting_method(wave, flux_noisy, flux_err, magic_number, N_max = 7, penalty_factor = 0.1, line_wave=['6562.819'], v_init=0.0, v_lower=-250.0, v_upper=250.0, sigma_init=80.0, sigma_lower=10.0, sigma_upper=1500.0):
    # Input: your wave, flux, flux_err arrays
    #N_max = 7   # try 1 to N_max components
    #penalty_factor = 0.1  # Try 2.0 first. If still too permissive, increase to 3.0

    # Wrap the function
    wrapped_model = lambda wave, *theta: base_func.model_curvefit(
        wave, *theta, line_wave=line_wave, magic_number=magic_number)

    bic_list = []
    chi2_list = []
    fit_results = []
    for N_components in range(1, N_max + 1):
        print(f"\n=== Trying N_components = {N_components} ===")
        ndim = magic_number * N_components
        # Initial guess
        p0 = []
        lower_bounds = []
        upper_bounds = []
        for w in range(N_components):
            for x in range(len(line_wave)):
                p0 += [np.nanmax(flux_noisy) * 0.5]
                lower_bounds += [0.0]
                upper_bounds += [np.nanmax(flux_noisy)]
            p0 += [v_init, sigma_init]
            lower_bounds += [v_lower, sigma_lower]
            upper_bounds += [v_upper, sigma_upper]

        try:
            # Fit
            popt, pcov = curve_fit(wrapped_model, wave, flux_noisy, sigma=flux_err, absolute_sigma=True, p0=p0, bounds=(lower_bounds, upper_bounds))
            perr = np.sqrt(np.diag(pcov))

            # Evaluate model
            model_flux = wrapped_model(wave, *popt)

            # Chi²
            #chi2 = np.sum(((flux - model_flux) / flux_err) ** 2)
            residuals = flux_noisy - wrapped_model(wave, *popt)
            chi2 = np.sum((residuals / flux_err) ** 2)

            # BIC
            k = ndim
            N_data = len(wave)
            #bic = k * np.log(N_data) + chi2
            #bic = penalty_factor * k * np.log(N_data) + chi2
            bic = k * np.log(N_data) + chi2

            # Save results
            bic_list.append(bic)
            chi2_list.append(chi2)
            fit_results.append((popt, perr))

            print(f"Chi² = {chi2:.2f}, BIC = {bic:.2f}")

        except Exception as e:
            print(f"Fit failed for N_components = {N_components}: {e}")
            bic_list.append(np.inf)
            chi2_list.append(np.inf)
            fit_results.append((None, None))
    return (bic_list, chi2_list, fit_results)


#bic_list, chi2_list, fit_results = BIC_fitting_method(wave, flux_noisy, flux_err, magic_number, N_max = 7, penalty_factor = 0.1, v_init=0.0, v_lower=-250.0, v_upper=250.0, sigma_init=80.0, sigma_lower=10.0, sigma_upper=1500.0)



def get_weaker_lines_curvefit(wave, flux_noisy, flux_unc, component_info, best_N, line_wave=['6562.819'], line_name=['Ha'], delta_v_kms=500.):
    v_shifts, sigma_vs = [], []
    for i in range(best_N):
        v_shifts.append(float(component_info[f'vel_{i}']))
        sigma_vs.append(float(component_info[f'sigma_{i}']))
    # Wrapped model for curve_fit
    wrapped_model_amplitudes_curvefit = lambda wave, *theta: base_func.model_amplitudes_only(wave, *theta, line_wave=line_wave, v_shifts=v_shifts, sigma_vs=sigma_vs)
    
    p0 = []
    lower_bounds = []
    upper_bounds = []
    for w in range(best_N):
        for x in range(len(line_wave)):
            #p0 += [np.abs(np.nanmin(flux_noisy))]
            p0 += [0.0]
            lower_bounds += [0.0]
            #upper_bounds += [np.nanmedian(flux_noisy)*0.5]
            upper_bounds += [np.nanmax(flux_noisy)]

        #p0 += [v_init, sigma_init]
        #lower_bounds += [v_lower, sigma_lower]
        #upper_bounds += [v_upper, sigma_upper]

    #for i in range(len(p0)):
        #if lower_bounds[i]<=p0[i]<=upper_bounds[i]:
        #print (lower_bounds[i], p0[i], upper_bounds[i])
    # Example call:
    popt, pcov = curve_fit(wrapped_model_amplitudes_curvefit, wave, flux_noisy, sigma=flux_unc, absolute_sigma=True, p0=p0, bounds=(lower_bounds, upper_bounds))
    perr = np.sqrt(np.diag(pcov))
    for j in range(len(line_name)):
        tmp_line_id_rev = str(line_name[j])+str(int(line_wave[j]))
        component_info[f'tot_flux_{tmp_line_id_rev}'] = 0.0
        component_info[f'tot_flux_unc_{tmp_line_id_rev}'] = 0.0
        component_info[f'integrated_flux_{tmp_line_id_rev}'] = 0.0
        component_info[f'integrated_flux_unc_{tmp_line_id_rev}'] = 0.0
    for k in range(best_N):
        for l in range(len(line_wave)):
            idx = k * len(line_wave) + l
            tmp_line_id_rev2 = str(line_name[l])+str(int(line_wave[l]))
            component_info[f'amp_{tmp_line_id_rev2}_{k}'] = float(popt[idx])
            component_info[f'amp_unc_{tmp_line_id_rev2}_{k}'] = float(perr[idx])
            #component_info[f'flux_{tmp_line_id_rev2}_{k}'] = float(popt[idx]) * float(component_info[f'sigma_{k}']) * np.sqrt(2 * np.pi)
            #component_info[f'flux_unc_{tmp_line_id_rev2}_{k}'] = component_info[f'flux_{tmp_line_id_rev2}_{k}'] * np.sqrt( (float(perr[idx])/float(popt[idx]))**2 + (float(component_info[f'sigma_unc_{k}'])/float(component_info[f'sigma_{k}']))**2 )
            component_info[f'flux_{tmp_line_id_rev2}_{k}'], component_info[f'flux_unc_{tmp_line_id_rev2}_{k}'] = base_func.integrated_flux_fit(wave, float(component_info[f'amp_{tmp_line_id_rev2}_{k}']), float(component_info[f'amp_unc_{tmp_line_id_rev2}_{k}']), float(component_info[f'sigma_{k}']), float(component_info[f'sigma_unc_{k}']), float(line_wave[l]))
            component_info[f'tot_flux_{tmp_line_id_rev2}']+=float(component_info[f'flux_{tmp_line_id_rev2}_{k}'])
            component_info[f'tot_flux_unc_{tmp_line_id_rev2}']+=float(component_info[f'flux_unc_{tmp_line_id_rev2}_{k}'])**2
    for m in range(len(line_wave)):
        tmp_line_id_rev3 = str(line_name[m])+str(int(line_wave[m]))
        component_info[f'tot_flux_unc_{tmp_line_id_rev3}'] = np.sqrt(component_info[f'tot_flux_unc_{tmp_line_id_rev3}'])
        component_info[f'integrated_flux_{tmp_line_id_rev3}'], component_info[f'integrated_flux_unc_{tmp_line_id_rev3}'] = base_func.integrated_flux(wave, flux_noisy, flux_unc, line_wave[m], delta_v_kms = delta_v_kms)
    return (component_info, popt, perr)

#component_info, popt, perr = get_weaker_lines_curvefit(wave, flux_noisy, flux_unc, component_info, best_N, line_wave=['6562.819'], line_name=['Ha'], delta_v_kms=500.)

# Function to extract individual component fluxes
def extract_component_fluxes(wave, flux_noisy, flux_unc, theta, theta_unc=None, line_wave=['6562.819'], line_name=['Ha'], magic_number=3, delta_v_kms=400.):
    N_components = len(theta) // magic_number
    component_info_dict = {}
    for i in range(len(line_name)):
        line_id = str(line_name[i])+str(int(line_wave[i]))
        component_info_dict[f'tot_flux_{line_id}'] = 0.0
        component_info_dict[f'tot_flux_unc_{line_id}'] = 0.0
        component_info_dict[f'integrated_flux_{line_id}'] = 0.0
        component_info_dict[f'integrated_flux_unc_{line_id}'] = 0.0

    for j in range(N_components):
        base = j * magic_number
        count=0
        for k in range(len(line_wave)):
            line_id = str(line_name[k])+str(int(line_wave[k]))
            #component_info_dict[f'amp_{line_id}_{j}'] = 10**float(theta[base+k])
            component_info_dict[f'amp_{line_id}_{j}'] = float(theta[base+k])
            if theta_unc is not None:
                component_info_dict[f'amp_unc_{line_id}_{j}'] = float(theta_unc[base+k])
            count+=1
        component_info_dict[f'vel_{j}'] = float(theta[base+count])
        component_info_dict[f'sigma_{j}'] = float(theta[base+count+1])
        if theta_unc is not None:
            component_info_dict[f'vel_unc_{j}'] = float(theta_unc[base+count])
            component_info_dict[f'sigma_unc_{j}'] = float(theta_unc[base+count+1])
        for l in range(len(line_wave)):
            line_id = str(line_name[l])+str(int(line_wave[l]))
            component_info_dict[f'flux_{line_id}_{j}'] = float(component_info_dict[f'amp_{line_id}_{j}'] * component_info_dict[f'sigma_{j}'] * np.sqrt(2 * np.pi))
            component_info_dict[f'tot_flux_{line_id}']+=component_info_dict[f'flux_{line_id}_{j}']
            if theta_unc is not None:
                component_info_dict[f'flux_unc_{line_id}_{j}'] = float(np.abs(component_info_dict[f'flux_{line_id}_{j}']) * np.sqrt( (component_info_dict[f'amp_unc_{line_id}_{j}']/component_info_dict[f'amp_{line_id}_{j}'])**2 + (component_info_dict[f'sigma_unc_{j}']/component_info_dict[f'sigma_{j}'])**2 ))
                component_info_dict[f'tot_flux_unc_{line_id}']+=float(component_info_dict[f'flux_unc_{line_id}_{j}']**2)

    for m in range(len(line_name)):
        line_id = str(line_name[m])+str(int(line_wave[m]))
        component_info_dict[f'tot_flux_unc_{line_id}'] = np.sqrt(component_info_dict[f'tot_flux_unc_{line_id}'])
        component_info_dict[f'integrated_flux_{line_id}'], component_info_dict[f'integrated_flux_unc_{line_id}'] = base_func.integrated_flux(wave, flux_noisy, flux_unc, line_wave[m], delta_v_kms = delta_v_kms)

    return (component_info_dict)

#component_info_dict = extract_component_fluxes(wave, flux, flux_unc, theta, theta_unc=None, line_wave=['6562.819'], line_name=['Ha'], magic_number=3, delta_v_kms=400.)



def plot_strong_line_fig(wave, flux_noisy, flux_err, magic_number, bic_list, fit_results, figname='test.pdf', line_wave=['6562.819']):

    # Find best BIC
    best_N = np.argmin(bic_list) + 1
    print(f"\n*** Optimal number of components = {best_N} ***")

    # Plot final best model
    best_popt = fit_results[best_N - 1][0]
    best_perr = fit_results[best_N - 1][1]

    # Wrap the function
    wrapped_model = lambda wave, *theta: base_func.model_curvefit(
        wave, *theta, line_wave=line_wave, magic_number=magic_number)

    best_model_flux = wrapped_model(wave, *best_popt)

    # Plot model and individual components
    plt.figure(figsize=(10,6))
    plt.plot(wave, flux_noisy, label='Data', color='black', lw=1)
    plt.fill_between(wave, flux_noisy - flux_err, flux_noisy + flux_err, color='gray', alpha=0.3)

    # Plot full model
    plt.plot(wave, best_model_flux, label=f'Best model ({best_N} components)', color='red', lw=2)

    # Plot each component separately
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']  # Up to 7 components

    for y in range(best_N):
        base = y * magic_number
        # Extract component i only → set other components to 0
        theta_i = np.zeros_like(best_popt)
        theta_i_unc = np.zeros_like(best_popt)
        theta_i[base:base+magic_number] = best_popt[base:base+magic_number]
        theta_i_unc[base:base+magic_number] = best_perr[base:base+magic_number]

        model_flux_i = wrapped_model(wave, *theta_i)
        plt.plot(wave, model_flux_i, label=f'Component {y+1}', color=colors[y % len(colors)], lw=1.5, linestyle='--')

    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.legend()
    plt.title('Fit + Individual Components')
    plt.show()
    #plt.savefig(figname, dpi=100)
    #plt.close('all')
    for all_figs in plt.get_fignums():
        plt.close(all_figs)
    return (None)

#_ = plot_strong_line_fig(wave, flux_noisy, flux_err, magic_number, bic_list, fit_results, figname='test.pdf', line_wave=['6562.819'])



def plot_strong_line_stamp(wave, flux_noisy, flux_err, magic_number, bic_list, fit_results, figname='test.pdf', line_wave=['6562.819'], line_name=['6562.819'], vel_lim=500.):

    # Find best BIC
    best_N = np.argmin(bic_list) + 1
    print(f"\n*** Optimal number of components = {best_N} ***")

    # Plot final best model
    best_popt = fit_results[best_N - 1][0]
    best_perr = fit_results[best_N - 1][1]

    # Wrap the function
    wrapped_model = lambda wave, *theta: base_func.model_curvefit(
        wave, *theta, line_wave=line_wave, magic_number=magic_number)
    #Best fit
    best_model_flux = wrapped_model(wave, *best_popt)
    
    cellnum = base_func.ceil_perfect_square(len(line_wave))
    fig4, axes = plt.subplots(cellnum, cellnum, figsize=(cellnum * 5, cellnum * 5), sharey=False)
    axes = axes.flatten()
    colors = ['blue', 'green', 'orange', 'purple', 'cyan']  # Up to 5 components
    for yj in range(best_N):
        base = yj * magic_number
        theta_i = np.zeros_like(best_popt)
        theta_i_unc = np.zeros_like(best_popt)
        theta_i[base:base+magic_number] = best_popt[base:base+magic_number]
        theta_i_unc[base:base+magic_number] = best_perr[base:base+magic_number]
        model_flux_i = wrapped_model(wave, *theta_i)
        for xi in range(len(line_wave)):
            tmp_vel_prof = base_func.vel_prof(wave, float(line_wave[xi]))
            tmp_mask = (tmp_vel_prof>-vel_lim) & (tmp_vel_prof<vel_lim)
            axes[xi].errorbar(tmp_vel_prof[tmp_mask], flux_noisy[tmp_mask], yerr=flux_err[tmp_mask], ds='steps-mid', color='black', lw=1, zorder=1)
            axes[xi].plot(tmp_vel_prof[tmp_mask], best_model_flux[tmp_mask], label=f'Best model ({best_N} components)', color='red', lw=2)
            axes[xi].plot(tmp_vel_prof[tmp_mask], model_flux_i[tmp_mask], label=f'Component {yj+1}', color=colors[yj % len(colors)], lw=1.5, linestyle='--')
            #axes[xi].set_xlim(-vel_lim, vel_lim)
            axes[xi].set_xlabel(r'Velocity (km/s)')
            axes[xi].set_ylabel(r'Flux')
            axes[xi].set_title(f'{line_name[xi]}{int(line_wave[xi])}')
            axes[xi].axvline(0.0, ls='dashed', color='green')
            axes[xi].axhline(0.0, ls='dashed', color='green')


    fig4.suptitle('Fit + Individual Components')
    #plt.show()
    plt.savefig(figname, dpi=100)
    print (f"{figname} saved")
    plt.close(fig4)
    #for all_figs in plt.get_fignums():
    #    plt.close(all_figs)
    return (None)

#_ = plot_strong_line_stamp(wave, flux_noisy, flux_err, magic_number, bic_list, fit_results, figname='test.pdf', line_wave=['6562.819'])



def plot_weak_line_stamp(wave, flux_noisy, flux_err, component_info, N_components, popt, perr, figname='test.pdf', line_wave_tot=['6562.819'], line_name_tot=['Ha'], line_wave=['6562.819'], line_name=['Ha'], vel_lim=500., fontsize=25.):

    # Find best BIC
    best_N = N_components
    print(f"\n*** Optimal number of components = {best_N} ***")
    v_shifts, sigma_vs = [], []
    for i in range(N_components):
        v_shifts.append(float(component_info[f'vel_{i}']))
        sigma_vs.append(float(component_info[f'sigma_{i}']))

    # Plot final best model
    best_popt = popt
    best_perr = perr

    # Wrapped model for curve_fit
    wrapped_model_amplitudes_curvefit = lambda wave, *amp: base_func.model_amplitudes_only(wave, *amp, line_wave=line_wave_tot, v_shifts=v_shifts, sigma_vs=sigma_vs)
    #Best fit
    best_model_flux = wrapped_model_amplitudes_curvefit(wave, *best_popt)
    
    cellnum = base_func.ceil_perfect_square(len(line_wave))
    fig5, axes = plt.subplots(cellnum, cellnum, figsize=(cellnum * 5, cellnum * 5), sharey=False, sharex=True)
    axes = axes.flatten()
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']  # Up to 7 components
    for xi in range(len(line_wave)):
        tmp_vel_prof = base_func.vel_prof(wave, float(line_wave[xi]))
        tmp_mask = (tmp_vel_prof>-vel_lim) & (tmp_vel_prof<vel_lim)
        axes[xi].errorbar(tmp_vel_prof[tmp_mask], flux_noisy[tmp_mask], yerr=flux_err[tmp_mask], ds='steps-mid', color='black', lw=2, zorder=1)
        axes[xi].plot(tmp_vel_prof[tmp_mask], best_model_flux[tmp_mask], label=f'Best model ({best_N} components)', color='red', lw=4)
        axes[xi].axvline(0.0, ls='dashed', color='green')
        axes[xi].axhline(0.0, ls='dashed', color='green')

        row, col = divmod(xi, cellnum)  # get row and column index
        if col == 0:
            axes[xi].set_ylabel("Flux", fontsize=fontsize)
        if row == cellnum-1:
            axes[xi].set_xlabel("Relative Velocity (km/s)", fontsize=fontsize)
        axes[xi].xaxis.set_minor_locator(AutoMinorLocator())
        axes[xi].tick_params(axis='both', which='major', length=10, width=2, direction='in', labelsize=fontsize)
        axes[xi].tick_params(axis='x', which='minor', length=5, width=1.5, direction='in', labelsize=fontsize)
        axes[xi].yaxis.set_minor_locator(AutoMinorLocator())
        axes[xi].tick_params(axis='both', which='major', length=10, width=2, direction='in', labelsize=fontsize)
        axes[xi].tick_params(axis='y', which='minor', length=5, width=1.5, direction='in', labelsize=fontsize)
        axes[xi].set_title(f'{line_name[xi]}{int(line_wave[xi])}', fontsize=fontsize)
    fig5.suptitle('Weak lines')
    plt.tight_layout()
    plt.savefig(figname, dpi=100)
    print (f"{figname} saved")
    plt.close(fig5)
    #for all_figs in plt.get_fignums():
    #    plt.close(all_figs)
    return (None)

#_ = plot_weak_line_stamp(wave, flux_noisy, flux_err, component_info, N_components, popt, perr, figname='test.pdf', line_wave=['6562.819'], line_name=['Ha'], vel_lim=500.)



def plot_bic(bic_list, N_max, penalty_factor, figname='bic_fig.pdf'):
    fig2 = plt.figure(figsize=(10,8))
    plt.plot(range(1, N_max+1), bic_list, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title(f'BIC vs Components (penalty_factor = {penalty_factor})')
    #plt.show()
    plt.savefig(figname, dpi=100)
    print (f"{figname} saved")
    plt.close(fig2)
    #for all_figs in plt.get_fignums():
    #    plt.close(all_figs)
    return None

#_ = plot_bic(bic_list, N_max, penalty_factor)





def curvefit_method_abs(wave, flux, flux_err, ions=['HI'], unique_ions=['HI'], transitions=['HI1215'], l0_array=[1215.6701], f_array=[0.416400], gam_array=[6.265E8], ncomp=1, fwhm_kms=50.0, col_den_lower=10.0, col_den_upper=20.0, col_den_init=14.0, vel_init=0.0, sigma_init=30.0, v_lower=-500.0, v_upper=500.0, sigma_lower=1.0, sigma_upper=500.0, p0_init=None, lower_bounds_init=None, upper_bounds_init=None, limit_region=True, maxfev=20000, method='trf'):
    
    if (limit_region):
        mask = np.zeros_like(wave, dtype=bool)
        dv = v_upper-v_lower
        for l0 in l0_array:
            delta_lambda = l0 * dv / c_kms
            mask |= (wave >= (l0 - delta_lambda)) & (wave <= (l0 + delta_lambda))

        wave, flux, flux_err = wave[mask], flux[mask], flux_err[mask]

    ions, unique_ions, transitions, l0_array, f_array, gam_array = np.array(ions), np.array(unique_ions), np.array(transitions), np.array(l0_array), np.array(f_array), np.array(gam_array)

    # Wrap the function
    wrapped_model_abs = lambda wave, *theta: base_func.abs_model_for_curvefit(wave, *theta, ions=ions, unique_ions=unique_ions, transitions=transitions, l0_array=l0_array, f_array=f_array, gam_array=gam_array, ncomp=ncomp, fwhm_kms=fwhm_kms)


    print(f"\n=== Trying N_components = {ncomp} ===")

    v_init_list = list(np.full([ncomp], fill_value=vel_init))
    v_lower_list = list(np.full([ncomp], fill_value=v_lower))
    v_upper_list = list(np.full([ncomp], fill_value=v_upper))

    sigma_init_list = list(np.full([ncomp], fill_value=sigma_init))
    sigma_lower_list = list(np.full([ncomp], fill_value=sigma_lower))
    sigma_upper_list = list(np.full([ncomp], fill_value=sigma_upper))

    col_den_init_list = list(np.full([ncomp*len(unique_ions)], fill_value=col_den_init))
    col_den_lower_list = list(np.full([ncomp*len(unique_ions)], fill_value=col_den_lower))
    col_den_upper_list = list(np.full([ncomp*len(unique_ions)], fill_value=col_den_upper))

    p0, lower_bounds, upper_bounds = [], [], []
    p0.extend(sigma_init_list)
    p0.extend(v_init_list)
    lower_bounds.extend(sigma_lower_list)
    lower_bounds.extend(v_lower_list)
    upper_bounds.extend(sigma_upper_list)
    upper_bounds.extend(v_upper_list)
    p0.extend(col_den_init_list)
    lower_bounds.extend(col_den_lower_list)
    upper_bounds.extend(col_den_upper_list)
    if p0_init is not None:
        p0 = p0_init
    if lower_bounds_init is not None:
        lower_bounds = lower_bounds_init
    if upper_bounds_init is not None:
        upper_bounds = upper_bounds_init


    try:
        popt_init, pcov_init = curve_fit(wrapped_model_abs, wave, flux, sigma=flux_err, absolute_sigma=True, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=maxfev, method=method)
        popt, pcov = curve_fit(wrapped_model_abs, wave, flux, sigma=flux_err, absolute_sigma=True, p0=popt_init, bounds=(lower_bounds, upper_bounds), maxfev=maxfev, method=method)
        perr = np.sqrt(np.diag(pcov))
        model_flux_init = wrapped_model_abs(wave, *p0)
        model_flux = wrapped_model_abs(wave, *popt)
        residuals = flux - wrapped_model_abs(wave, *popt)
        chi2 = np.nansum((residuals / flux_err) ** 2)
        print (f'Best fit chi-sq:{chi2:.2}')
        #fig2, (ax2) = plt.subplots(nrows=1, figsize=(10, 6))  # The figsize argument controls the figure size
        #ax2.plot(wave, flux)
        #ax2.plot(wave, model_flux_init)
        #ax2.plot(wave, model_flux)
        #plt.show()
        return (popt, perr)
    except Exception as e:
        print(f"Fit failed for N_components = {ncomp}: {e}")
        return (None, None)

#popt, perr = curvefit_method_abs(wave, flux, flux_err, ions=['HI'], unique_ions=['HI'], transitions=['HI1215'], l0_array=[1215.6701], f_array=[0.416400], gam_array=[6.265E8], ncomp=1, fwhm_kms=50.0, col_den_lower=10.0, col_den_upper=20.0, col_den_init=14.0, vel_init=0.0, sigma_init=30.0, v_lower=-500.0, v_upper=500.0, sigma_lower=1.0, sigma_upper=500.0, p0=None, lower_bounds=None, upper_bounds=None, limit_region=True, maxfev=20000, method='trf')




def curvefit_method_abs_fixed(wave, flux, flux_err, vel_init_array, sigma_init_array, ions=['HI'], unique_ions=['HI'], transitions=['HI1215'], l0_array=[1215.6701], f_array=[0.416400], gam_array=[6.265E8], ncomp=1, fwhm_kms=50.0, col_den_lower=10.0, col_den_upper=20.0, col_den_init=14.0, vel_init=0.0, sigma_init=30.0, v_lower=-500.0, v_upper=500.0, sigma_lower=1.0, sigma_upper=500.0, p0_init=None, lower_bounds_init=None, upper_bounds_init=None, limit_region=True, maxfev=20000, method='trf'):
    
    if (limit_region):
        mask = np.zeros_like(wave, dtype=bool)
        dv = v_upper-v_lower
        for l0 in l0_array:
            delta_lambda = l0 * dv / c_kms
            mask |= (wave >= (l0 - delta_lambda)) & (wave <= (l0 + delta_lambda))

        wave, flux, flux_err = wave[mask], flux[mask], flux_err[mask]

    ions, unique_ions, transitions, l0_array, f_array, gam_array = np.array(ions), np.array(unique_ions), np.array(transitions), np.array(l0_array), np.array(f_array), np.array(gam_array)

    # Wrap the function
    wrapped_model_abs_fixed = lambda wave, *theta: base_func.abs_model_for_curvefit_fixed(wave, *theta, ions=ions, unique_ions=unique_ions, transitions=transitions, l0_array=l0_array, f_array=f_array, gam_array=gam_array, ncomp=ncomp, fwhm_kms=fwhm_kms, b_val=sigma_init_array, vel=vel_init_array)


    print(f"\n=== Trying N_components = {ncomp} ===")

    col_den_init_list = list(np.full([ncomp*len(unique_ions)], fill_value=col_den_init))
    col_den_lower_list = list(np.full([ncomp*len(unique_ions)], fill_value=col_den_lower))
    col_den_upper_list = list(np.full([ncomp*len(unique_ions)], fill_value=col_den_upper))

    p0, lower_bounds, upper_bounds = [], [], []
    p0.extend(col_den_init_list)
    lower_bounds.extend(col_den_lower_list)
    upper_bounds.extend(col_den_upper_list)
    if p0_init is not None:
        p0 = p0_init
    if lower_bounds_init is not None:
        lower_bounds = lower_bounds_init
    if upper_bounds_init is not None:
        upper_bounds = upper_bounds_init

    try:
        popt_init, pcov_init = curve_fit(wrapped_model_abs_fixed, wave, flux, sigma=flux_err, absolute_sigma=True, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=maxfev, method=method)
        popt, pcov = curve_fit(wrapped_model_abs_fixed, wave, flux, sigma=flux_err, absolute_sigma=True, p0=popt_init, bounds=(lower_bounds, upper_bounds), maxfev=maxfev, method=method)
        perr = np.sqrt(np.diag(pcov))
        model_flux_init = wrapped_model_abs_fixed(wave, *p0)
        model_flux = wrapped_model_abs_fixed(wave, *popt)
        residuals = flux - wrapped_model_abs_fixed(wave, *popt)
        chi2 = np.nansum((residuals / flux_err) ** 2)
        print (f'Best fit chi-sq:{chi2:.2}')
        return (popt, perr)
    except Exception as e:
        print(f"Fit failed for N_components = {ncomp}: {e}")
        return (None, None)

#popt, perr = curvefit_method_abs_fixed(wave, flux, flux_err, vel_init_array, sigma_init_array, ions=['HI'], unique_ions=['HI'], transitions=['HI1215'], l0_array=[1215.6701], f_array=[0.416400], gam_array=[6.265E8], ncomp=1, fwhm_kms=50.0, col_den_lower=10.0, col_den_upper=20.0, col_den_init=14.0, vel_init=0.0, sigma_init=30.0, v_lower=-500.0, v_upper=500.0, sigma_lower=1.0, sigma_upper=500.0, p0=None, lower_bounds=None, upper_bounds=None, limit_region=True, maxfev=20000, method='trf')
