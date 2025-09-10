import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from dynesty import NestedSampler
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
import corner


from matplotlib.ticker import AutoMinorLocator
import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 2})
import matplotlib as mpl
mpl.rcParams.update({'font.size': 10})

import basic_functions_module as base_func



def make_prior_transform(N_components, line_wave, magic_number, flux):
    ndim = N_components * magic_number
    flux_min = np.abs(np.nanmin(flux))
    flux_max = np.nanmax(flux)

    v_lower, v_upper = -250.0, 250.0
    sigma_lower, sigma_upper = 10.0, 1500.0

    def prior_transform(u):
        out = np.zeros_like(u)
        for o in range(N_components):
            base = o * magic_number
            # amplitudes per line
            for i in range(len(line_wave)):
                out[base + i] = flux_min + u[base + i] * (flux_max - flux_min)
            # velocity shift
            out[base + len(line_wave)] = v_lower + u[base + len(line_wave)] * (v_upper - v_lower)
            # velocity dispersion
            out[base + len(line_wave) + 1] = sigma_lower + u[base + len(line_wave) + 1] * (sigma_upper - sigma_lower)
        return out

    return prior_transform


def make_prior_transform_popt(popt):
    def prior_transform_popt(u):
        theta = np.zeros_like(u)
        for i in range(len(u)):
            # Example: uniform prior ± 20% around popt[i]
            lower = popt[i] * 0.8
            upper = popt[i] * 1.2
            theta[i] = lower + u[i] * (upper - lower)
        return theta
    return prior_transform_popt

def make_log_likelihood(wave, flux, flux_err, line_wave, magic_number):
    def log_likelihood(theta):
        model = base_func.model_curvefit(wave, *theta, line_wave=line_wave, magic_number=magic_number)
        chi2 = np.sum(((flux - model) / flux_err)**2)
        return -0.5 * chi2
    return log_likelihood


def mcmc_fit_dynesty(wave, flux, flux_err, N_components=2, magic_number=3, line_wave = [6562.819], line_name = ['Ha'], nwalkers = 50, nsteps=1000, discard_val=200, thin_val=10, delta_v_kms=500., corner_plot=True, popt=None, cornerplot_figname='corner_dynesty.pdf', dlogz=0.5):

    ndim = N_components * magic_number
    # Run nested sampler
    if popt is None:
        prior_transform = make_prior_transform(N_components, line_wave, magic_number, flux)
    else:
        prior_transform = make_prior_transform_popt(popt)
        #prior_transform = partial(make_prior_transform_popt, popt=popt)

    #print (prior_transform)
    log_likelihood = make_log_likelihood(wave, flux, flux_err, line_wave, magic_number)
    sampler = NestedSampler(log_likelihood, prior_transform, ndim, nlive=int(2.2*ndim), sample='rwalk')
    sampler.run_nested(dlogz=dlogz, print_progress=True)
    # Get results
    results = sampler.results

    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    samples_equal = samples[np.random.choice(len(samples), size=3000, p=weights/weights.sum(), replace=True)]

    if corner_plot:
        # Plot corner
        fig = plt.figure(figsize=(20,20))
        corner.corner(samples_equal, labels=[f"θ_{i}" for i in range(ndim)])
        plt.savefig(cornerplot_figname, dpi=100)
        print (f"{cornerplot_figname} saved")
        for all_figs in plt.get_fignums():
            plt.close(all_figs)
        #plt.close('all')
        #plt.show()

    # If you want a flat sample like emcee's get_chain(flat=True)
    flat_samples = dyfunc.resample_equal(samples, weights)
    #flat_samples = sampler.get_chain(discard=discard_val, thin=thin_val, flat=True)

    # Parameters
    n_lines = len(line_wave)
    n_components = len(flat_samples[0]) // magic_number

    component_mcmc_info_dict = {}
    for jj in range(len(line_name)):
        line_id = str(line_name[jj])+str(int(line_wave[jj]))
        component_mcmc_info_dict[f'tot_flux_{line_id}'] = 0.0
        component_mcmc_info_dict[f'tot_flux_unc_{line_id}'] = 0.0

    # Loop through each component
    for comp in range(n_components):
        base = comp * magic_number
        for kk in range(n_lines):
            line_id2 = str(line_name[kk])+str(int(line_wave[kk]))
            param_index = base + kk
            component_mcmc_info_dict[f'amp_{line_id2}_{comp}'] = np.nanmedian(flat_samples[:, param_index])
            component_mcmc_info_dict[f'amp_unc_{line_id2}_{comp}'] = np.nanstd(flat_samples[:, param_index])

        vshift_index = base + n_lines
        sigma_index = base + n_lines + 1
        component_mcmc_info_dict[f'vel_{comp}'] = np.nanmedian(flat_samples[:, vshift_index])
        component_mcmc_info_dict[f'sigma_{comp}'] = np.nanmedian(flat_samples[:, sigma_index])
        component_mcmc_info_dict[f'vel_unc_{comp}'] = np.nanstd(flat_samples[:, vshift_index])
        component_mcmc_info_dict[f'sigma_unc_{comp}'] = np.nanstd(flat_samples[:, sigma_index])
        for ll in range(n_lines):
            line_id3 = str(line_name[ll])+str(int(line_wave[ll]))
            #component_mcmc_info_dict[f'flux_{line_id3}_{comp}'] = float(component_mcmc_info_dict[f'amp_{line_id3}_{comp}'] * component_mcmc_info_dict[f'sigma_{comp}'] * np.sqrt(2 * np.pi))
            #component_mcmc_info_dict[f'flux_unc_{line_id3}_{comp}'] = float(np.abs(component_mcmc_info_dict[f'flux_{line_id3}_{comp}']) * np.sqrt( (component_mcmc_info_dict[f'amp_unc_{line_id3}_{comp}']/component_mcmc_info_dict[f'amp_{line_id3}_{comp}'])**2 + (component_mcmc_info_dict[f'sigma_unc_{comp}']/component_mcmc_info_dict[f'sigma_{comp}'])**2 ))
            component_mcmc_info_dict[f'flux_{line_id3}_{comp}'], component_mcmc_info_dict[f'flux_unc_{line_id3}_{comp}'] = base_func.integrated_flux_fit(wave, float(component_mcmc_info_dict[f'amp_{line_id3}_{comp}']), float(component_mcmc_info_dict[f'amp_unc_{line_id3}_{comp}']), float(component_mcmc_info_dict[f'sigma_{comp}']), float(component_mcmc_info_dict[f'sigma_unc_{comp}']), float(line_wave[ll]))
            component_mcmc_info_dict[f'tot_flux_{line_id3}']+=component_mcmc_info_dict[f'flux_{line_id3}_{comp}']
            component_mcmc_info_dict[f'tot_flux_unc_{line_id3}']+=float(component_mcmc_info_dict[f'flux_unc_{line_id3}_{comp}']**2)

    for mm in range(n_lines):
        line_id4 = str(line_name[mm])+str(int(line_wave[mm]))
        component_mcmc_info_dict[f'tot_flux_unc_{line_id4}'] = np.sqrt(component_mcmc_info_dict[f'tot_flux_unc_{line_id4}'])
        component_mcmc_info_dict[f'integrated_flux_{line_id4}'], component_mcmc_info_dict[f'integrated_flux_unc_{line_id4}'] = base_func.integrated_flux(wave, flux, flux_err, line_wave[mm], delta_v_kms = delta_v_kms)

    return (sampler, component_mcmc_info_dict)


#sampler, component_mcmc_info_dict = mcmc_fit_dynesty(wave, flux, flux_err, N_components=2, magic_number=3, line_wave = [6562.819], line_name = ['Ha'], nwalkers = 50, nsteps=1000, discard_val=200, thin_val=10, corner_plot=True)


def plot_dynesty_model(wave, flux_noisy, flux_err, flat_samples, magic_number, line_wave=['6562.819'], figname='test.pdf'):

    N_samples, ndim = flat_samples.shape
    N_components = ndim // magic_number

    # Compute median + 1σ errors
    best_popt = np.median(flat_samples, axis=0)
    lower = np.percentile(flat_samples, 16, axis=0)
    upper = np.percentile(flat_samples, 84, axis=0)
    best_perr = 0.5 * (upper - lower)

    # Wrap model
    wrapped_model = lambda wave, *theta: base_func.model_curvefit(wave, *theta, line_wave=line_wave, magic_number=magic_number)

    # Plot median model
    best_model_flux = wrapped_model(wave, *best_popt)

    plt.figure(figsize=(10,6))
    plt.plot(wave, flux_noisy, label='Data', color='black', lw=1)
    plt.fill_between(wave, flux_noisy - flux_err, flux_noisy + flux_err, color='gray', alpha=0.3)
    plt.plot(wave, best_model_flux, label=f'Median model ({N_components} components)', color='red', lw=2)

    # Plot uncertainty band from posterior models
    model_ensemble = np.array([wrapped_model(wave, *theta) for theta in flat_samples[np.random.choice(N_samples, 200, replace=False)]])
    lower_model = np.percentile(model_ensemble, 16, axis=0)
    upper_model = np.percentile(model_ensemble, 84, axis=0)
    plt.fill_between(wave, lower_model, upper_model, color='red', alpha=0.2, label='1σ band')

    # Plot individual components
    colors = ['blue', 'green', 'orange', 'purple', 'cyan']
    for y in range(N_components):
        base = y * magic_number
        theta_i = np.zeros_like(best_popt)
        theta_i[base:base+magic_number] = best_popt[base:base+magic_number]
        model_flux_i = wrapped_model(wave, *theta_i)
        plt.plot(wave, model_flux_i, label=f'Component {y+1}', color=colors[y % len(colors)], linestyle='--', lw=1.5)

    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.legend()
    plt.title('Dynesty Fit + Posterior Components')
    #plt.tight_layout()
    #plt.savefig(figname, dpi=120)
    plt.show()
    for all_figs in plt.get_fignums():
        plt.close(all_figs)


#_ = plot_dynesty_model(wave, flux_noisy, flux_err, flat_samples, magic_number, line_wave=['6562.819'], figname='test.pdf')


def plot_dynesty_stamp(wave_orig, flux_noisy, flux_err, sampler, magic_number, line_wave=['6562.819'], line_name=['Ha'], figname='test.pdf', vel_lim=500, fontsize=20):

    tmp_cluster_name = figname.replace('.pdf', '').replace('stamp_dynesty_', '').replace('stamp_dynesty3_', '')
    
    results = sampler.results
    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    flat_samples = dyfunc.resample_equal(samples, weights)

    #wave = 10**(np.logspace(np.log10(np.nanmin(wave_orig)), np.log10(np.nanmax(wave_orig)), len(wave_orig)*10))
    #wave = np.linspace(np.nanmin(wave_orig), np.nanmax(wave_orig), len(wave_orig)*10)
    wave = wave_orig

    N_samples, ndim = flat_samples.shape
    N_components = ndim // magic_number

    # Compute median + 1σ errors
    best_popt = np.median(flat_samples, axis=0)
    lower = np.percentile(flat_samples, 16, axis=0)
    upper = np.percentile(flat_samples, 84, axis=0)
    best_perr = 0.5 * (upper - lower)

    # Wrap model
    wrapped_model = lambda wave, *theta: base_func.model_curvefit(wave, *theta, line_wave=line_wave, magic_number=magic_number)

    # Plot median model
    best_model_flux = wrapped_model(wave, *best_popt)

    cellnum = base_func.ceil_perfect_square(len(line_wave))
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for i in range(3):
        axes[i, 0].set_ylabel("Emission Flux", fontsize=fontsize)
    for j in range(3):  # loop over columns
        axes[-1, j].set_xlabel(r"Relative Velocity kms$\rm ^{-1}$", fontsize=fontsize)


    axes = axes.flatten()
    colors = ['blue', 'green', 'orange', 'purple', 'cyan']  # Up to 5 components
    for yj in range(N_components):
        base = yj * magic_number
        theta_i = np.zeros_like(best_popt)
        theta_i_unc = np.zeros_like(best_popt)
        theta_i[base:base+magic_number] = best_popt[base:base+magic_number]
        theta_i_unc[base:base+magic_number] = best_perr[base:base+magic_number]
        model_flux_i = wrapped_model(wave, *theta_i)
        for xi in range(len(line_wave)):
            tmp_vel_prof = base_func.vel_prof(wave_orig, float(line_wave[xi]))
            tmp_mask = (tmp_vel_prof>-vel_lim) & (tmp_vel_prof<vel_lim)
            axes[xi].errorbar(tmp_vel_prof[tmp_mask], flux_noisy[tmp_mask], yerr=flux_err[tmp_mask], ds='steps-mid', color='black', lw=1, zorder=1)
            tmp_vel_prof2 = base_func.vel_prof(wave, float(line_wave[xi]))
            tmp_mask2 = (tmp_vel_prof2>-vel_lim) & (tmp_vel_prof2<vel_lim)
            axes[xi].plot(tmp_vel_prof2[tmp_mask2], best_model_flux[tmp_mask2], label=f'Best model ({N_components} components)', color='red', lw=2)
            axes[xi].plot(tmp_vel_prof2[tmp_mask2], model_flux_i[tmp_mask2], label=f'Component {yj+1}', color=colors[yj % len(colors)], lw=1.5, linestyle='--')
            #axes[xi].set_xlim(-vel_lim, vel_lim)
            #axes[xi].set_xlabel(r'Velocity (km/s)', fontsize=fontsize)
            #axes[xi].set_ylabel(r'Flux', fontsize=fontsize)
            axes[xi].set_title(f'{line_name[xi]}{int(line_wave[xi])}', fontsize=fontsize)

            axes[xi].xaxis.set_minor_locator(AutoMinorLocator())
            axes[xi].tick_params(axis='both', which='major', length=10, width=2, direction='in', labelsize=fontsize)
            axes[xi].tick_params(axis='x', which='minor', length=5, width=1.5, direction='in', labelsize=fontsize)
            axes[xi].yaxis.set_minor_locator(AutoMinorLocator())
            axes[xi].tick_params(axis='both', which='major', length=10, width=2, direction='in', labelsize=fontsize)
            axes[xi].tick_params(axis='y', which='minor', length=5, width=1.5, direction='in', labelsize=fontsize)


    #fig.suptitle('Fit + Individual Components')
    fig.suptitle(f'{tmp_cluster_name}: Fit + Individual Components', fontsize=1.5*fontsize)
    plt.tight_layout()
    #plt.show()
    plt.savefig(figname, dpi=100)
    print (f"{figname} saved")
    #plt.close('all')
    for all_figs in plt.get_fignums():
        plt.close(all_figs)
    return (None)

#_ = plot_dynesty_stamp(wave, flux_noisy, flux_err, flat_samples, magic_number, line_wave=['6562.819'], figname='test.pdf')



def make_prior_transform_amplitudes(flux, N_params):
    flux_min = np.abs(np.nanmin(flux))
    flux_max = np.nanmax(flux)

    def prior_transform(u):
        out = np.zeros_like(u)
        for i in range(N_params):
            out[i] = flux_min + u[i] * (flux_max - flux_min)
        return out

    return prior_transform


def make_prior_transform_popt_amplitudes(popt):
    def prior_transform_popt_amplitudes(u):
        theta = np.zeros_like(u)
        for i in range(len(u)):
            lower = popt[i] * 0.8
            upper = popt[i] * 1.2
            theta[i] = lower + u[i] * (upper - lower)
        return theta
    return prior_transform_popt_amplitudes



def make_log_likelihood_amplitudes(wave, flux, flux_err, line_wave, v_shifts, sigma_vs):
    def log_likelihood_amplitudes(theta):
        model = base_func.model_amplitudes_only(wave, *theta, line_wave=line_wave, v_shifts=v_shifts, sigma_vs=sigma_vs)
        chi2 = np.sum(((flux - model) / flux_err)**2)
        return -0.5 * chi2
    return log_likelihood_amplitudes



def mcmc_fit_dynesty_amplitudes(wave, flux, flux_err, best_N, component_mcmc_info_dict, line_wave = [6562.819], line_name = ['Ha'], popt=None, corner_plot=False, cornerplot_figname='corner_dynesty_amp.pdf', delta_v_kms=500., dlogz=0.5):

    line_wave_shifted, sigma_aa_values = [], []
    for i in range(best_N):
        line_wave_shifted.append(float(component_mcmc_info_dict[f'vel_{i}']))
        sigma_aa_values.append(float(component_mcmc_info_dict[f'sigma_{i}']))

    ndim = len(popt)  # same as number of amplitudes

    # Prior setup
    if popt is None:
        prior_transform = make_prior_transform_amplitudes(flux, ndim)
    else:
        prior_transform = make_prior_transform_popt_amplitudes(popt)

    # Likelihood
    log_likelihood_amplitudes = make_log_likelihood_amplitudes(wave, flux, flux_err, line_wave, line_wave_shifted, sigma_aa_values)

    # Run dynesty
    sampler = NestedSampler(log_likelihood_amplitudes, prior_transform, ndim, nlive=int(2.2*ndim), sample='rwalk')
    sampler.run_nested(dlogz=dlogz, print_progress=True)

    # Get results
    results = sampler.results

    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    samples_equal = samples[np.random.choice(len(samples), size=3000, p=weights/weights.sum(), replace=True)]

    if corner_plot:
        # Plot corner
        fig = plt.figure(figsize=(20,20))
        corner.corner(samples_equal, labels=[f"θ_{i}" for i in range(ndim)])
        plt.savefig(cornerplot_figname, dpi=100)
        print (f"{cornerplot_figname} saved")
        for all_figs in plt.get_fignums():
            plt.close(all_figs)
        #plt.close('all')
        #plt.show()

    # If you want a flat sample like emcee's get_chain(flat=True)
    flat_samples = dyfunc.resample_equal(samples, weights)
    #flat_samples = sampler.get_chain(discard=discard_val, thin=thin_val, flat=True)

    # Parameters
    n_lines = len(line_wave)
    n_components = best_N

    #component_mcmc_info_dict = {}
    for jj in range(len(line_name)):
        line_id = str(line_name[jj])+str(int(line_wave[jj]))
        component_mcmc_info_dict[f'tot_flux_{line_id}'] = 0.0
        component_mcmc_info_dict[f'tot_flux_unc_{line_id}'] = 0.0

    for k in range(best_N):
        for l in range(len(line_wave)):
            idx = k * len(line_wave) + l
            tmp_line_id_rev2 = str(line_name[l])+str(int(line_wave[l]))
            component_mcmc_info_dict[f'amp_{tmp_line_id_rev2}_{k}'] = np.nanmedian(flat_samples[:, idx])
            component_mcmc_info_dict[f'amp_unc_{tmp_line_id_rev2}_{k}'] = np.nanstd(flat_samples[:, idx])
            #component_mcmc_info_dict[f'flux_{tmp_line_id_rev2}_{k}'] = float(np.nanmedian(flat_samples[:, idx])) * float(component_mcmc_info_dict[f'sigma_{k}']) * np.sqrt(2 * np.pi)
            #component_mcmc_info_dict[f'flux_unc_{tmp_line_id_rev2}_{k}'] = component_mcmc_info_dict[f'flux_{tmp_line_id_rev2}_{k}'] * np.sqrt( (float(np.nanstd(flat_samples[:, idx]))/float(np.nanmedian(flat_samples[:, idx])))**2 + (float(component_mcmc_info_dict[f'sigma_unc_{k}'])/float(component_mcmc_info_dict[f'sigma_{k}']))**2 )
            component_mcmc_info_dict[f'flux_{tmp_line_id_rev2}_{k}'], component_mcmc_info_dict[f'flux_unc_{tmp_line_id_rev2}_{k}'] = base_func.integrated_flux_fit(wave, float(component_mcmc_info_dict[f'amp_{tmp_line_id_rev2}_{k}']), float(component_mcmc_info_dict[f'amp_unc_{tmp_line_id_rev2}_{k}']), float(component_mcmc_info_dict[f'sigma_{k}']), float(component_mcmc_info_dict[f'sigma_unc_{k}']), float(line_wave[l]))
            component_mcmc_info_dict[f'tot_flux_{tmp_line_id_rev2}']+=float(component_mcmc_info_dict[f'flux_{tmp_line_id_rev2}_{k}'])
            component_mcmc_info_dict[f'tot_flux_unc_{tmp_line_id_rev2}']+=float(component_mcmc_info_dict[f'flux_unc_{tmp_line_id_rev2}_{k}'])**2
    for m in range(len(line_wave)):
        tmp_line_id_rev3 = str(line_name[m])+str(int(line_wave[m]))
        component_mcmc_info_dict[f'tot_flux_unc_{tmp_line_id_rev3}'] = np.sqrt(component_mcmc_info_dict[f'tot_flux_unc_{tmp_line_id_rev3}'])
        component_mcmc_info_dict[f'integrated_flux_{tmp_line_id_rev3}'], component_mcmc_info_dict[f'integrated_flux_unc_{tmp_line_id_rev3}'] = base_func.integrated_flux(wave, flux, flux_err, line_wave[m], delta_v_kms = delta_v_kms)
    return (sampler, component_mcmc_info_dict)

#sampler, component_mcmc_info_dict = mcmc_fit_dynesty_amplitudes(wave, flux, flux_err, line_wave_shifted, sigma_aa_values, best_N, component_mcmc_info_dict, line_wave = [6562.819], line_name = ['Ha'], popt=None, cornerplot_figname='corner_dynesty_amp.pdf', delta_v_kms=500.)



def plot_weak_line_stamp_dynesty(wave, flux_noisy, flux_err, sampler, component_info, N_components, figname='test.pdf', tot_line_wave=['6562.819'], tot_line_name=['Ha'], line_wave=['6562.819'], line_name=['Ha'], vel_lim=500., fontsize=10.):
    
    cluster_name_tmp = figname.replace('.pdf', '').replace('stamp_dynesty2_', '').replace('stamp_dynesty4_', '')

    # Get results
    results = sampler.results

    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    flat_samples = dyfunc.resample_equal(samples, weights)
    n_lines = len(line_wave)
    #n_components = N_components
    best_popt, best_perr = [], []
    for k in range(N_components):
        for l in range(len(tot_line_wave)):
            idx = k * len(tot_line_wave) + l
            #component_mcmc_info_dict[f'amp_{tmp_line_id_rev2}_{k}'] = np.nanmedian(flat_samples[:, idx])
            #component_mcmc_info_dict[f'amp_unc_{tmp_line_id_rev2}_{k}'] = np.nanstd(flat_samples[:, idx])
            best_popt.append(float(np.nanmedian(flat_samples[:, idx])))
            best_perr.append(float(np.nanstd(flat_samples[:, idx])))

    # Find best BIC
    best_N = N_components
    print(f"\n*** Optimal number of components = {best_N} ***")
    v_shifts, sigma_vs = [], []
    for i in range(N_components):
        v_shifts.append(float(component_info[f'vel_{i}']))
        sigma_vs.append(float(component_info[f'sigma_{i}']))

    # Plot final best model
    #best_popt = popt
    #best_perr = perr

    # Wrapped model for curve_fit
    wrapped_model_amplitudes_curvefit = lambda wave, *amp: base_func.model_amplitudes_only(wave, *amp, line_wave=tot_line_wave, v_shifts=v_shifts, sigma_vs=sigma_vs)
    #Best fit
    best_model_flux = wrapped_model_amplitudes_curvefit(wave, *best_popt)
    
    cellnum = base_func.ceil_perfect_square(len(line_wave))
    #fig, axes = plt.subplots(cellnum, cellnum, figsize=(cellnum * 5, cellnum * 5), sharey=False, sharex=True)
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharey=False, sharex=True)
    for i in range(4):
        axes[i, 0].set_ylabel("Emission Flux", fontsize=fontsize)
    for j in range(4):  # loop over columns
        axes[-1, j].set_xlabel(r"Relative Velocity kms$\rm ^{-1}$", fontsize=fontsize)
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
        #if col == 0:
            #axes[xi].set_ylabel("Flux", fontsize=fontsize)
        #if row == cellnum-1:
            #axes[xi].set_xlabel("Relative Velocity (km/s)", fontsize=fontsize)
        axes[xi].xaxis.set_minor_locator(AutoMinorLocator())
        axes[xi].tick_params(axis='both', which='major', length=10, width=2, direction='in', labelsize=fontsize)
        axes[xi].tick_params(axis='x', which='minor', length=5, width=1.5, direction='in', labelsize=fontsize)
        axes[xi].yaxis.set_minor_locator(AutoMinorLocator())
        axes[xi].tick_params(axis='both', which='major', length=10, width=2, direction='in', labelsize=fontsize)
        axes[xi].tick_params(axis='y', which='minor', length=5, width=1.5, direction='in', labelsize=fontsize)
        axes[xi].set_title(f'{line_name[xi]}{int(line_wave[xi])}', fontsize=fontsize)

    axes[-1].xaxis.set_minor_locator(AutoMinorLocator())
    axes[-1].tick_params(axis='both', which='major', length=10, width=2, direction='in', labelsize=fontsize)
    axes[-1].tick_params(axis='x', which='minor', length=5, width=1.5, direction='in', labelsize=fontsize)
    axes[-1].yaxis.set_minor_locator(AutoMinorLocator())
    axes[-1].tick_params(axis='both', which='major', length=10, width=2, direction='in', labelsize=fontsize)
    axes[-1].tick_params(axis='y', which='minor', length=5, width=1.5, direction='in', labelsize=fontsize)

    #fig.suptitle('Weak lines', fontsize=1.5*fontsize, ha='center')
    fig.suptitle(f'{cluster_name_tmp}', fontsize=1.5*fontsize, ha='center')
    #plt.tight_layout()
    plt.savefig(figname, dpi=100)
    print (f"{figname} saved")
    for all_figs in plt.get_fignums():
        plt.close(all_figs)
    return (None)

#_ = plot_weak_line_stamp_dynesty(wave, flux_noisy, flux_err, sampler, component_info, N_components, figname='test.pdf', line_wave=['6562.819'], line_name=['Ha'], vel_lim=500., fontsize=25.)

