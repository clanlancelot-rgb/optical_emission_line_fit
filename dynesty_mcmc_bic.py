import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import scipy as sp

from astropy import constants as const
c_cms = const.c.to('cm/s').value
c_kms = const.c.to('km/s').value

#import continuum_function_module as cont_func
import dynesty_mcmc_module as dynesty_mcmc_func
import curvefit_module as curvefit_func
import basic_functions_module as base_func

from basic_functions_module import base_config
inst_cont = base_config()
config = inst_cont.load_config()
cwd = inst_cont.cwd
codewd = inst_cont.codewd

this_file_extension = config['this_file_extension']
#Velocity range for figure stamps
fig_vel_lim = config['fig_vel_lim']
#Velocity range for fitting area
allowable_velocity_range = config['allowable_velocity_range']

#Curevfit Params
curvefit_v_init = config['curvefit_v_init']
curvefit_v_lower = config['curvefit_v_lower']
curvefit_v_upper = config['curvefit_v_upper']
curvefit_sigma_init = config['curvefit_sigma_init']
curvefit_sigma_lower = config['curvefit_sigma_lower']
curvefit_sigma_upper = config['curvefit_sigma_upper']
# Generate model
#N_max is the maximum number of components that needs to be fit by BIC
N_max = config['N_max']
#penalty_factor is the BIC penalty factor
penalty_factor = config['penalty_factor']
z = float(config['z'])  # Actual redshift of the galaxy


#MCMC Dynesty params
dynesty_nwalkers = config['dynesty_nwalkers']
dynesty_nsteps = config['dynesty_nsteps']
dynesty_discard_val = config['dynesty_discard_val']
dynesty_thin_val = config['dynesty_thin_val']
dlogz_mcmc = config['dlogz_mcmc']

file_with_cont = f'data_with_cont_{this_file_extension}.dat'
tmp_bic_file = f'bic_result_{this_file_extension}.pkl'
tmp_weak_file = f'weak_result_{this_file_extension}.pkl'
tmp_bic_fig = f'bic_fig_{this_file_extension}.pdf'
tmp_curvefit_stamp_fig = f'stamp_curvefit_{this_file_extension}.pdf'
tmp_curvefit_stamp_fig2 = f'stamp_curvefit2_{this_file_extension}.pdf'
tmp_compare_curvefit_result_fig = f'checking_results_curvefit_{this_file_extension}.pdf'
tmp_compare_curvefit_result_fig2 = f'checking_results_curvefit2_{this_file_extension}.pdf'
tmp_dynesty_file = f'dynesty_{this_file_extension}.pkl'
tmp_dynesty_file_weak = f'dynesty_weak_{this_file_extension}.pkl'
emcee_corner_fig = f'corner_mcmc_{this_file_extension}.pdf'
emcee_stamp_plot = f'stamp_emcee_{this_file_extension}.pdf'

if os.path.exists(file_with_cont):
    orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit = np.genfromtxt(file_with_cont, delimiter=',', unpack=True, encoding=None, dtype=np.float32)
    print (f"{file_with_cont} loaded")
else:
    print (f"{file_with_cont} not found. Quitting.")
    quit()

line_list_file = np.genfromtxt(f'{codewd}nebular_emission_line_list - Sheet2.csv', delimiter=',', encoding=None, dtype=None, names=True)
orig_line_id, orig_line_wave, orig_line_id_rev = base_func.obtain_lines(orig_wave, line_list_file)




if config['instrument']=='MUSE':
    strong_lines = np.array(['Hβ4861', '[O_III]4958', '[O_III]5006', '[N_II]6548', 'Hα6562', '[N_II]6583', '[S_II]6716', '[S_II]6730', '[S_III]9068'])
    #Removing [Fe_IV]6739 because it is affecting the SII fit
    #Removing '[Ar_IV]4740', '[Ar_IV]4711' for later
    weak_lines = np.array(['[Fe_III]4658', '[Fe_III]4701', '[Ar_IV]4740', '[Ar_IV]4711', '[Ar_III]5191', '[Cl_III]5517', '[Cl_III]5537', '[N_II]5754', '[S_III]6312', '[Fe_IV]6739', '[Ar_III]7135', '[O_II]7320', '[O_II]7329', '[Fe_II]8616', '[Fe_II]8891'])
    #weak_lines = np.array(['[Fe_III]4658', '[Fe_III]4701', '[Ar_III]5191', '[Cl_III]5517', '[Cl_III]5537', '[N_II]5754', '[S_III]6312', '[Ar_III]7135', '[O_II]7320', '[O_II]7329', '[Fe_II]8616', '[Fe_II]8891'])
    dynesty_corner_fig = f'corner_dynesty_{this_file_extension}.pdf'
    dynesty_corner_fig2 = f'corner_dynesty2_{this_file_extension}.pdf'
    dynesty_stamp_fig = f'stamp_dynesty_{this_file_extension}.pdf'
    dynesty_stamp_fig2 = f'stamp_dynesty2_{this_file_extension}.pdf'
    checking_dynesty_res_fig = f'checking_results_dynesty_{this_file_extension}.pdf'
    checking_dynesty_res_fig2 = f'checking_results_dynesty2_{this_file_extension}.pdf'
    checking_dynesty_res_fig3 = f'checking_results_dynesty3_{this_file_extension}.pdf'
    checking_dynesty_res_fig4 = f'checking_results_dynesty4_{this_file_extension}.pdf'


elif config['instrument']=='LBT-BLUE':
    tmp_limit = 5490.
    #orig_wave = orig_wave*(1.+z)
    orig_flux_noisy = orig_flux_noisy[(orig_wave<tmp_limit)]
    orig_flux_err = orig_flux_err[(orig_wave<tmp_limit)]
    orig_wave = orig_wave[(orig_wave<tmp_limit)]
    strong_lines = np.array(['[O_II]3726', '[O_II]3728', 'Hβ4861', '[O_III]4958', '[O_III]5006'])
    #Removing '[Ar_IV]4740', '[Ar_IV]4711' for later
    weak_lines = np.array(['[Fe_V]4227', '[O_III]4363', '[Fe_III]4658', '[Fe_III]4701', '[Ar_IV]4740', '[Ar_IV]4711', '[Ar_III]5191'])
    #weak_lines = np.array(['[Fe_V]4227', '[O_III]4363', '[Fe_III]4658', '[Fe_III]4701', '[Ar_III]5191'])
    dynesty_corner_fig = f'corner_dynesty_{this_file_extension}.pdf'
    dynesty_corner_fig2 = f'corner_dynesty2_{this_file_extension}.pdf'
    dynesty_stamp_fig = f'stamp_dynesty_{this_file_extension}.pdf'
    dynesty_stamp_fig2 = f'stamp_dynesty2_{this_file_extension}.pdf'
    checking_dynesty_res_fig = f'checking_results_dynesty_{this_file_extension}.pdf'
    checking_dynesty_res_fig2 = f'checking_results_dynesty2_{this_file_extension}.pdf'
    checking_dynesty_res_fig3 = f'checking_results_dynesty3_{this_file_extension}.pdf'
    checking_dynesty_res_fig4 = f'checking_results_dynesty4_{this_file_extension}.pdf'


elif config['instrument']=='LBT-RED':
    tmp_limit = 5490.
    #orig_wave = orig_wave*(1.+z)
    orig_flux_noisy = orig_flux_noisy[(orig_wave>tmp_limit)]
    orig_flux_err = orig_flux_err[(orig_wave>tmp_limit)]
    orig_wave = orig_wave[(orig_wave>tmp_limit)]
    strong_lines = np.array(['[N_II]6548', 'Hα6562', '[N_II]6583', '[S_II]6716', '[S_II]6730', '[S_III]9068', '[S_III]9530'])
    #Removing [Fe_IV]6739 because it is affecting the SII fit
    #weak_lines = np.array(['[Cl_III]5517', '[Cl_III]5537', '[N_II]5754', '[S_III]6312', '[Ar_III]7135', '[O_II]7320', '[O_II]7329', '[Fe_II]8616', '[Fe_II]8891'])
    weak_lines = np.array(['[Cl_III]5517', '[Cl_III]5537', '[N_II]5754', '[S_III]6312', '[Fe_IV]6739', '[Ar_III]7135', '[O_II]7320', '[O_II]7329', '[Fe_II]8616', '[Fe_II]8891'])
    dynesty_corner_fig = f'corner_dynesty3_{this_file_extension}.pdf'
    dynesty_corner_fig2 = f'corner_dynesty4_{this_file_extension}.pdf'
    dynesty_stamp_fig = f'stamp_dynesty3_{this_file_extension}.pdf'
    dynesty_stamp_fig2 = f'stamp_dynesty4_{this_file_extension}.pdf'
    checking_dynesty_res_fig = f'checking_results_dynesty3_{this_file_extension}.pdf'
    checking_dynesty_res_fig2 = f'checking_results_dynesty4_{this_file_extension}.pdf'
    checking_dynesty_res_fig3 = f'checking_results_dynesty3_{this_file_extension}.pdf'
    checking_dynesty_res_fig4 = f'checking_results_dynesty4_{this_file_extension}.pdf'






'''
if config['instrument']=='MUSE':
    strong_lines = np.array(['Hβ4861', '[O_III]4958', '[O_III]5006', '[N_II]6548', 'Hα6562', '[N_II]6583', '[S_II]6716', '[S_II]6730', '[S_III]9068'])
    #Removing [Fe_IV]6739 because it is affecting the SII fit
    #Removing '[Ar_IV]4740', '[Ar_IV]4711' for later
    #weak_lines = np.array(['[Fe_III]4658', '[Fe_III]4701', '[Ar_IV]4740', '[Ar_IV]4711', '[Ar_III]5191', '[Cl_III]5517', '[Cl_III]5537', '[N_II]5754', '[S_III]6312', '[Ar_III]7135', '[O_II]7320', '[O_II]7329', '[Fe_II]8616', '[Fe_II]8891'])
    weak_lines = np.array(['[Fe_III]4658', '[Fe_III]4701', '[Ar_III]5191', '[Cl_III]5517', '[Cl_III]5537', '[N_II]5754', '[S_III]6312', '[Ar_III]7135', '[O_II]7320', '[O_II]7329', '[Fe_II]8616', '[Fe_II]8891'])
    dynesty_corner_fig = f'corner_dynesty_{this_file_extension}.pdf'
    dynesty_corner_fig2 = f'corner_dynesty2_{this_file_extension}.pdf'
    dynesty_stamp_fig = f'stamp_dynesty_{this_file_extension}.pdf'
    dynesty_stamp_fig2 = f'stamp_dynesty2_{this_file_extension}.pdf'
    checking_dynesty_res_fig = f'checking_results_dynesty_{this_file_extension}.pdf'
    checking_dynesty_res_fig2 = f'checking_results_dynesty2_{this_file_extension}.pdf'


elif config['instrument']=='LBT-BLUE':
    tmp_limit = 5490.
    #orig_wave = orig_wave*(1.+z)
    orig_flux_noisy = orig_flux_noisy[(orig_wave<tmp_limit)]
    orig_flux_err = orig_flux_err[(orig_wave<tmp_limit)]
    orig_continuum = orig_continuum[(orig_wave<tmp_limit)]
    orig_fit = orig_fit[(orig_wave<tmp_limit)]
    orig_wave = orig_wave[(orig_wave<tmp_limit)]
    strong_lines = np.array(['[O_II]3726', '[O_II]3728', 'Hβ4861', '[O_III]4958', '[O_III]5006'])
    #Removing '[Ar_IV]4740', '[Ar_IV]4711' for later
    #weak_lines = np.array(['[Fe_V]4227', '[O_III]4363', '[Fe_III]4658', '[Fe_III]4701', '[Ar_IV]4740', '[Ar_IV]4711', '[Ar_III]5191'])
    weak_lines = np.array(['[Fe_V]4227', '[O_III]4363', '[Fe_III]4658', '[Fe_III]4701', '[Ar_III]5191'])
    dynesty_corner_fig = f'corner_dynesty_{this_file_extension}.pdf'
    dynesty_corner_fig2 = f'corner_dynesty2_{this_file_extension}.pdf'
    dynesty_stamp_fig = f'stamp_dynesty_{this_file_extension}.pdf'
    dynesty_stamp_fig2 = f'stamp_dynesty2_{this_file_extension}.pdf'
    checking_dynesty_res_fig = f'checking_results_dynesty_{this_file_extension}.pdf'
    checking_dynesty_res_fig2 = f'checking_results_dynesty2_{this_file_extension}.pdf'


elif config['instrument']=='LBT-RED':
    tmp_limit = 5490.
    #orig_wave = orig_wave*(1.+z)
    orig_flux_noisy = orig_flux_noisy[(orig_wave>tmp_limit)]
    orig_flux_err = orig_flux_err[(orig_wave>tmp_limit)]
    orig_continuum = orig_continuum[(orig_wave>tmp_limit)]
    orig_fit = orig_fit[(orig_wave>tmp_limit)]
    orig_wave = orig_wave[(orig_wave>tmp_limit)]
    strong_lines = np.array(['[N_II]6548', 'Hα6562', '[N_II]6583', '[S_II]6716', '[S_II]6730', '[S_III]9068', '[S_III]9530'])
    #Removing [Fe_IV]6739 because it is affecting the SII fit
    weak_lines = np.array(['[Cl_III]5517', '[Cl_III]5537', '[N_II]5754', '[S_III]6312', '[Ar_III]7135', '[O_II]7320', '[O_II]7329', '[Fe_II]8616', '[Fe_II]8891'])
    dynesty_corner_fig = f'corner_dynesty3_{this_file_extension}.pdf'
    dynesty_corner_fig2 = f'corner_dynesty4_{this_file_extension}.pdf'
    dynesty_stamp_fig = f'stamp_dynesty3_{this_file_extension}.pdf'
    dynesty_stamp_fig2 = f'stamp_dynesty4_{this_file_extension}.pdf'
    checking_dynesty_res_fig = f'checking_results_dynesty3_{this_file_extension}.pdf'
    checking_dynesty_res_fig2 = f'checking_results_dynesty4_{this_file_extension}.pdf'
'''




strong_line_idx = np.where(np.isin(orig_line_id_rev, strong_lines))[0]
line_id, line_wave, line_id_rev = orig_line_id[strong_line_idx], orig_line_wave[strong_line_idx], orig_line_id_rev[strong_line_idx]
magic_number = int(len(line_wave)+2)
weak_line_idx = np.where(np.isin(orig_line_id_rev, weak_lines))[0]
line_id_weak, line_wave_weak, line_id_rev_weak = orig_line_id[weak_line_idx], orig_line_wave[weak_line_idx], orig_line_id_rev[weak_line_idx]

tot_line_wave = np.append(line_wave, line_wave_weak)
tot_line_id = np.append(line_id, line_id_weak)
tot_line_name = np.append(line_id_rev, line_id_rev_weak)



if os.path.exists(tmp_bic_file):
    bic_list, chi2_list, fit_results = base_func.load_pickle(tmp_bic_file)
    print (f"{tmp_bic_file} loaded")
else:
    print (f"{tmp_bic_file} not found. Quitting")
    quit()



# Find best BIC
best_N = np.argmin(bic_list) + 1
print(f"\n*** Optimal number of components = {best_N} ***")
# Plot final best model
best_popt = fit_results[best_N - 1][0]
best_perr = fit_results[best_N - 1][1]
#best_model_flux = wrapped_model(wave, *best_popt)

#orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit
flux_noisy = orig_flux_noisy - orig_continuum
tmp_wave_strong, tmp_flux_noisy_strong, tmp_flux_err_strong, tmp_continuum_strong = base_func.obtain_fittable_area(orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, line_wave, allowable_velocity_range=fig_vel_lim)
flux_noisy_strong = tmp_flux_noisy_strong - tmp_continuum_strong
tmp_wave_weak, tmp_flux_noisy_weak, tmp_flux_err_weak, tmp_continuum_weak = base_func.obtain_fittable_area(orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, tot_line_wave, allowable_velocity_range=fig_vel_lim)
flux_noisy_weak = tmp_flux_noisy_weak - tmp_continuum_weak



#'''
if os.path.exists(tmp_dynesty_file):
    user_input = input(f"'{tmp_dynesty_file}' already exists. Load previous results? (y/n): ").strip().lower()
    if user_input == 'y':
        print("Loading existing results...")
        sampler, component_mcmc_info_dict = base_func.load_pickle(tmp_dynesty_file)
        print (f"{tmp_dynesty_file} loaded")
    else:
        sampler, component_mcmc_info_dict = dynesty_mcmc_func.mcmc_fit_dynesty(tmp_wave_strong, flux_noisy_strong, tmp_flux_err_strong, N_components=best_N, magic_number=magic_number, line_wave=line_wave, line_name=line_id, nwalkers = dynesty_nwalkers, nsteps=dynesty_nsteps, discard_val=dynesty_discard_val, thin_val=dynesty_thin_val, delta_v_kms=fig_vel_lim, corner_plot=False, popt=best_popt, cornerplot_figname=dynesty_corner_fig, dlogz=dlogz_mcmc)
        tmp_objects = [sampler, component_mcmc_info_dict]
        base_func.save_pickle(tmp_dynesty_file, *tmp_objects)
        print (f"{tmp_dynesty_file} saved")
else:
    sampler, component_mcmc_info_dict = dynesty_mcmc_func.mcmc_fit_dynesty(tmp_wave_strong, flux_noisy_strong, tmp_flux_err_strong, N_components=best_N, magic_number=magic_number, line_wave=line_wave, line_name=line_id, nwalkers = dynesty_nwalkers, nsteps=dynesty_nsteps, discard_val=dynesty_discard_val, thin_val=dynesty_thin_val, delta_v_kms=fig_vel_lim, corner_plot=False, popt=best_popt, cornerplot_figname=dynesty_corner_fig, dlogz=dlogz_mcmc)
    tmp_objects = [sampler, component_mcmc_info_dict]
    base_func.save_pickle(tmp_dynesty_file, *tmp_objects)
    print (f"{tmp_dynesty_file} saved")
#'''

_ = dynesty_mcmc_func.plot_dynesty_stamp(orig_wave, flux_noisy, orig_flux_err, sampler, magic_number, line_wave=line_wave, line_name=line_id, figname=dynesty_stamp_fig, vel_lim=fig_vel_lim)

#_ = base_func.print_results(component_mcmc_info_dict, best_N, tot_original_flux=None, line_id=line_id, line_wave=line_wave)
_ = base_func.print_results_rev(component_mcmc_info_dict, best_N, line_id=line_id, line_wave=line_wave)

_ = base_func.checking_results_plot(component_mcmc_info_dict, tot_original_flux=None, line_id=line_id, line_wave=line_wave, figname=checking_dynesty_res_fig)

line_val = []
component_info_rev, popt_weak, perr_weak = base_func.load_pickle(tmp_weak_file)
for i in range(len(line_id)):
    tot_flux_str = 'tot_flux_'+line_id[i]+str(int(line_wave[i]))
    tot_flux_unc_str = 'tot_flux_unc_'+line_id[i]+str(int(line_wave[i]))
    plt.errorbar(component_info_rev[tot_flux_str], component_mcmc_info_dict[tot_flux_str], xerr=component_info_rev[tot_flux_unc_str], yerr=component_mcmc_info_dict[tot_flux_unc_str], fmt='o')
    line_val.append(component_info_rev[tot_flux_str])
    line_val.append(component_mcmc_info_dict[tot_flux_str])

line = np.linspace(np.nanmin(line_val), np.nanmax(line_val), 100)
plt.plot(line, line, 'g--')
plt.savefig(f'{checking_dynesty_res_fig3}', dpi=100)
plt.close('all')
print (f'{checking_dynesty_res_fig3} saved')


# Fit Weak lines with fixed mu and sigma
if os.path.exists(tmp_weak_file):
    component_info_rev, popt_weak, perr_weak = base_func.load_pickle(tmp_weak_file)
    print (f"{tmp_weak_file} loaded")
else:
    print (f"{tmp_weak_file} not found. Quitting.")
    quit()


if os.path.exists(tmp_dynesty_file_weak):
    user_input = input(f"'{tmp_dynesty_file_weak}' already exists. Load previous results? (y/n): ").strip().lower()
    if user_input == 'y':
        print("Loading existing results...")
        sampler_weak, component_mcmc_info_dict_weak = base_func.load_pickle(tmp_dynesty_file_weak)
        print (f"{tmp_dynesty_file_weak} loaded")
    else:
        sampler_weak, component_mcmc_info_dict_weak = dynesty_mcmc_func.mcmc_fit_dynesty_amplitudes(tmp_wave_weak, flux_noisy_weak, tmp_flux_err_weak, best_N, component_mcmc_info_dict, line_wave = tot_line_wave, line_name = tot_line_id, popt=popt_weak, corner_plot=False, cornerplot_figname=dynesty_corner_fig2, delta_v_kms=fig_vel_lim, dlogz=dlogz_mcmc)
        tmp_objects_weak = [sampler_weak, component_mcmc_info_dict_weak]
        base_func.save_pickle(tmp_dynesty_file_weak, *tmp_objects_weak)
        print (f"{tmp_dynesty_file_weak} saved")
else:
    sampler_weak, component_mcmc_info_dict_weak = dynesty_mcmc_func.mcmc_fit_dynesty_amplitudes(tmp_wave_weak, flux_noisy_weak, tmp_flux_err_weak, best_N, component_mcmc_info_dict, line_wave = tot_line_wave, line_name = tot_line_id, popt=popt_weak, corner_plot=False, cornerplot_figname=dynesty_corner_fig2, delta_v_kms=fig_vel_lim, dlogz=dlogz_mcmc)
    tmp_objects_weak = [sampler_weak, component_mcmc_info_dict_weak]
    base_func.save_pickle(tmp_dynesty_file_weak, *tmp_objects_weak)
    print (f"{tmp_dynesty_file_weak} saved")

_ = dynesty_mcmc_func.plot_weak_line_stamp_dynesty(orig_wave, flux_noisy, orig_flux_err, sampler_weak, component_mcmc_info_dict_weak, best_N, figname=dynesty_stamp_fig2, tot_line_wave=tot_line_wave, tot_line_name=tot_line_id, line_wave=line_wave_weak, line_name=line_id_weak, vel_lim=fig_vel_lim, fontsize=20.)

#_ = base_func.print_results(component_mcmc_info_dict_weak, best_N, tot_original_flux=None, line_id=line_id_weak, line_wave=line_wave_weak)

_ = base_func.print_results_rev(component_mcmc_info_dict_weak, best_N, line_id=line_id_weak, line_wave=line_wave_weak)

_ = base_func.checking_results_plot(component_mcmc_info_dict_weak, tot_original_flux=None, line_id=line_id_weak, line_wave=line_wave_weak, figname=checking_dynesty_res_fig2)

line_val = []
component_info_rev, popt_weak, perr_weak = base_func.load_pickle(tmp_weak_file)
for i in range(len(line_id_weak)):
    tot_flux_str = 'tot_flux_'+line_id_weak[i]+str(int(line_wave_weak[i]))
    tot_flux_unc_str = 'tot_flux_unc_'+line_id_weak[i]+str(int(line_wave_weak[i]))
    plt.errorbar(component_info_rev[tot_flux_str], component_mcmc_info_dict_weak[tot_flux_str], xerr=component_info_rev[tot_flux_unc_str], yerr=component_mcmc_info_dict_weak[tot_flux_unc_str], fmt='o')
    line_val.append(component_info_rev[tot_flux_str])
    line_val.append(component_mcmc_info_dict_weak[tot_flux_str])

line = np.linspace(np.nanmin(line_val), np.nanmax(line_val), 100)
plt.plot(line, line, 'g--')
plt.savefig(f'{checking_dynesty_res_fig4}', dpi=100)
plt.close('all')
print (f'{checking_dynesty_res_fig4} saved')



os.system('say "Task complete"')  # macOS
os.system('say "Task complete"')  # macOS
os.system('say "Task complete"')  # macOS

