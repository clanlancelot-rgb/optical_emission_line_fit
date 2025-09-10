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


def get_continuum_from_points(x, coordsx, coordsy):
    points = zip(coordsx, coordsy)
    points = sorted(points, key=lambda point: point[0])
    x1, y1 = zip(*points)
    #new_length = len(x)
    # Find unique elements in x
    unique_indices = np.unique(x1, return_index=True)[1]
    # Update x and y based on unique elements
    new_x1_tmp = np.array([x1[i] for i in sorted(unique_indices)])
    new_y1_tmp = np.array([y1[i] for i in sorted(unique_indices)])
    #new_length = len(x)
    l1 = np.searchsorted(x, min(new_x1_tmp))
    l2 = np.searchsorted(x, max(new_x1_tmp))
    new_x = []
    new_y = []
    cont = np.zeros([(len(x))])
    if (len(x1)>3):
        new_x = np.linspace(min(new_x1_tmp), max(new_x1_tmp), (l2-l1))
        new_y = sp.interpolate.splrep(new_x1_tmp, new_y1_tmp)
        cont = sp.interpolate.splev(x, new_y, der=0)
    return (x1, y1, cont)

#x1, y1, cont = get_continuum_from_points(x, coordsx, coordsy)

######################KEY_PRESS_EVENTS##################################
start_point = []
end_point = []

def on_key(event):
    global coordsx, coordsy, cont, x1, y1, ix, iy, ix2, iy2, ix3, iy3, x
    if ((event.key).lower() == 'a'):
        ix, iy = event.xdata, event.ydata
        coordsx = np.append(coordsx, (ix))
        coordsy = np.append(coordsy, (iy))

    elif ((event.key).lower() == 'r'):
        ix2, iy2 = event.xdata, event.ydata
        ixnew = base_func.find_nearest(coordsx, ix2, return_idx=False)
        ixnew2 = np.where(coordsx==ixnew)
        coordsx = np.delete(coordsx, ixnew2)
        coordsy = np.delete(coordsy, ixnew2)

    elif ((event.key).lower() == 'm'):
        ix3, iy3 = event.xdata, event.ydata
        ixnew3 = base_func.find_nearest(coordsx, ix3, return_idx=False)
        ixnew4 = np.where(coordsx==ixnew3)
        coordsx[ixnew4] = ix3
        coordsy[ixnew4] = iy3

    elif ((event.key).lower() == 'u'):
        if not (len(start_point) > len(end_point)):
            start_point.append(event.xdata)

    elif ((event.key).lower() == 'i'):
        if not (len(end_point) >= len(start_point)):
            end_point.append(event.xdata)
            if end_point[-1]<start_point[-1]:
                print ("Warning!! You did something wrong. The end point in smaller than the start point")
                del start_point[-1]
                del end_point[-1]

    x1, y1, cont = get_continuum_from_points(x, coordsx, coordsy)
    
    line6.set_data(x1, y1)
    line2.set_data(x, cont)
    fig.canvas.draw()

    #Disconnect if you press 'x'
    if event.key == 'x':
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)

    return

def press(event):
    print('press', event.key)
    sys.stdout.flush()

######################KEY_PRESS_EVENTS##################################


cluster_name = str(config['cluster_id'])
z = float(config['z'])  # Actual redshift of the galaxy
vel_window = float(config['vel_window'])  # Velocity window for fitting
number_of_components = int(config['number_of_components'])
flux_red = float(config['flux_red'])
print ('\n')
print (f'Cluster: {cluster_name}')
print ('\n')
norm_file_orig = str(config['original_spectra_address']) + str(config['cluster_full_name'])
cont_file_name = cwd + f'continuum_points_{cluster_name}_rev.dat'
cont_file = cwd + f'cont_{cluster_name}_rev.dat'
this_file_extension = config['this_file_extension']
fig_vel_lim = config['fig_vel_lim']
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
dynesty_corner_fig = f'corner_dynesty_{this_file_extension}.pdf'
dynesty_corner_fig2 = f'corner_dynesty2_{this_file_extension}.pdf'
dynesty_stamp_fig = f'stamp_dynesty_{this_file_extension}.pdf'
dynesty_stamp_fig2 = f'stamp_dynesty2_{this_file_extension}.pdf'
checking_dynesty_res_fig = f'checking_results_dynesty_{this_file_extension}.pdf'
checking_dynesty_res_fig2 = f'checking_results_dynesty2_{this_file_extension}.pdf'
emcee_corner_fig = f'corner_mcmc_{this_file_extension}.pdf'
emcee_stamp_plot = f'stamp_emcee_{this_file_extension}.pdf'

line_list_file = np.genfromtxt(f"{codewd}{config['emission_line_list']}", delimiter=',', encoding=None, dtype=None, names=True)
#orig_wave, orig_flux_noisy, orig_flux_err, blah  = np.genfromtxt(f'{norm_file_orig}', unpack=True)
orig_wave, orig_flux_noisy, orig_flux_err, *blah = np.genfromtxt(norm_file_orig, unpack=True, ndmin=2)
orig_wave = orig_wave/(1.+z)
orig_flux_noisy = orig_flux_noisy/1e-16
orig_flux_err = orig_flux_err/1e-16

orig_line_id, orig_line_wave, orig_line_id_rev = base_func.obtain_lines(orig_wave, line_list_file)


if config['instrument']=='MUSE':
    strong_lines = np.array(['Hβ4861', '[O_III]4958', '[O_III]5006', '[N_II]6548', 'Hα6562', '[N_II]6583', '[S_II]6716', '[S_II]6730', '[S_III]9068'])
    #Removing [Fe_IV]6739 because it is affecting the SII fit
    #Removing '[Ar_IV]4740', '[Ar_IV]4711' for later
    weak_lines = np.array(['[Fe_III]4658', '[Fe_III]4701', '[Ar_IV]4740', '[Ar_IV]4711', '[Ar_III]5191', '[Cl_III]5517', '[Cl_III]5537', '[N_II]5754', '[S_III]6312', '[Fe_IV]6739', '[Ar_III]7135', '[O_II]7320', '[O_II]7329', '[Fe_II]8616', '[Fe_II]8891'])
    #weak_lines = np.array(['[Fe_III]4658', '[Fe_III]4701', '[Ar_III]5191', '[Cl_III]5517', '[Cl_III]5537', '[N_II]5754', '[S_III]6312', '[Ar_III]7135', '[O_II]7320', '[O_II]7329', '[Fe_II]8616', '[Fe_II]8891'])

elif config['instrument']=='LBT-BLUE':
    tmp_limit = 5490.
    orig_wave = orig_wave*(1.+z)
    orig_flux_noisy = orig_flux_noisy[(orig_wave<tmp_limit)]
    orig_flux_err = orig_flux_err[(orig_wave<tmp_limit)]
    orig_wave = orig_wave[(orig_wave<tmp_limit)]
    strong_lines = np.array(['[O_II]3726', '[O_II]3728', 'Hβ4861', '[O_III]4958', '[O_III]5006'])
    #Removing '[Ar_IV]4740', '[Ar_IV]4711' for later
    weak_lines = np.array(['[Fe_V]4227', '[O_III]4363', '[Fe_III]4658', '[Fe_III]4701', '[Ar_IV]4740', '[Ar_IV]4711', '[Ar_III]5191'])
    #weak_lines = np.array(['[Fe_V]4227', '[O_III]4363', '[Fe_III]4658', '[Fe_III]4701', '[Ar_III]5191'])

elif config['instrument']=='LBT-RED':
    tmp_limit = 5490.
    orig_wave = orig_wave*(1.+z)
    orig_flux_noisy = orig_flux_noisy[(orig_wave>tmp_limit)]
    orig_flux_err = orig_flux_err[(orig_wave>tmp_limit)]
    orig_wave = orig_wave[(orig_wave>tmp_limit)]
    strong_lines = np.array(['[N_II]6548', 'Hα6562', '[N_II]6583', '[S_II]6716', '[S_II]6730', '[S_III]9068', '[S_III]9530'])
    #Removing [Fe_IV]6739 because it is affecting the SII fit
    #weak_lines = np.array(['[Cl_III]5517', '[Cl_III]5537', '[N_II]5754', '[S_III]6312', '[Ar_III]7135', '[O_II]7320', '[O_II]7329', '[Fe_II]8616', '[Fe_II]8891'])
    weak_lines = np.array(['[Cl_III]5517', '[Cl_III]5537', '[N_II]5754', '[S_III]6312', '[Fe_IV]6739', '[Ar_III]7135', '[O_II]7320', '[O_II]7329', '[Fe_II]8616', '[Fe_II]8891'])



strong_line_idx = np.where(np.isin(orig_line_id_rev, strong_lines))[0]
line_id, line_wave, line_id_rev = orig_line_id[strong_line_idx], orig_line_wave[strong_line_idx], orig_line_id_rev[strong_line_idx]

weak_line_idx = np.where(np.isin(orig_line_id_rev, weak_lines))[0]
line_id_weak, line_wave_weak, line_id_rev_weak = orig_line_id[weak_line_idx], orig_line_wave[weak_line_idx], orig_line_id_rev[weak_line_idx]

tot_line_wave = np.append(line_wave, line_wave_weak)
tot_line_id = np.append(line_id, line_id_weak)
tot_line_name = np.append(line_id_rev, line_id_rev_weak)

#line_wave_weak = np.append(line_wave, line_wave_weak)
#line_id_weak = np.append(line_id, line_id_weak)
#line_id_rev_weak = np.append(line_id_rev, line_id_rev_weak)

if os.path.exists(file_with_cont):
    user_input = input(f"'{file_with_cont}' already exists. Load previous results? (y/n): ").strip().lower()
    if user_input == 'y':
        print("Loading existing results...")
        orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit = np.genfromtxt(file_with_cont, delimiter=',', unpack=True, encoding=None, dtype=np.float32)
        print (f"{file_with_cont} loaded")
        if (os.path.exists(cont_file_name)):
            print (f"{cont_file_name.split('/')[-1]} also found. Making continuum from scratch.")
            coordsx, coordsy = np.loadtxt(cont_file_name, unpack=True)
            x1, y1, orig_continuum = get_continuum_from_points(orig_wave, coordsx, coordsy)
    else:
        orig_continuum = base_func.get_automated_continuum(orig_wave, orig_flux_noisy)
        tmp_prof_to_use = base_func.get_automated_fit(orig_wave, orig_flux_noisy, tot_line_wave, number_of_components_tmp = 1, vel_window_tmp = 2000.)
        orig_fit = tmp_prof_to_use+orig_continuum
        orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit = base_func.clean_spectrum(orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit)
        tmp_array = np.transpose(np.array([orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit]))
        np.savetxt(file_with_cont, tmp_array, delimiter=',')
        print (f"{file_with_cont} saved")
        indices = np.linspace(0, len(orig_wave)-1, 50).astype(int)
        coordsx = np.array(orig_wave[indices])
        coordsy = np.array(orig_continuum[indices])
        points = zip(coordsx, coordsy)
        points = sorted(points, key=lambda point: point[0])
        x1, y1 = zip(*points)

else:
    orig_continuum = base_func.get_automated_continuum(orig_wave, orig_flux_noisy)
    tmp_prof_to_use = base_func.get_automated_fit(orig_wave, orig_flux_noisy, tot_line_wave, number_of_components_tmp = 1, vel_window_tmp = 2000.)
    orig_fit = tmp_prof_to_use+orig_continuum
    orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit = base_func.clean_spectrum(orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit)
    tmp_array = np.transpose(np.array([orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit]))
    np.savetxt(file_with_cont, tmp_array, delimiter=',')
    print (f"{file_with_cont} saved")
    indices = np.linspace(0, len(orig_wave)-1, 50).astype(int)
    coordsx = np.array(orig_wave[indices])
    coordsy = np.array(orig_continuum[indices])
    points = zip(coordsx, coordsy)
    points = sorted(points, key=lambda point: point[0])
    x1, y1 = zip(*points)
    if (os.path.exists(cont_file_name)):
        print (f"{cont_file_name.split('/')[-1]} found. Making continuum from scratch.")
        coordsx, coordsy = np.loadtxt(cont_file_name, unpack=True)
        x1, y1, orig_continuum = get_continuum_from_points(orig_wave, coordsx, coordsy)





#wave_to_use, flux_to_use, flux_unc_to_use, cont_to_use, prof_to_use = orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit

wave_strong, flux_noisy_strong, flux_err_strong, continuum_strong = base_func.obtain_fittable_area(orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, line_wave, allowable_velocity_range=allowable_velocity_range)
wave_weak, flux_noisy_weak, flux_err_weak, continuum_weak = base_func.obtain_fittable_area(orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, line_wave_weak, allowable_velocity_range=allowable_velocity_range)
init_fittable_wave = np.append(wave_strong, wave_weak)
init_fittable_flux = np.append(flux_noisy_strong, flux_noisy_weak)


#Plotting
fig, (ax) = plt.subplots(nrows=1, figsize=(10, 6))  # The figsize argument controls the figure size

line1, = ax.plot(orig_wave, orig_flux_noisy, 'k-', alpha=0.25, drawstyle='steps-mid', label='data')
plt.title(cluster_name)
line6, = ax.plot(x1,y1, 'yo', label='cont-points')
line2, = ax.plot(orig_wave,orig_continuum, 'g--', label='cont')
cid = fig.canvas.mpl_connect('key_press_event', on_key)
l, = ax.plot(orig_wave, orig_fit, 'r', visible=True)
line8, = ax.plot(init_fittable_wave, init_fittable_flux, 'ko', markersize=2)
line3, = ax.plot([],[], 'co', markersize=1, alpha=0.)
species_ID = orig_line_id_rev
emission_line_waveslength = orig_line_wave
global x, y
x = orig_wave
y = orig_flux_noisy


x_axis_tmp = []
y_axis_tmp = []
x_axis_for_text_tmp = []
y_axis_for_text_tmp = []
str_for_text_tmp = []

y_axis_for_plot = list(np.linspace(np.nanmin(y), np.nanmax(y), len(y)))
for i in range(len(tot_line_wave)):
    if (np.nanmin(x) < tot_line_wave[i]*(1.+0.0) < np.nanmax(x)):
        x_axis_for_plot = list(np.full_like(y_axis_for_plot, float(emission_line_waveslength[i]*(1.+0.0))))
        y_axis_tmp.extend(y_axis_for_plot)
        x_axis_tmp.extend(x_axis_for_plot)
        x_axis_idx_tmp = base_func.find_nearest(x, emission_line_waveslength[i], return_idx=True)
        x_axis_for_text_tmp.append(x[x_axis_idx_tmp])
        y_axis_for_text_tmp.append(float(y[x_axis_idx_tmp]+0.1))
        str_for_text_tmp.append(str(orig_line_id_rev[i]))
line3.set_data(x_axis_tmp, y_axis_tmp)


texts3 = [ax.text(x_axis_for_text_tmp[i], y_axis_for_text_tmp[i], f'{str_for_text_tmp[i]}', color='black', visible=False) for i, pos in enumerate(str_for_text_tmp)]

def toggle_text_visibility3(label3, text_index3):
    texts3[text_index3].set_visible(not texts3[text_index3].get_visible())
    plt.draw()


# Create CheckButtons with a checkbox for each text
rax = plt.axes([0.91, 0.8, 0.08, 0.04])
checkbox_labels = [f'names']
checkbox_states3 = [False] * len(texts3)
check3 = CheckButtons(rax, checkbox_labels, checkbox_states3)

# Attach the callback function to the checkboxes using a default argument
for i in range(len(texts3)):
    check3.on_clicked(lambda event, i=i: toggle_text_visibility3(event, i))

rax = plt.axes([0.91, 0.6, 0.08, 0.15])
check = CheckButtons(rax, ('data', 'fit', 'cont', 'lines'), (True, True, True, True))
def func(label):
    if label == 'data':
        line1.set_visible(not line1.get_visible())
    #elif label == 'model':
        #line5.set_visible(not line5.get_visible())
    elif label == 'fit':
        l.set_visible(not l.get_visible())
    elif label == 'cont':
        line2.set_visible(not line2.get_visible())
        line6.set_visible(not line6.get_visible())
    elif label == 'lines':
        line3.set_visible(not line3.get_visible())
        #line4.set_visible(not line4.get_visible())
    fig.canvas.draw()
check.on_clicked(func)



#DEFINITION FOR THE 4 POINTS ON PLT.AXES
#([1 - SIGNIFIES WHERE THE BAR WILL START ON THE LEFT (0.0 - EXTREME LEFT, 1.0 - EXTREME RIGHT(OUT OF SIGHT))])
#([2 - SIGNIFIES WHERE THE BAR WILL START FROM THE BOTTOM (0.0 - EXTREME BOTTOM, 1.0 - EXTREME TOP(OUT OF SIGHT))])
#([3 - LENGTH OF THE BAR (0.0 - NO LENGTH, 1.0 - FULL MATPLOTLIB WINDOW])
#([4 - WIDTH OF THE BAR (0.0 - ERROR(NO-WIDTH), 0.05 - FAT BAR, 0.01/0.02 - IDEAL VALUES])

axcolor = 'lightgoldenrodyellow'
#axqso = plt.axes([0.15, 0.03, 0.1, 0.02])
#sqso = Slider(axqso, 'z_QSO', (0.0), (z_qso+1.0), valinit=z_qso)

axemission_str = plt.axes([0.08, 0.03, 0.08, 0.02])
semission_str = Slider(axemission_str, 'amplitude', (0.0), (np.nanmax(y)), valinit=np.nanmedian(y)*0.5)

axwind_vel = plt.axes([0.36, 0.03, 0.08, 0.02])
swind_vel = Slider(axwind_vel, 'wind_vel', (1.0), (2000.0), valinit=100.0)

#axredshift = plt.axes([0.42, 0.03, 0.08, 0.02])
#sredshift = Slider(axredshift, 'z_gal', (z-0.1), (z+0.1), valinit=0.0)

axvel_shift = plt.axes([0.62, 0.03, 0.08, 0.02])
svel_shift = Slider(axvel_shift, 'vel_shift', (-vel_window/5.), (vel_window/5.), valinit=0.0)

#global data_smooth
data_smooth = 1.0
axdata_smooth = plt.axes([0.80, 0.03, 0.08, 0.02])
sdata_smooth = Slider(axdata_smooth, 'Smooth', (data_smooth+0), (data_smooth+101), valinit=data_smooth)

axnum_comp = plt.axes([0.80, 0.92, 0.08, 0.02])
snum_comp = Slider(axnum_comp, 'components', (0), (10), valinit=1)

axvel_window = plt.axes([0.18, 0.92, 0.08, 0.02])
svel_window = Slider(axvel_window, 'vel_window', (50), (5000), valinit=2000)

def update(val):
    global cont, x1, y1, x, y, coordsx, coordsy, orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit
    try:
        cont = cont
        line6.set_xdata(x1)
        line6.set_ydata(y1)
        line2.set_ydata(cont)
    except:
        cont = orig_continuum

    number_of_components_tmp = int(snum_comp.val)
    vel_window_tmp = float(svel_window.val)
    v_shifts = np.array([float(svel_shift.val)]*number_of_components_tmp)  # Wind velocity in km/s
    sig_shifts = np.array([float(swind_vel.val)]*number_of_components_tmp)  # Wind velocity in km/s
    amps = np.array([float(semission_str.val)]*(number_of_components_tmp*len(tot_line_wave)))
    tmp_prof_to_use = base_func.model_amplitudes_only(orig_wave, *amps, line_wave=tot_line_wave, v_shifts=v_shifts, sigma_vs=sig_shifts)
    orig_fit = tmp_prof_to_use+cont
    line1.set_ydata(base_func.smooth(y, int(sdata_smooth.val)))
    l.set_ydata(base_func.smooth(orig_fit, int(sdata_smooth.val)))

    tmp_wave_strong, tmp_flux_noisy_strong, flux_err_strong, continuum_strong = base_func.obtain_fittable_area(orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, line_wave, allowable_velocity_range=vel_window_tmp)
    tmp_wave_weak, tmp_flux_noisy_weak, flux_err_weak, continuum_weak = base_func.obtain_fittable_area(orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, line_wave_weak, allowable_velocity_range=vel_window_tmp)
    tmp_init_fittable_wave = np.append(tmp_wave_strong, tmp_wave_weak)
    tmp_init_fittable_flux = np.append(tmp_flux_noisy_strong, tmp_flux_noisy_weak)
    line8.set_xdata(tmp_init_fittable_wave)
    line8.set_ydata(tmp_init_fittable_flux)

fig.canvas.draw_idle()
semission_str.on_changed(update)
swind_vel.on_changed(update)
svel_shift.on_changed(update)
sdata_smooth.on_changed(update)
snum_comp.on_changed(update)
svel_window.on_changed(update)




################SAVE_AND_LOAD_CONTINUUM_POINTS####################

save_cont_ax = plt.axes([0.91, 0.45, 0.08, 0.04])
button2 = Button(save_cont_ax, 'save_data', color=axcolor, hovercolor='0.975')
def save_cont_ax(event):
    #global cont, x1, y1, x, y, coordsx, coordsy, prof_to_use
    global cont, x1, y1, x, y, coordsx, coordsy, orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit
    try:
        cont = cont
        cont_points = np.transpose(np.array([coordsx, coordsy]))
        np.savetxt(cont_file_name, cont_points)
        print ("continuum points saved")
    except:
        cont = orig_continuum

    if os.path.exists(file_with_cont):
        user_input = input(f"'{file_with_cont}' already exists. Replace? (y/n): ").strip().lower()
        if user_input=='y':
            orig_cust_continuum = cont
            tmp_array = np.transpose(np.array([orig_wave, orig_flux_noisy, orig_flux_err, orig_cust_continuum, orig_fit]))
            np.savetxt(file_with_cont, tmp_array, delimiter=',')
            print (f"{file_with_cont} saved")
        else:
            print (f"{file_with_cont} unchanged")
    else:
        orig_cust_continuum = cont
        tmp_array = np.transpose(np.array([orig_wave, orig_flux_noisy, orig_flux_err, orig_cust_continuum, orig_fit]))
        np.savetxt(file_with_cont, tmp_array, delimiter=',')
        print (f"{file_with_cont} saved")

button2.on_clicked(save_cont_ax)


get_cont_ax = plt.axes([0.91, 0.4, 0.08, 0.04])
button3 = Button(get_cont_ax, 'get_cont', color=axcolor, hovercolor='0.975')
def get_cont_ax(event):
    global x
    try:
        #cont_points = np.loadtxt(cont_file_name)
        coordsx, coordsy = np.loadtxt(cont_file_name, unpack=True)
        print ("continuum points loaded")
    except:
        print ("Failed")

    x1, y1, cont = get_continuum_from_points(x, coordsx, coordsy)

    line6.set_data(x1, y1)
    line2.set_data(x, cont)
    fig.canvas.draw()
button3.on_clicked(get_cont_ax)

################SAVE_AND_LOAD_CONTINUUM_POINTS####################




fit_scipy_ax = plt.axes([0.91, 0.35, 0.08, 0.04])
button4 = Button(fit_scipy_ax, 'fit_data', color=axcolor, hovercolor='0.975')
def fit_scipy_ax(event):
    print ("Fitting using curve fit...")

    global cont, x1, y1, x, y, coordsx, coordsy, orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit
    try:
        cont = cont
        cont_points = np.transpose(np.array([coordsx, coordsy]))
        np.savetxt(cont_file_name, cont_points)
        print ("continuum points saved")
    except:
        cont = orig_continuum

    magic_number = len(line_wave)+2
    vel_window_tmp = float(svel_window.val)
    tmp_wave_strong, tmp_flux_noisy_strong, tmp_flux_err_strong, tmp_continuum_strong = base_func.obtain_fittable_area(orig_wave, orig_flux_noisy, orig_flux_err, cont, line_wave, allowable_velocity_range=vel_window_tmp)
    number_of_components_tmp = int(snum_comp.val)
    flux_noisy_strong = tmp_flux_noisy_strong - tmp_continuum_strong
    tmp_popt, tmp_perr = curvefit_func.curvefit_method(tmp_wave_strong, flux_noisy_strong, tmp_flux_err_strong, number_of_components_tmp, line_wave=line_wave, v_init=float(svel_shift.val), v_lower=-250.0, v_upper=250.0, sigma_init=float(swind_vel.val), sigma_lower=1.0, sigma_upper=1500.0)
    # Wrap the function
    wrapped_model = lambda wave, *theta: base_func.model_curvefit(
        wave, *theta, line_wave=line_wave, magic_number=magic_number)
    tmp_prof_to_use_strong = wrapped_model(orig_wave, *tmp_popt)

    tmp_v_shifts, tmp_sig_shifts = curvefit_func.get_vel_sigma_from_popt(tmp_popt, number_of_components_tmp, line_wave)
    tmp_wave_weak, tmp_flux_noisy_weak, tmp_flux_err_weak, tmp_continuum_weak = base_func.obtain_fittable_area(orig_wave, orig_flux_noisy, orig_flux_err, cont, tot_line_wave, allowable_velocity_range=vel_window_tmp)
    flux_noisy_weak = tmp_flux_noisy_weak - tmp_continuum_weak
    #tmp_popt_amp, tmp_perr_amp = curvefit_func.curvefit_amp_method(tmp_wave_weak, flux_noisy_weak, tmp_flux_err_weak, tmp_v_shifts, tmp_sig_shifts, line_wave_weak, number_of_components_tmp, amp_val_init=0.0, amp_val_min=0.0, amp_val_max=np.nanmax(tmp_flux_noisy_weak))
    tmp_popt_amp, tmp_perr_amp = curvefit_func.curvefit_amp_method(tmp_wave_weak, flux_noisy_weak, tmp_flux_err_weak, tmp_v_shifts, tmp_sig_shifts, tot_line_wave, number_of_components_tmp, amp_val_init=0.0, amp_val_min=0.0, amp_val_max=np.nanmax(flux_noisy_weak))

    # Wrapped model for curve_fit
    wrapped_model_amplitudes_curvefit = lambda wave, *theta: base_func.model_amplitudes_only(wave, *theta, line_wave=tot_line_wave, v_shifts=tmp_v_shifts, sigma_vs=tmp_sig_shifts)
    tmp_prof_to_use_weak = wrapped_model_amplitudes_curvefit(orig_wave, *tmp_popt_amp)

    #orig_fit = tmp_prof_to_use_strong + tmp_prof_to_use_weak + cont
    orig_fit = tmp_prof_to_use_weak + cont

    l.set_ydata(base_func.smooth((orig_fit), int(sdata_smooth.val)))
    fig.canvas.draw()

    if os.path.exists(file_with_cont):
        user_input = input(f"'{file_with_cont}' already exists. Replace? (y/n): ").strip().lower()
        if user_input=='y':
            orig_cust_continuum = cont
            tmp_array = np.transpose(np.array([orig_wave, orig_flux_noisy, orig_flux_err, orig_cust_continuum, orig_fit]))
            np.savetxt(file_with_cont, tmp_array, delimiter=',')
            print (f"{file_with_cont} saved")
        else:
            print (f"{file_with_cont} unchanged")
    else:
        orig_cust_continuum = cont
        tmp_array = np.transpose(np.array([orig_wave, orig_flux_noisy, orig_flux_err, orig_cust_continuum, orig_fit]))
        np.savetxt(file_with_cont, tmp_array, delimiter=',')
        print (f"{file_with_cont} saved")

button4.on_clicked(fit_scipy_ax)



update_ax = plt.axes([0.91, 0.30, 0.08, 0.04])
button5 = Button(update_ax, 'update', color=axcolor, hovercolor='0.975')
def update_ax(event):
    print ("Updating profile...")
    global cont, x1, y1, x, y, coordsx, coordsy, orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit
    try:
        cont = cont
        cont_points = np.transpose(np.array([coordsx, coordsy]))
        np.savetxt(cont_file_name, cont_points)
        print ("continuum points saved")
    except:
        cont = orig_continuum

    orig_fit = orig_fit - orig_continuum + cont
    l.set_ydata(base_func.smooth((orig_fit), int(sdata_smooth.val)))
    fig.canvas.draw()
button5.on_clicked(update_ax)





bic_ax = plt.axes([0.91, 0.20, 0.08, 0.04])
button6 = Button(bic_ax, 'BIC_fit', color=axcolor, hovercolor='0.975')
def bic_ax(event):
    print ("Implementing BIC Method...")

    global cont, x1, y1, x, y, coordsx, coordsy, orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit
    try:
        cont = cont
        cont_points = np.transpose(np.array([coordsx, coordsy]))
        np.savetxt(cont_file_name, cont_points)
        print ("continuum points saved")
    except:
        cont = orig_continuum

    magic_number = len(line_wave)+2
    vel_window_tmp = float(svel_window.val)
    tmp_wave_strong, tmp_flux_noisy_strong, tmp_flux_err_strong, tmp_continuum_strong = base_func.obtain_fittable_area(orig_wave, orig_flux_noisy, orig_flux_err, cont, line_wave, allowable_velocity_range=vel_window_tmp)
    number_of_components_tmp = int(snum_comp.val)
    flux_noisy_strong = tmp_flux_noisy_strong - tmp_continuum_strong

    if os.path.exists(tmp_bic_file):
        user_input = input(f"'{tmp_bic_file}' already exists. Load previous results? (y/n): ").strip().lower()
        if user_input == 'y':
            print("Loading existing results...")
            bic_list, chi2_list, fit_results = base_func.load_pickle(tmp_bic_file)
            print (f"{tmp_bic_file} loaded")
        else:
            bic_list, chi2_list, fit_results = curvefit_func.BIC_fitting_method(tmp_wave_strong, flux_noisy_strong, tmp_flux_err_strong, magic_number, N_max = N_max, penalty_factor = penalty_factor, line_wave=line_wave, v_init=float(svel_shift.val), v_lower=curvefit_v_lower, v_upper=curvefit_v_upper, sigma_init=float(swind_vel.val), sigma_lower=curvefit_sigma_lower, sigma_upper=curvefit_sigma_upper)
            tmp_objects = [bic_list, chi2_list, fit_results]
            base_func.save_pickle(tmp_bic_file, *tmp_objects)
            print (f"{tmp_bic_file} saved")
    else:
        bic_list, chi2_list, fit_results = curvefit_func.BIC_fitting_method(tmp_wave_strong, flux_noisy_strong, tmp_flux_err_strong, magic_number, N_max = N_max, penalty_factor = penalty_factor, line_wave=line_wave, v_init=float(svel_shift.val), v_lower=curvefit_v_lower, v_upper=curvefit_v_upper, sigma_init=float(swind_vel.val), sigma_lower=curvefit_sigma_lower, sigma_upper=curvefit_sigma_upper)
        tmp_objects = [bic_list, chi2_list, fit_results]
        base_func.save_pickle(tmp_bic_file, *tmp_objects)
        print (f"{tmp_bic_file} saved")

    # Find best BIC
    best_N = np.argmin(bic_list) + 1
    print(f"\n*** Optimal number of components = {best_N} ***")
    snum_comp.set_val(best_N)

    _ = curvefit_func.plot_bic(bic_list, N_max, penalty_factor, figname=tmp_bic_fig)

    # Plot final best model
    best_popt = fit_results[best_N - 1][0]
    best_perr = fit_results[best_N - 1][1]
    # Wrap the function
    wrapped_model = lambda wave, *theta: base_func.model_curvefit(
        wave, *theta, line_wave=line_wave, magic_number=magic_number)
    tmp_prof_to_use_strong = wrapped_model(orig_wave, *best_popt)

    # Extract component fluxes
    component_info = curvefit_func.extract_component_fluxes(orig_wave, orig_flux_noisy, orig_flux_err, best_popt, theta_unc=best_perr, line_wave=line_wave, line_name=line_id, magic_number=magic_number, delta_v_kms=fig_vel_lim)

    #_ = base_func.print_results(component_info, best_N, tot_original_flux=None, line_id=line_id, line_wave=line_wave)


    tmp_wave_weak, tmp_flux_noisy_weak, tmp_flux_err_weak, tmp_continuum_weak = base_func.obtain_fittable_area(orig_wave, orig_flux_noisy, orig_flux_err, cont, tot_line_wave, allowable_velocity_range=vel_window_tmp)
    flux_noisy_weak = tmp_flux_noisy_weak - tmp_continuum_weak
    tmp_v_shifts, tmp_sig_shifts = curvefit_func.get_vel_sigma_from_popt(best_popt, best_N, line_wave)

    # Fit Weak lines with fixed mu and sigma
    if os.path.exists(tmp_weak_file):
        user_input = input(f"'{tmp_weak_file}' already exists. Load previous results? (y/n): ").strip().lower()
        if user_input == 'y':
            print("Loading existing results...")
            component_info_rev, popt_weak, perr_weak = base_func.load_pickle(tmp_weak_file)
            print (f"{tmp_weak_file} loaded")
        else:
            component_info_rev, popt_weak, perr_weak = curvefit_func.get_weaker_lines_curvefit(tmp_wave_weak, flux_noisy_weak, tmp_flux_err_weak, component_info, best_N, line_wave=tot_line_wave, line_name=tot_line_id, delta_v_kms=fig_vel_lim)
            tmp_objects = [component_info_rev, popt_weak, perr_weak]
            base_func.save_pickle(tmp_weak_file, *tmp_objects)
            print (f"{tmp_weak_file} saved")
    else:
        component_info_rev, popt_weak, perr_weak = curvefit_func.get_weaker_lines_curvefit(tmp_wave_weak, flux_noisy_weak, tmp_flux_err_weak, component_info, best_N, line_wave=tot_line_wave, line_name=tot_line_id, delta_v_kms=fig_vel_lim)
        tmp_objects = [component_info_rev, popt_weak, perr_weak]
        base_func.save_pickle(tmp_weak_file, *tmp_objects)
        print (f"{tmp_weak_file} saved")

    #_ = base_func.print_results(component_info_rev, best_N, tot_original_flux=None, line_id=line_id_weak, line_wave=line_wave_weak)

    # Wrapped model for curve_fit
    wrapped_model_amplitudes_curvefit = lambda wave, *theta: base_func.model_amplitudes_only(wave, *theta, line_wave=tot_line_wave, v_shifts=tmp_v_shifts, sigma_vs=tmp_sig_shifts)
    tmp_prof_to_use_weak = wrapped_model_amplitudes_curvefit(orig_wave, *popt_weak)

    #orig_fit = tmp_prof_to_use_strong + tmp_prof_to_use_weak + cont
    orig_fit = tmp_prof_to_use_weak + cont
    l.set_ydata(base_func.smooth((orig_fit), int(sdata_smooth.val)))
    fig.canvas.draw()

    if os.path.exists(file_with_cont):
        user_input = input(f"'{file_with_cont}' already exists. Replace? (y/n): ").strip().lower()
        if user_input=='y':
            orig_cust_continuum = cont
            tmp_array = np.transpose(np.array([orig_wave, orig_flux_noisy, orig_flux_err, orig_cust_continuum, orig_fit]))
            np.savetxt(file_with_cont, tmp_array, delimiter=',')
            print (f"{file_with_cont} saved")
        else:
            print (f"{file_with_cont} unchanged")
    else:
        orig_cust_continuum = cont
        tmp_array = np.transpose(np.array([orig_wave, orig_flux_noisy, orig_flux_err, orig_cust_continuum, orig_fit]))
        np.savetxt(file_with_cont, tmp_array, delimiter=',')
        print (f"{file_with_cont} saved")

button6.on_clicked(bic_ax)



print_BIC_ax = plt.axes([0.91, 0.15, 0.08, 0.04])
button7 = Button(print_BIC_ax, 'print_BIC', color=axcolor, hovercolor='0.975')
def print_BIC_ax(event):
    print ("Printing BIC...")
    if os.path.exists(tmp_weak_file):
        print("Loading existing results...")
        bic_list, chi2_list, fit_results = base_func.load_pickle(tmp_bic_file)
        best_N = np.argmin(bic_list) + 1
        component_info_rev, popt_weak, perr_weak = base_func.load_pickle(tmp_weak_file)
        print (f"{tmp_weak_file} loaded")
        _ = base_func.print_results(component_info_rev, best_N, tot_original_flux=None, line_id=tot_line_id, line_wave=tot_line_wave)
    else:
        print (f"{tmp_weak_file} not found")
button7.on_clicked(print_BIC_ax)



check_BIC_ax = plt.axes([0.91, 0.10, 0.08, 0.04])
button8 = Button(check_BIC_ax, 'check_BIC', color=axcolor, hovercolor='0.975')
def check_BIC_ax(event):
    print ("Checking BIC result...")
    if os.path.exists(tmp_weak_file):
        print("Loading existing results...")
        bic_list, chi2_list, fit_results = base_func.load_pickle(tmp_bic_file)
        best_N = np.argmin(bic_list) + 1
        component_info_rev, popt_weak, perr_weak = base_func.load_pickle(tmp_weak_file)
        print (f"{tmp_weak_file} loaded")
        _ = base_func.checking_results_plot(component_info_rev, tot_original_flux=None, line_id=line_id, line_wave=line_wave, strong_line_id=line_id, figname=tmp_compare_curvefit_result_fig)
        _ = base_func.checking_results_plot(component_info_rev, tot_original_flux=None, line_id=line_id_weak, line_wave=line_wave_weak, strong_line_id=line_id, figname=tmp_compare_curvefit_result_fig2)
    else:
        print (f"{tmp_weak_file} not found")
button8.on_clicked(check_BIC_ax)



plot_BIC_ax = plt.axes([0.91, 0.05, 0.08, 0.04])
button9 = Button(plot_BIC_ax, 'plot_BIC', color=axcolor, hovercolor='0.975')
def plot_BIC_ax(event):
    print ("Plotting BIC result...")
    global cont, x1, y1, x, y, coordsx, coordsy, orig_wave, orig_flux_noisy, orig_flux_err, orig_continuum, orig_fit
    try:
        cont = cont
        cont_points = np.transpose(np.array([coordsx, coordsy]))
        np.savetxt(cont_file_name, cont_points)
        print ("continuum points saved")
    except:
        cont = orig_continuum

    magic_number = len(line_wave)+2
    if os.path.exists(tmp_weak_file):
        print("Loading existing results...")
        bic_list, chi2_list, fit_results = base_func.load_pickle(tmp_bic_file)
        best_N = np.argmin(bic_list) + 1
        component_info_rev, popt_weak, perr_weak = base_func.load_pickle(tmp_weak_file)
        print (f"{tmp_weak_file} loaded")
        orig_flux_noisy_rev = orig_flux_noisy - cont
        _ = curvefit_func.plot_strong_line_stamp(orig_wave, orig_flux_noisy_rev, orig_flux_err, magic_number, bic_list, fit_results, figname=tmp_curvefit_stamp_fig, line_wave=line_wave, line_name=line_id, vel_lim=fig_vel_lim)
        _ = curvefit_func.plot_weak_line_stamp(orig_wave, orig_flux_noisy_rev, orig_flux_err, component_info_rev, best_N, popt_weak, perr_weak, figname=tmp_curvefit_stamp_fig2, line_wave_tot=tot_line_wave, line_name_tot=tot_line_id, line_wave=line_wave_weak, line_name=line_id_weak, vel_lim=fig_vel_lim)
    else:
        print (f"{tmp_weak_file} not found")
button9.on_clicked(plot_BIC_ax)




plt.xlim((np.nanmin(x)-1.), (np.nanmax(x)+1.))
plt.ylim((np.nanmin(y)-(0.1*np.nanmean(y))), (np.nanmax(y)+(2*np.nanmean(y))))

plt.show()
plt.close(fig)


quit()


