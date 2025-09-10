import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.ndimage.morphology import binary_erosion
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy import interpolate
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from statsmodels.nonparametric.smoothers_lowess import lowess




##################################GET_CONTINUUM##################################


class continuum_fitClass:
    def __init__(self):
        self.plot=False
        self.print=False
        self.legend=False
        self.ax=False
        self.label=False
        self.color='gray'
        self.default_smooth=5
        self.default_order=8
        self.default_allowed_percentile=75
        self.default_filter_points_len=10
        self.data_height_upscale=2
        self.default_poly_order=3
        self.default_window_size_default=999
        self.default_fwhm_galaxy_min=10
        self.default_noise_level_sigma=10
        self.default_fwhm_ratio_upscale=10
        #self.spline_smoothing_factor = 1
        self.pca_component_number = 1
        self.lowess_cont_frac = 0.05
        self.gaussian_cont_fit = 200
        self.peak_prominence = 0.05
        self.peak_width = 10
        self.median_filter_window = 101
        self.continuum_finding_method = 'custom'
        pass
    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    def gaussian(self, x, amp, mu, sig):
        return amp*(1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2))
    #This function has been created taking inference from Martin+2021 (2021MNRAS.500.4937M)
    def continuum_finder(self, wave, *pars, **kwargs):
        flux = self.flux
        #n_smooth=5, order=8, allowed_percentile=75, poly_order=3, window_size_default=None
        if pars:
            n_smooth, allowed_percentile, filter_points_len, data_height_upscale, poly_order, window_size_default, fwhm_galaxy_min, noise_level_sigma, fwhm_ratio_upscale = pars
        else:
            n_smooth, allowed_percentile, filter_points_len, data_height_upscale, poly_order, window_size_default, fwhm_galaxy_min, noise_level_sigma, fwhm_ratio_upscale = np.array([self.default_smooth, self.default_allowed_percentile, self.default_filter_points_len, self.data_height_upscale, self.default_poly_order, self.default_window_size_default, self.default_fwhm_galaxy_min, self.default_noise_level_sigma, self.default_fwhm_ratio_upscale])

        n_smooth = int(n_smooth)
        allowed_percentile = int(allowed_percentile)
        filter_points_len = int(filter_points_len)
        poly_order = int(poly_order)
        window_size_default = int(window_size_default)
        fwhm_galaxy_min = int(fwhm_galaxy_min)
        noise_level_sigma = int(noise_level_sigma)
        fwhm_ratio_upscale = int(fwhm_ratio_upscale)

        pick = np.isfinite(flux) #remove NaNs
        flux = flux[pick] #remove NaNs
        smoothed_data = self.smooth(flux, int(n_smooth)) #smooth data
        local_std = np.median([ np.std(s) for s in np.array_split(flux, int(n_smooth)) ]) #find local standard deviation
        mask_less = argrelextrema(smoothed_data, np.less)[0] #find relative extreme points in absorption
        mask_greater = argrelextrema(smoothed_data, np.greater)[0] #find relative extreme points in emission
        mask_less_interpolate_func = interpolate.interp1d(wave[mask_less], flux[mask_less], kind='cubic', fill_value="extrapolate") #interpolate wavelength array like function from relative extreme points in absorption
        mask_greater_interpolate_func = interpolate.interp1d(wave[mask_greater], flux[mask_greater], kind='cubic', fill_value="extrapolate") #interpolate wavelength array like function from relative extreme points in emission
        absolute_array = mask_greater_interpolate_func(wave)-mask_less_interpolate_func(wave) #obtain the absolute array for find_peaks algorithm
        filter_points = np.array([int(i*len(absolute_array)/filter_points_len) for i in range(1,filter_points_len)])
        noise_height_max_default = noise_level_sigma*np.nanmin(np.array([np.nanstd(absolute_array[filter_points[i]-10:filter_points[i]+10]) for i in range(len(filter_points))]))
        data_height_max_default = data_height_upscale*np.nanmax(np.array([np.abs(np.nanmax(absolute_array)), np.abs(np.nanmin(absolute_array))]))
        noise_height_max = kwargs.get('noise_height_max', noise_height_max_default)  # Maximal height for noise
        data_height_max = kwargs.get('data_height_max', data_height_max_default)  # Maximal height for data
        peaks = find_peaks(absolute_array, height=[noise_height_max, data_height_max], prominence=(local_std*3.), width = [fwhm_galaxy_min, int(fwhm_ratio_upscale*fwhm_galaxy_min)]) #run scipy.signal find_peaks algorithm to find peak points
        edges = np.int32([np.round(peaks[1]['left_ips']), np.round(peaks[1]['right_ips'])]) #find edges of peaks
        d = (np.diff(flux, n=1))
        w = 1./np.concatenate((np.asarray([np.median(d)]*1),d))
        w[0] = np.max(w)
        w[-1] = np.max(w)
        for edge in edges.T:
            diff_tmp = int((edge[1] - edge[0])/2)
            w[edge[0]-diff_tmp:edge[1]+diff_tmp] = 1./10000.
        w = np.abs(w)
        pick_2 = np.where(w > np.percentile(w, allowed_percentile * (float(len(flux)) / float(len(wave)))))[0]
        if len(wave[pick][pick_2])>3:
            xx = np.linspace(np.min(wave[pick][pick_2]), np.max(wave[pick][pick_2]), 1000)
            itp = interpolate.interp1d(wave[pick][pick_2], flux[pick_2], kind='linear')
        else:
            mask = np.ones_like(wave, dtype=np.bool_)
            ynew = np.abs(np.diff(flux[mask], prepend=1e-10))
            ynew2 = np.percentile(ynew, allowed_percentile)
            xx = wave[mask][ynew < ynew2]
            y_rev = flux[mask][ynew < ynew2]
            itp = interpolate.interp1d(xx, y_rev, axis=0, fill_value="extrapolate", kind='linear')

        window_size = int(fwhm_ratio_upscale*fwhm_galaxy_min)
        if window_size % 2 == 0:
            window_size = window_size + 1
        fit_savgol = savgol_filter(itp(xx), window_size, poly_order)
        fit = interpolate.interp1d(xx, fit_savgol, kind='cubic', fill_value="extrapolate")
        std_cont = np.std(flux[pick_2] - fit(wave[pick][pick_2]))
        return fit, std_cont, flux, pick, pick_2, peaks, std_cont

    # Function for fitting a polynomial continuum
    def poly_fit_continuum(self, wave):
        degree = self.default_poly_order
        flux = self.flux
        wavelength = wave
        coefficients = np.polyfit(wavelength, flux, degree)
        continuum = np.polyval(coefficients, wavelength)
        return continuum

    # Function for fitting a spline continuum
    def spline_fit_continuum(self, wave):
        flux = self.flux
        wavelength = wave
        continuum = interp1d(wavelength, flux, kind='cubic')
        return continuum(wavelength)

    # Function for estimating the continuum using PCA
    def pca_continuum(self, wave):
        flux = self.flux
        wavelength = wave
        pca_component_number = self.pca_component_number
        pca = PCA(n_components = pca_component_number)
        X = flux.reshape(-1, 1)
        pca.fit(X)
        continuum = pca.inverse_transform(pca.transform(X)).flatten()
        return continuum

    # Function for estimating the continuum using a lowess smoother
    def lowess_continuum(self, wave):
        wavelength = wave
        flux = self.flux
        lowess_continuum_fraction = self.lowess_cont_frac
        continuum = lowess(flux, wavelength, frac=lowess_continuum_fraction)[:, 1]
        return continuum

    # Function for estimating the continuum using a Gaussian fit
    def gaussian_func(self, x, a, b, c, d):
        return a * np.exp(-((x-b)/c)**2) + d

    def gaussian_fit_continuum(self, wave):
        wavelength = wave
        flux = self.flux
        window = self.gaussian_cont_fit
        continuum = np.zeros_like(flux)
        for i in range(len(flux)):
            low = max(0, i-window//2)
            high = min(len(flux), i+window//2)
            x = wavelength[low:high]
            y = flux[low:high]
            try:
                popt, _ = curve_fit(self.gaussian_func, x, y, p0=[1, wavelength[i], 5, 0])
                continuum[i] = self.gaussian_func(wavelength[i], *popt)
            except RuntimeError:
                continuum[i] = np.nan
        mask = np.isnan(continuum)
        continuum[mask] = np.interp(wavelength[mask], wavelength[~mask], continuum[~mask])
        return continuum

    # Function for estimating the continuum using peak finding
    def peak_find_continuum(self, wave):
        wavelength = wave
        flux = self.flux
        prominence = self.peak_prominence
        width = self.peak_width
        peaks, _ = find_peaks(flux, prominence=prominence, width=width)
        troughs, _ = find_peaks(-flux, prominence=prominence, width=width)
        indices = np.concatenate([peaks, troughs, [0, len(flux)-1]])
        continuum = np.interp(wavelength, wavelength[indices], flux[indices])
        return continuum

    # Function for estimating the continuum using median filtering
    def median_filter_continuum(self, wave):
        wavelength = wave
        flux = self.flux
        window = self.median_filter_window
        continuum = np.zeros_like(flux)
        for i in range(len(flux)):
            low = max(0, i-window//2)
            high = min(len(flux), i+window//2)
            continuum[i] = np.nanmedian(flux[low:high])
        mask = np.isnan(continuum)
        continuum[mask] = np.interp(wavelength[mask], wavelength[~mask], continuum[~mask])
        return continuum

    # Define distance metric function
    def dist_metric(self, x, y):
        return np.abs(x - y)

    def continuum_using_fof(self, wave):
        wavelength = wave
        flux = self.flux
        # Calculate distance matrix
        X = wavelength.reshape(-1, 1)
        D = cdist(X, X, self.dist_metric)
        # Find clusters using DBSCAN
        db = DBSCAN(eps=3, min_samples=3, metric='precomputed').fit(D)
        labels = db.labels_
        # Calculate continuum
        mask = (labels == 0)
        continuum = np.median(flux[mask])
        return continuum


    def continuum_finder_flux(self, wave, *pars, **kwargs):
        cont_find_method = self.continuum_finding_method
        if (cont_find_method=='custom'):
            fit, std_cont, flux, pick, pick_2, peaks, std_cont = self.continuum_finder(wave, *pars)
            cont_flux = fit(wave[pick])
        elif (cont_find_method=='poly'):
            cont_flux = self.poly_fit_continuum(wave)
        elif (cont_find_method=='spline'):
            cont_flux = self.spline_fit_continuum(wave)
        elif (cont_find_method=='pca'):
            cont_flux = self.pca_continuum(wave)
        elif (cont_find_method=='lowess'):
            cont_flux = self.lowess_continuum(wave)
        elif (cont_find_method=='gauss'):
            cont_flux = self.gaussian_fit_continuum(wave)
        elif (cont_find_method=='peak_find'):
            cont_flux = self.peak_find_continuum(wave)
        elif (cont_find_method=='median_filtering'):
            cont_flux = self.median_filter_continuum(wave)
        elif (cont_find_method=='fof'):
            continuum_fof = self.continuum_using_fof(wave)
            cont_flux = continuum_fof*np.ones_like(wave)
        else:
            bf.print_cust(f'Continuum finding method: {cont_find_method} not found. Reverting back to custom method.')
            fit, std_cont, flux, pick, pick_2, peaks, std_cont = self.continuum_finder(wave, *pars)
            cont_flux = fit(wave[pick])
        return (cont_flux)


