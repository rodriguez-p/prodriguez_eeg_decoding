import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection

# probar con la implementación de la librería pinguoin que da valores para el tamaño del efecto

def wilcoxon_fdr_tempdecod(x_data, y_data, times, p_threshold = .001):
    """
    Loads decoding scores on time-series data (either with voltage or TF power) and performs
    a Wilcoxon signed-rank test in each time point for both datasets. The resulting p values 
    are then FDR-adjusted for multiple comparisons. Returns the corrected p values, 
    a boolean array indicating whether or not the comparison is significant and a
    temporal array with the significant time points.

    Args:
        x_data (array): the first dataset
        y_data (array): the second dataset
        times (array): temporal data
        p_threshold (int): significance threshold, default is 0.001

    Returns:
        reject: boolean array the same size as the input data indicating whether
        or not the comparison is significant
        p_corrected: arrray with the FDR-adjusted p values (also called q values)
        sign_points: array with the time points where the comparison is
        significant (used for plotting)
    """
    w_array = []  # initialize array to store w values
    p_vals_array = []  # initialize array to store p values

    for i, _ in enumerate(times):  # loop over the whole time-series and
        # compare x vs y in each time point
        x = x_data[:, i]
        y = y_data[:, i]
        w, p_val = wilcoxon(x, y)

        if np.any(w_array):  # store w values
            w_array = np.append(w_array, w)
        else:
            w_array = w

        if np.any(p_vals_array):  # store p values
            p_vals_array = np.append(p_vals_array, p_val)
        else:
            p_vals_array = p_val

    reject, p_corrected = fdrcorrection(
        p_vals_array, alpha=0.05, method='indep')  # FDR-adjust p values

    # save the significant time points
    sign_points = [i for i, p in enumerate(p_corrected) if p < p_threshold]

    return reject, p_corrected, sign_points

def wilcoxon_fdr_tfdecod(x_data,  p_threshold = .001):
    """
    Loads decoding scores on time-frequency data and performs a Wilcoxon signed-rank test in each 
    time-frequency point against chance (50%). The resulting p values are then FDR-adjusted
    for multiple comparisons. Returns the corrected p values, a boolean array indicating
    whether or not the comparison is significant and a temporal array with the
    significant time points.

    Args:
        x_data (array): the first dataset
        times (array): temporal data
        p_threshold (int): significance threshold, default is 0.001

    Returns:
        reject: boolean array the same size as the input data indicating whether
        or not the comparison is significant
        p_corrected: arrray with the FDR-adjusted p values (also called q values)
        sign_points: array with the time points where the comparison is
        significant (used for plotting)
    """
    x_data = np.reshape(x_data, (30, 25*1024)) # reshape time-frequency dimensions
    y_data = np.full((30, 25*1024), 0.5) # create chance array with the same shape

    w_array = []  # initialize array to store w values
    p_vals_array = []  # initialize array to store p values

    for i in range(0, 25600):  # loop over the whole time-frequency series and
        # compare x vs y in each time point
        x = x_data[:, i]
        y = y_data[:, i]
        w, p_val = wilcoxon(x, y)

        if np.any(w_array):  # store w values
            w_array = np.append(w_array, w)
        else:
            w_array = w

        if np.any(p_vals_array):  # store p values
            p_vals_array = np.append(p_vals_array, p_val)
        else:
            p_vals_array = p_val

    reject, p_corrected = fdrcorrection(
        p_vals_array, alpha=0.05, method='indep')  # FDR-adjust p values

    reject = np.reshape(reject, (25, 1024))
    p_corrected = np.reshape(p_corrected, (25, 1024))
    sign_points = p_corrected < .001
    
    return reject, p_corrected, sign_points
