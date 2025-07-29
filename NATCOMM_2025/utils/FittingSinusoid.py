from scipy.optimize import curve_fit
import numpy as np
import pandas as pd

def sin_func(x, freq, amplitude, phase, offset):
    '''
    Create the sine function we want to fit.
    Input: 
            x: data points of a sinusoid curve with frequency = RR interval (an array).
            freq, amplitude, phase, offset: parameters that can be adjusted to find the optimal fit.
    '''
    return np.sin(x * freq + phase) * amplitude + offset


def fit_sin(x, area_pred):
    '''
    Provide sin_func function to curve_fit along with the measure data and an initial guess for the amplitude, phase, offset and
    frequency. A good initial guess is important, as the optimal solution can't always be found from an arbitrary initial guess. 
    
    Input: 
        area_pred: predicted area (numpy array)
    return:
        The optimal values of the four parameters (freq, amplitude, phase, offset). 
    
    '''
    guess_freq =  1 
    guess_amplitude = 3*np.std(area_pred)/(2**0.5)
    guess_phase = 0
    guess_offset = np.mean(area_pred)

    p0=[guess_freq, guess_amplitude,
        guess_phase, guess_offset]

    fit = curve_fit(sin_func, x, area_pred, p0=p0)
    return fit


def get_fitted_r2(x, area_pred):
    '''
    Find the R2 between predicted area by frame and sinusoid function.
    Input: 
        x: data points of a sinusoid curve with frequency = RR interval (an array).
        return: R2, MSE (mean squared errors), RMSE (Root Mean Squared Error), 
    '''
    
    fit = fit_sin(x, area_pred)
    data_fit = sin_func(x, *fit[0])

    absError = data_fit - area_pred

    SE = np.square(absError) # squared errors
    MSE = np.mean(SE) # mean squared errors
    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(area_pred))
    
    return MSE, RMSE, Rsquared, data_fit, fit


def computeR2(df_frames):
    """
    Compute R-squared values for each clip in the given dataframe.
    Parameters:
    - df_frames (DataFrame): The dataframe containing the clip data.
    - plot (bool, optional): Whether to plot the results. Defaults to False.
    Returns:
    - df_rsquare (DataFrame): The dataframe containing the computed R-squared values for each clip.
    """

    df_rsquare = pd.DataFrame([])
    list_clips = df_frames.ID_clip.unique()
    j=0    
    for ID_clip in np.sort(list_clips):

        ID = ID_clip.split('_',2)[0]

        print("    Status: %s / %s" %(j, len(list_clips)), end="\r")
        j+=1

        df_clip = df_frames[df_frames.ID_clip == ID_clip]

        total_frames = df_clip.shape[0]
        rr = df_clip.RR.values[0] / 2
        

        data = df_clip['area'].values
        
        try:
            t = np.linspace(0, round(total_frames / rr) * np.pi, total_frames)
            MSE, RMSE, Rsquared, data_fit, fit = get_fitted_r2(t, data)
            freq = round(fit[0][0],3)
            amplitude = round(fit[0][1],3)
            phase = round(fit[0][2],3)
            offset = round(fit[0][3],3)

            if Rsquared < 0.4:
                t = np.linspace(0, round(total_frames/ (rr * 2)) * np.pi, total_frames)
                MSE, RMSE, Rsquared, data_fit, fit = get_fitted_r2(t, data)                
                freq = round(fit[0][0],3)
                amplitude = round(fit[0][1],3)
                phase = round(fit[0][2],3)
                offset = round(fit[0][3],3)


        except:
            Rsquared, RMSE, MSE, data_fit = 0, 0, 0, 0
            freq = 0
            amplitude = 0
            phase = 0
            offset = 0


        data = pd.DataFrame({'anonid':[ID],'ID_clip':[ID_clip],'Rsquared':[Rsquared],'RMSE':[RMSE], 'MSE':[MSE], 'cycles':[int(total_frames/rr)],
                            'freq':freq, 'amplitude':amplitude, 'phase':phase, 'offset':offset})
        df_rsquare = df_rsquare.append(data)
        
    return df_rsquare
