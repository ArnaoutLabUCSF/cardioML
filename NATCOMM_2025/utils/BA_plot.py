from random import shuffle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import warnings
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


def detrendFun(method, data1, data2):
    """
    Model and remove a mutiplicative offset between data1 and data2 by method
    :param method: Detrending method to use 
    :type method: None or str
    :param numpy.array data1: Array of first measures
    :param numpy.array data2: Array of second measures
    """

    slope = slopeErr = None

    if method is None:
        pass
    elif method.lower() == 'linear':
        reg = stats.linregress(data1, data2)

        slope = reg.slope
        slopeErr = reg.stderr

        data2 = data2 / slope

    elif method.lower() == 'odr':
        from scipy import odr

        def f(B, x):
            return B[0]*x + B[1]
        linear = odr.Model(f)

        odrData = odr.Data(data1, data2, wd=1./numpy.power(numpy.std(data1),2), we=1./numpy.power(numpy.std(data2),2))

        odrModel = odr.ODR(odrData, linear, beta0=[1., 2.])

        myoutput = odrModel.run()

        slope = myoutput.beta[0]
        slopeErr = myoutput.sd_beta[0]

        data2 = data2 / slope

    else:
        raise NotImplementedError(f"detrend is not a valid detrending method.")

    return data2, slope, slopeErr


def carkeetCIest(n, gamma, limitOfAgreement):
    """
    Calculate  CI intervals on the paired LoA by the Carkeet method.
    Returns the coefficient determining the (gamma x 100)% confidence interval on the SD x limitOfAggreement.
    Position of the limit is calculated as :math:`mean difference + (coefficient * sd of differences)`
    :param int n: Number of paired observations
    :param float gamma: Calculate coefficient for this bound
    :param float limitOfAgreement: Multiples of SD being considered
    :return: Coefficient determining the (gamma x 100)% confidence interval on the on the SD x limitOfAggreement limit
    :rtype: float
    """

    Degf = n - 1
    gammaest = 0
    Kest = 0
    Kstep = 4
    directK = 1

    threshold = 1e-8

    p = stats.norm.cdf(limitOfAgreement) - stats.norm.cdf(- limitOfAgreement)

    while numpy.abs(gammaest - gamma) > threshold:
        Kest = Kest + Kstep
        K = Kest
        stepper = 0.05 / n
        toprange = 8 / (n**0.5) + stepper
        xdist = numpy.arange(0, toprange, stepper)
        boxes = len(xdist)
        boxes = int(numpy.round(boxes / 2 + .1)) * 2 - 1
        Prchi = numpy.zeros(boxes)
        Combpdf = numpy.zeros(boxes)
        halfgauss = numpy.exp(-(n/2) * xdist **2)
        shrinkfactor = 2 * (n/(2 * numpy.pi)) **.5

        for s in range(boxes - 1):
            xtest = xdist[s]
            startp = (0.5 + p/2)

            resti = stats.norm.ppf(startp) + xtest - .1
            restiprior = resti
            phigh = stats.norm.cdf(xtest + resti)
            plow = stats.norm.cdf(xtest - resti)
            pesti = phigh - plow

            pestiprior = pesti
            resti = resti + .11
            phigh = stats.norm.cdf(xtest + resti)
            plow = stats.norm.cdf(xtest - resti)
            pesti = phigh - plow
            perror = pesti - p
            deltap = pesti - pestiprior

            deltaresti = resti - restiprior
            newresti = resti - perror / deltap * deltaresti
            restiprior = resti

            resti = newresti

            pestiprior = pesti
            phigh = stats.norm.cdf(xtest + resti)
            plow = stats.norm.cdf(xtest - resti)
            pesti = phigh - plow

            perror = pesti - p
            while numpy.abs(perror) > 2e-15:
                deltap = pesti - pestiprior

                deltaresti = resti - restiprior
                newresti = resti - perror / deltap * deltaresti
                restiprior = resti

                resti = newresti

                pestiprior = pesti
                phigh = stats.norm.cdf(xtest + resti)
                plow = stats.norm.cdf(xtest - resti)
                pesti = phigh - plow

                perror = pesti - p

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                chiprob = 1 - stats.chi2.cdf((Degf * resti**2) / (K**2), Degf)
            Prchi[s] = chiprob
            Combpdf[s] = chiprob * halfgauss[s]

        Integ = 0
        for s in range(0, boxes - 2, 2):
            M = Combpdf[s+1] * stepper * 2

            T = (Combpdf[s] + Combpdf[s+2]) * stepper
            Integ = Integ + (M*2+T)/ 3 * shrinkfactor

        gammaest = Integ
        if (gammaest * directK) > (gamma * directK):
            directK = directK * -1
            Kstep = - Kstep / 2

    return Kest

def calculateConfidenceIntervals(md, sd, n, limitOfAgreement, confidenceInterval, confidenceIntervalMethod):
    """
    Calculate confidence intervals on the mean difference and limits of agreement.
    Two methods are supported, the approximate method descibed by Bland & Altman, and the exact paired method described by Carket.
    :param float md:
    :param float sd:
    :param int n: Number of paired observations
    :param float limitOfAgreement:
    :param float confidenceInterval: Calculate confidence intervals over this range
    :param str confidenceIntervalMethod: Algorithm to calculate CIs
    """
    confidenceIntervals = dict()

    if not (confidenceInterval < 99.9) & (confidenceInterval > 1):
        raise ValueError(f'"confidenceInterval" must be a number in the range 1 to 99, "{confidenceInterval}" provided.')

    confidenceInterval = confidenceInterval / 100.

    confidenceIntervals['mean'] = stats.t.interval(confidenceInterval, n-1, loc=md, scale=sd/numpy.sqrt(n))

    if confidenceIntervalMethod.lower() == 'exact paired':

        coeffs = parallelCarkeetCIest(n, confidenceInterval, limitOfAgreement)

        coefInner = coeffs[0]
        coefOuter = coeffs[1]

        confidenceIntervals['upperLoA'] = (md + (coefInner * sd),
                                           md + (coefOuter * sd))

        confidenceIntervals['lowerLoA'] = (md - (coefOuter * sd),
                                           md - (coefInner * sd))

    elif confidenceIntervalMethod.lower() == 'approximate':

        seLoA = ((1/n) + (limitOfAgreement**2 / (2 * (n - 1)))) * (sd**2)
        loARange = numpy.sqrt(seLoA) * stats.t._ppf((1-confidenceInterval)/2., n-1)

        confidenceIntervals['upperLoA'] = ((md + limitOfAgreement*sd) + loARange,
                                           (md + limitOfAgreement*sd) - loARange)

        confidenceIntervals['lowerLoA'] = ((md - limitOfAgreement*sd) + loARange,
                                           (md - limitOfAgreement*sd) - loARange)

    else:
        raise NotImplementedError(f"'{confidenceIntervalMethod}' is not an valid method of calculating confidance intervals")

    return confidenceIntervals


def parallelCarkeetCIest(n, confidenceInterval, limitOfAgreement): # pragma: no cover
    coeffs = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        for result in executor.map(carkeetCIest, repeat(n), [(1 - confidenceInterval) / 2., 1 - (1 - confidenceInterval) / 2.], repeat(limitOfAgreement)):
            coeffs.append(result)
    return coeffs



def blandAltman(data1, data2,dataCat, unity, limitOfAgreement=1.96, confidenceInterval=95, confidenceIntervalMethod='approximate', percentage=False,
                 detrend=None, title=None, ax=None, figureSize=(10,7), fontsize = 14, dpi=72, savePath=None, figureFormat='pdf', meanColour='#6495ED',
                   loaColour='coral', pointColour='#6495ED',lim_inf=-70, lim_sup=70, lim_inf_x=0, lim_sup_x=70,cat=False, sev=False, sliced=None):
    """
    blandAltman(data1, data2, limitOfAgreement=1.96, confidenceInterval=None, **kwargs)
    Generate a Bland-Altman [#]_ [#]_ plot to compare two sets of measurements of the same value.
    Confidence intervals on the limit of agreement may be calculated using:
    - 'exact paired' uses the exact paired method described by Carkeet [#]_
    - 'approximate' uses the approximate method described by Bland & Altman
    The exact paired method will give more accurate results when the number of paired measurements is low (approx < 100), at the expense of much slower plotting time.
    The *detrend* option supports the following options:
    - ``None`` do not attempt to detrend data - plots raw values
    - 'Linear' attempt to model and remove a multiplicative offset between each assay by linear regression
    - 'ODR' attempt to model and remove a multiplicative offset between each assay by Orthogonal distance regression
    :param data1: List of values from the first method
    :type data1: list like
    :param data2: List of paired values from the second method
    :type data2: list like
    :param float limitOfAgreement: Multiples of the standard deviation to plot limit of agreement bounds at (defaults to 1.96)
    :param confidenceInterval: If not ``None``, plot the specified percentage confidence interval on the mean and limits of agreement
    :param str confidenceIntervalMethod: Method used to calculated confidence interval on the limits of agreement
    :type confidenceInterval: None or float
    :param detrend: If not ``None`` attempt to detrend by the method specified
    :type detrend: None or str
    :param bool percentage: If ``True``, plot differences as percentages (instead of in the units the data sources are in)
    :param str title: Title text for the figure
    :param matplotlib.axes._subplots.AxesSubplot ax: Matplotlib axis handle - if not `None` draw into this axis rather than creating a new figure
    :param figureSize: Figure size as a tuple of (width, height) in inches
    :type figureSize: (float, float)
    :param int dpi: Figure resolution
    :param str savePath: If not ``None``, save figure at this path
    :param str figureFormat: When saving figure use this format
    :param str meanColour: Colour to use for plotting the mean difference
    :param str loaColour: Colour to use for plotting the limits of agreement
    :param str pointColour: Colour for plotting data points
    .. [#] Altman, D. G., and Bland, J. M. “Measurement in Medicine: The Analysis of Method Comparison Studies” Journal of the Royal Statistical Society. Series D (The Statistician), vol. 32, no. 3, 1983, pp. 307–317. `JSTOR <https://www.jstor.org/stable/2987937>`_.
    .. [#] Altman, D. G., and Bland, J. M. “Measuring agreement in method comparison studies” Statistical Methods in Medical Research, vol. 8, no. 2, 1999, pp. 135–160. `DOI <https://doi.org/10.1177/096228029900800204>`_.
    .. [#] Carkeet, A. "Exact Parametric Confidence Intervals for Bland-Altman Limits of Agreement" Optometry and Vision Science, vol. 92, no 3, 2015, pp. e71–e80 `DOI <https://doi.org/10.1097/OPX.0000000000000513>`_.
    """
    if not limitOfAgreement > 0:
        raise ValueError('"limitOfAgreement" must be a number greater than zero.') 

    # Try to coerce variables to numpy arrays
    data1 = numpy.asarray(data1)
    data2 = numpy.asarray(data2)

    data2, slope, slopeErr = detrendFun(detrend, data1, data2)

    mean = numpy.mean([data1, data2], axis=0)

    if percentage:
      diff = ((data1 - data2) / mean) * 100
    else:
        diff = data1 - data2

    md = numpy.mean(diff)
    sd = numpy.std(diff, axis=0)

    if confidenceInterval:
        confidenceIntervals = calculateConfidenceIntervals(md, sd, len(diff), limitOfAgreement, confidenceInterval, confidenceIntervalMethod)

    else:
        confidenceIntervals = dict()

    ax = _drawBlandAltman(mean, diff, md, sd, percentage, limitOfAgreement, confidenceIntervals, (detrend, slope, slopeErr), title, ax, figureSize,
                           fontsize, dpi, savePath, figureFormat, meanColour, loaColour, pointColour, lim_inf,  lim_sup, lim_inf_x, lim_sup_x, 
                           dataCat,cat,unity, sev=sev, sliced=sliced)

    if ax is not None:
        return ax


def _drawBlandAltman(mean, diff, md, sd, percentage, limitOfAgreement, confidenceIntervals, detrend, title, ax, figureSize, fontsize, dpi, savePath,figureFormat, meanColour, loaColour, pointColour,lim_inf, lim_sup, lim_inf_x, lim_sup_x,dataCat,cat,unity, sev=False,
                        sliced=None):
    """
    Draw a Bland-Altman plot.
    Parameters:
    - mean (array-like): Array of mean values.
    - diff (array-like): Array of difference values.
    - md (float): Mean difference.
    - sd (float): Standard deviation of the differences.
    - percentage (float): Percentage of the mean difference.
    - limitOfAgreement (float): Limit of agreement.
    - confidenceIntervals (dict): Dictionary of confidence intervals.
    - detrend (bool): Whether to detrend the data.
    - title (str): Title of the plot.
    - ax (matplotlib.axes.Axes): Axes object to plot on.
    - figureSize (tuple): Figure size in inches.
    - fontsize (int): Font size of the plot.
    - dpi (int): Dots per inch of the plot.
    - savePath (str): Path to save the plot.
    - figureFormat (str): Format of the saved plot.
    - meanColour (str): Color of the mean line.
    - loaColour (str): Color of the limit of agreement lines.
    - pointColour (str): Color of the data points.
    - lim_inf (float): Lower limit of the plot.
    - lim_sup (float): Upper limit of the plot.
    - lim_inf_x (float): Lower limit of the x-axis.
    - lim_sup_x (float): Upper limit of the x-axis.
    - dataCat (array-like): Array of categorical data.
    - cat (bool): Whether the data is categorical.
    - unity (bool): Whether to plot the unity line.
    - sev (bool): Whether to plot severity categories.
    - sliced (str): Type of slicing for categorical data.
    Returns:
    - None
    """
        
    if ax is None:
        fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)
        draw = True
    else:
        draw = False

    if 'mean' in confidenceIntervals.keys():
        ax.axhspan(confidenceIntervals['mean'][0],
                   confidenceIntervals['mean'][1],
                   facecolor=meanColour, alpha=0.2)

    if 'upperLoA' in confidenceIntervals.keys():
        ax.axhspan(confidenceIntervals['upperLoA'][0],
                   confidenceIntervals['upperLoA'][1],
                   facecolor=loaColour, alpha=0.2)

    if 'lowerLoA' in confidenceIntervals.keys():
        ax.axhspan(confidenceIntervals['lowerLoA'][0],
                   confidenceIntervals['lowerLoA'][1],
                   facecolor=loaColour, alpha=0.2)

    ax.axhline(md, color=meanColour, linestyle='--',  linewidth = 2)
    ax.axhline(md + limitOfAgreement*sd, color=loaColour, linestyle='--',  linewidth = 2)
    ax.axhline(md - limitOfAgreement*sd, color=loaColour, linestyle='--',  linewidth = 2)

    if cat:

        if sev:
            label_map = ["normal", "mildly abnormal", "moderately abnormal", "severely abnormal"]
            thecolors = ['#1f78b4', '#33a02c', '#ff7f00', '#e31a1c']
            theshapes=['o', 'o', 'o', 'o']

        elif sliced != None:
            if sliced == 'Study Quality':
                label_map = ['normal', 'suboptimal']
                thecolors = ['#0173b2', '#d55e00']
                theshapes = ['o', 'o']

            elif sliced == 'Race':
                label_map = ['Latinx', 'Black or African American', 'White', 'Other', 'Asian']
                thecolors = ['#FF0000', '#ADD8E6', '#008000', '#008080', '#800080']
                theshapes = ['o', 'o', 'o', 'o', 'o']
            
            elif sliced == 'Age':
                label_map = ['age < 40', 'age 40-60', 'age > 60']
                thecolors = ['#FF0000', '#0000FF', '#FFA500']
                theshapes = ['o', 'o', 'o']
            
            elif sliced == 'Sex':
                label_map = ['F', 'M']
                thecolors = ['#FF0000', '#0000FF']
                theshapes = ['o', 'o']

            elif sliced == 'Sex Age':
                sex_label_map = ['F', 'M']
                age_label_map = ['age < 40', 'age 40-60', 'age > 60']
                thecolors = ['#FF0000', '#0000FF', '#FFA500']
                theshapes = ['o', 's']

                thecolors_dict = {}
                for age_label, color in zip(age_label_map, thecolors):
                    thecolors_dict[age_label] = color
                
                theshapes_dict = {}
                for sex_label, shape in zip(sex_label_map, theshapes):
                    theshapes_dict[sex_label] = shape

            elif sliced == 'Demographics':
                race_label_map = ['Latinx', 'Black or African American', 'White', 'Other', 'Asian']
                age_label_map = ['age < 40', 'age 40-60', 'age > 60']
                sex_label_map = ['F', 'M']

                thecolors = ['#FF0000', '#ADD8E6', '#008000', '#008080', '#800080']
                thecolors_dict = {}
                for race_label, color in zip(race_label_map, thecolors):
                    thecolors_dict[race_label] = color

                theshapes = ['o', 's', '^']
                theshapes_dict = {}
                for age_label, shape in zip(age_label_map, theshapes):
                    theshapes_dict[age_label] = shape

                thefills = ['none', 'full']
                thefills_dict = {}
                for sex_label, fill in zip(sex_label_map, thefills):
                    thefills_dict[sex_label] = fill
                
            else:
                label_map = [True, False]
                thecolors = ['#FF0000', '#0000FF']
                theshapes = ['o', 'o']
        else:
            label_map = ["Normal", "Abnormal"] #, "TOF"]
            thecolors = ['#0173b2', '#d55e00'] #, '#C2A5CF']
            theshapes=['o', 'o']

        if sliced == 'Demographics':
            for race in race_label_map:
                for age in age_label_map:
                    for sex in sex_label_map:
                        mask = (dataCat['UCSF Race / Ethnicity New'] == race) & (dataCat['categorical_age'] == age) & (dataCat['patient_sex'] == sex)
                        if thefills_dict[sex] == 'full':
                            ax.scatter(mean[mask], diff[mask], color=thecolors_dict[race], marker=theshapes_dict[age], facecolors=thecolors_dict[race], 
                                       alpha=0.5, label=f'{race}, {age}, {sex}', s=60)
                        else:
                            ax.scatter(mean[mask], diff[mask], color=thecolors_dict[race], marker=theshapes_dict[age], facecolors='none', 
                                       alpha=0.5, label=f'{race}, {age}, {sex}', s=60)
            legend_elements = []
            for race in thecolors_dict:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=thecolors_dict[race], markersize=10, label=race))
            for age in theshapes_dict:
                legend_elements.append(Line2D([0], [0], marker=theshapes_dict[age], color='grey', label=age))
            for sex in thefills_dict:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='k' if thefills_dict[sex] == 'none' else 'w', 
                                              markeredgewidth=1, linestyle='None', label=sex, markeredgecolor='black'))

            plt.legend(handles=legend_elements, loc='best', fontsize = 8)
        
        elif sliced == 'Sex Age':
            ax_xs = []
            ax_ys = []
            c = []
            shapes = []
            labels = []
            for age in age_label_map:
                    for sex in sex_label_map:
                        mask = (dataCat['categorical_age'] == age) & (dataCat['patient_sex'] == sex)
                        ax_xs.extend(mean[mask])
                        ax_ys.extend(diff[mask])
                        c.extend([thecolors_dict[age],] * mask.sum())
                        shapes.extend([theshapes_dict[sex],] * mask.sum())
                        labels.extend([f'{sex}, {age}',] * mask.sum())
            ax_x = numpy.array(ax_xs)
            ax_y = numpy.array(ax_ys)
            xycsl = list(zip(ax_x, ax_y, c, shapes, labels))
            shuffle(xycsl)
            ax_x, ax_y, c, shapes, labels = zip(*xycsl)
            print(len(ax_x), len(ax_y), len(c), len(shapes), len(labels))

            for i in range(len(ax_x)):
                ax.scatter(ax_x[i], ax_y[i], color=c[i], marker=shapes[i], alpha=0.5, label=labels[i], s=60)
            legend_elements = []
            for age in thecolors_dict:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=thecolors_dict[age], markersize=10, label=age))
            for sex in theshapes_dict:
                legend_elements.append(Line2D([0], [0], marker=theshapes_dict[sex], color='grey', label=sex))

            plt.legend(handles=legend_elements, loc='best', fontsize = 8)
        else:
            # Concatenate data for all labels
            print(label_map)
            print(dataCat.unique())
            ax_x = numpy.concatenate([mean[dataCat == label] for label in label_map])
            ax_y = numpy.concatenate([diff[dataCat == label] for label in label_map])

            # color string
            c = [thecolors[0],]*(dataCat == label_map[0]).sum() + [thecolors[1],]*(dataCat == label_map[1]).sum() 
            
            if sev:
                c = c + [thecolors[2],]*(dataCat == label_map[2]).sum() + [thecolors[3],]*(dataCat == label_map[3]).sum()
            elif sliced == 'Race':
                c = c + [thecolors[2],]*(dataCat == label_map[2]).sum() + [thecolors[3],]*(dataCat == label_map[3]).sum()
                c = c + [thecolors[4],]*(dataCat == label_map[4]).sum()
            elif sliced == 'Age':
                c = c + [thecolors[2],]*(dataCat == label_map[2]).sum()

            # randomize order
            xyc = list(zip(ax_x, ax_y, c))
            shuffle(xyc)
            ax_xx, ax_yy, c = zip(*xyc)

            ax.scatter(ax_xx, ax_yy, color=c, alpha=0.5, label = label_map[0],s=60)
            ax.scatter(ax_x[0], ax_y[0], color=thecolors[0], alpha=0.5, label = label_map[0],s=60)
            ax.scatter(ax_x[1], ax_y[1], color=thecolors[1], alpha=0.5, label = label_map[1],s=60)
            if sev:
                ax.scatter(ax_x[2], ax_y[2], color=thecolors[2], alpha=0.5, label = label_map[2],s=60)
                ax.scatter(ax_x[3], ax_y[3], color=thecolors[3], alpha=0.5, label = label_map[3],s=60)
            elif sliced == 'Race':
                ax.scatter(ax_x[2], ax_y[2], color=thecolors[2], alpha=0.5, label = label_map[2],s=60)
                ax.scatter(ax_x[3], ax_y[3], color=thecolors[3], alpha=0.5, label = label_map[3],s=60)
                ax.scatter(ax_x[4], ax_y[4], color=thecolors[4], alpha=0.5, label = label_map[4],s=60)
            elif sliced == 'Age':
                ax.scatter(ax_x[2], ax_y[2], color=thecolors[2], alpha=0.5, label = label_map[2],s=60)

            legend_handles = [Patch(facecolor=thecolors[i], edgecolor=thecolors[i], label=label_map[i]) for i in range(len(label_map))]
            
            plt.legend(handles=legend_handles, loc='best', fontsize = 16 if sliced != 'Race' else 8)
    else:
        ax.scatter(mean, diff, alpha=0.5, c=pointColour)

    trans = transforms.blended_transform_factory(
        ax.transAxes, ax.transData)

    limitOfAgreementRange = (md + (limitOfAgreement * sd)) - (md - limitOfAgreement*sd)
    offset = (limitOfAgreementRange / 100.0) * 1.5

    ax.text(1.05, md + offset, 'Mean', ha="right", va="bottom", transform=trans,size=fontsize)
    ax.text(1.05, md - offset, f'{md:.2f}', ha="right", va="top", transform=trans,size=fontsize)

    ax.text(1.05, md + (limitOfAgreement * sd) + offset, f'+{limitOfAgreement:.2f} SD', ha="right", va="bottom", transform=trans,size=fontsize)
    ax.text(1.05, md + (limitOfAgreement * sd) - offset, f'{md + limitOfAgreement*sd:.2f}', ha="right", va="top", transform=trans,size=fontsize)

    ax.text(1.05, md - (limitOfAgreement * sd) - offset, f'-{limitOfAgreement:.2f} SD', ha="right", va="top", transform=trans,size=fontsize)
    ax.text(1.05, md - (limitOfAgreement * sd) + offset, f'{md - limitOfAgreement*sd:.2f}', ha="right", va="bottom", transform=trans,size=fontsize)

    # Only draw spine between extent of the data
    ax.spines['left'].set_bounds(lim_inf,lim_sup) #min(diff),max(diff)
    ax.spines['bottom'].set_bounds(lim_inf_x,lim_sup_x) #round(min(mean)), round(max(mean)))

    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(width=2)


    if percentage:
        ax.set_ylabel(f'Percentage difference between methods ({unity})',size=20)
    else:
        ax.set_ylabel(f'Difference between methods ({unity})',size=20)
    ax.set_xlabel(f'Mean of methods ({unity})',size=20)
    ax.set_xlim([lim_inf_x, lim_sup_x])
    ax.set_ylim([lim_inf, lim_sup])

    if detrend[0] is None:
        pass
    else:
        plt.text(1, -0.1, f'{detrend[0]} slope correction factor: {detrend[1]:.2f} ± {detrend[2]:.2f}', ha='right', transform=ax.transAxes)

    if title:
        ax.set_title(title, fontsize=24, loc = 'left')

    if (savePath is not None) & draw:
        fig.savefig(savePath, format=figureFormat, dpi=dpi)
        plt.close()
    elif draw:
        plt.show()