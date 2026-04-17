
def significance(array):
    '''
Description
----------------------------------------------------------------

    Transform significance values from float to stars ('****')


Params
-----------------------------------------------------------------

    array: list-like. To pass a single value is pass you should pass it
    as a list like '[]'. Otherwise is not going to work


Returns
------------------------------------------------------------------

    List of significant levels for each number on the listlike array


Examples
------------------------------------------------------------------
    a = np.float(0.64)
    significance([a])
    >>> ['ns']

    b = np.array([0.01, 0.001, 0.0001])
    significance(b)
    >>> ['**', '***', '****']

    c = [0.05,0.04, 0.06]
    significance(c)
    >>> ['ns', '*', 'ns']
    '''
    values = []

    for i in array:
        if i >=0.05:
            i='ns'
        elif (i <= 0.05 ) & (i > 0.01):
            i = '*'
        elif (i <= 0.01) & (i > 0.001):
            i = '**'
        elif (i <= 0.001) & (i > 0.0001):
            i = '***'
        elif i <= 0.0001:
            i = '****'
        else:
            i = 'ns'
        values.append(i)
    return values


def outliers(x, method='std',std_n=3, q = [5,95]):
    """
Description
-----------------------------------------------------------

Function to calculate outliers by Std, Percentile and IQR
Excludes nan by default
Requires Numpy

Params
-------------------------------------------------------------
x: numeric array like.

method: string. 'std' (default), 'percentile' or 'iqr'.

std_n: Number to standar deviations to use on (Default = 3).

q: Percentiles to calculate the outliers on. Default: [5,95].
only significant for percentile method.

Returns
-------------------------------------------------
    Method, lower bound, upper bound

    """
    import numpy as np
    method = method
    std_n = std_n
    q = q
    try:
        if method == 'std':
            x_mean = np.nanmean(x)
            x_std = np.nanstd(x)
            lower_x = x_mean - (std_n * x_std)
            higher_x = x_mean + (std_n* x_std)

        elif method == 'percentile':

            lower_x = np.nanpercentile(x, q=q[0])
            higher_x = np.nanpercentile(x, q=q[1])

        elif method == 'iqr':

            q1_x = np.nanpercentile(x, 25, interpolation='midpoint')
            q3_x = np.nanpercentile(x, 75, interpolation='midpoint')
            iqr_x = q3_x - q1_x
            lower_x = q1_x - (1.5 * iqr_x)
            higher_x = q3_x + (1.5 * iqr_x)

        print(f'Method = {method.upper()}.\nLower: {lower_x}, Higher: {higher_x}')
    except AttributeError:
        print(f'WARNING: x is {type(x[0])}! It must be numeric!')




def rmse(y_true, y_pred):
    """
Description
-------------------------------------------------------------
    Computes RMSE based on sklearn.metrics.mean_squared_error. You should import it as mse
    Also needs Numpy.sqrt()

Params
--------------------------------------------------------------
    y_true: array like of true values
    y_pred: array like of predicted values

Returns
--------------------------------------------------------------
    RSME

    """
    from sklearn.metrics import mean_squared_error as mse
    mse = mse(y_true, y_pred)
    rmse = np.sqrt(mse)

    return rmse



def iqr(x):
    q1_x = np.nanpercentile(x, 25, interpolation='midpoint')
    q3_x = np.nanpercentile(x, 75, interpolation='midpoint')
    iqr_x = q3_x - q1_x

    return iqr_x



def movecol(df, cols_to_move=[], ref_col='', place='After'):

    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]

    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]

    return(df[seg1 + seg2 + seg3])


# Defining function to create pie chart and bar plot as subplots
'''
Simple function that takes two args and plots a pie chart and a barplot
df: data frame
var: variable to plot. Should be between '' or ""
'''
def plot_piechart(df, var):
  plt.figure(figsize=(14,7))
  plt.subplot(121)
  label_list = df[var].unique().tolist()
  df[var].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",7),startangle = 60,labels=label_list,
  wedgeprops={"linewidth":2,"edgecolor":"k"},shadow =True)
  plt.title("Distribution of "+ var +" variable")

  plt.subplot(122)
  ax = df[var].value_counts().plot(kind="barh")

  for i,j in enumerate(df[var].value_counts().values):
    ax.text(.7,i,j,weight = "bold",fontsize=20)

  plt.title("Count of "+ var +" cases")
  plt.show()


# 10th Percentile
def q10(x):
    return x.quantile(0.1)

# 90th Percentile
def q90(x):
    return x.quantile(0.9)

# 25th Percentile
def q25(x):
    return x.quantile(0.25)

# 75th Percentile
def q75(x):
    return x.quantile(0.75)



####### Muestra estratificada

# the functions:
def stratified_sample(df, strata, size=None, seed=None, keep_index= True):
    '''
    It samples data from a pandas dataframe using strata. These functions use
    proportionate stratification:
    n1 = (N1/N) * n
    where:
        - n1 is the sample size of stratum 1
        - N1 is the population size of stratum 1
        - N is the total population size
        - n is the sampling size
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    :seed: sampling seed
    :keep_index: if True, it keeps a column with the original population index indicator

    Returns
    -------
    A sampled pandas dataframe based in a set of strata.
    Examples
    --------
    >> df.head()
    	id  sex age city
    0	123 M   20  XYZ
    1	456 M   25  XYZ
    2	789 M   21  YZX
    3	987 F   40  ZXY
    4	654 M   45  ZXY
    ...
    # This returns a sample stratified by sex and city containing 30% of the size of
    # the original data
    >> stratified = stratified_sample(df=df, strata=['sex', 'city'], size=0.3)
    Requirements
    ------------
    - pandas
    - numpy
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)

    # controlling variable to create the dataframe or append to it
    first = True
    for i in range(len(tmp_grpd)):
        # query generator for each iteration
        qry=''
        for s in range(len(strata)):
            stratum = strata[s]
            value = tmp_grpd.iloc[i][stratum]
            n = tmp_grpd.iloc[i]['samp_size']

            if type(value) == str:
                value = "'" + str(value) + "'"

            if s != len(strata)-1:
                qry = qry + stratum + ' == ' + str(value) +' & '
            else:
                qry = qry + stratum + ' == ' + str(value)

        # final dataframe
        if first:
            stratified_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            first = False
        else:
            tmp_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            stratified_df = stratified_df.append(tmp_df, ignore_index=True)

    return stratified_df



def stratified_sample_report(df, strata, size=None):
    '''
    Generates a dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Returns
    -------
    A dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)
    return tmp_grpd


def __smpl_size(population, size):
    '''
    A function to compute the sample size. If not informed, a sampling
    size will be calculated using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Parameters
    ----------
        :population: population size
        :size: sample size (default = None)
    Returns
    -------
    Calculated sample size to be used in the functions:
        - stratified_sample
        - stratified_sample_report
    '''
    if size is None:
        cochran_n = round(((1.96)**2 * 0.5 * 0.5)/ 0.02**2)
        n = round(cochran_n/(1+((cochran_n -1) /population)))
    elif size >= 0 and size < 1:
        n = round(population * size)
    elif size < 0:
        raise ValueError('Parameter "size" must be an integer or a proportion between 0 and 0.99.')
    elif size >= 1:
        n = size
    return n


############################## Standardize by Site and HCs

def centrar_by(dataframe, by, strata=None, test_size=0.2, seed=None, keep_na_rows=False):
    """
    Warning!!! - Developed for ReDLats pre-existing data only. For other uses it sould be addapted and tested.
    This funtions standardizes numerical data by some specific categorical value. Instead of taking the mean and standard deviation of the full array it will use these values within a defined group

    Parameters
    ----------
    dataframe: Pandas dataframe to standardize
    by: list like ['column_name', value]. Column name and value to use. ie ['diagnosis', 0]
    strata: List like. Categorical variables to use as strata for splitting in train and test. First column will be also used to standardize within groups.
    keep_na_rows: bool. Keep or drop rows with missing values. Default is to drop. Not droping NaNs may cause conflicts.
    test_size: float. Defines the % of rows for train and test, default is 0.2 (20% for test).
    seed: Int.

    Returns
    --------
    Train and test set + Standardized columns by a categorical column.

    Requirements
    ------------
    - pandas
    - numpy
    - sklearn
    - stratified_sample
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    column = by[0]
    value = by[1]
    strata1, strata2 = strata[0], strata[1]
    # print(f"Column and categorical value to use for stadardization: {column, str(value)}\nColumns to use for stratified sample: {strata1, strata2}\n")
    train_size, test_size = 1-test_size, test_size
    data = dataframe.copy()
    df = dataframe.copy()

    if keep_na_rows==False:
        df.dropna(axis=0, inplace=True)
    else:
        nans = df.isna().sum()
        if nans > 0:
            print(f"Warning! {df.isna().sum()} Missing values")
        else:
            print('No NaNs found')

    # condition
    df['condition'] = np.where(df[column] == value, 1,0)
    # strata_col
    df['strata_col'] = df[strata1].astype('str') + " / " + df[strata2].astype('str')
    # controles
    controles = df.loc[df['condition'] ==1]
    train_controles = stratified_sample(controles, strata=['strata_col'], size=int(controles.shape[0]* train_size), seed=seed, keep_index=True)
    test_controles = controles.drop(labels=train_controles['index'], axis=0)
    #Category to exclude from diagnosis variable
    df = df.loc[df['condition'] !=1]
    df.reset_index(drop=True, inplace=True)

    # # split
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, t_test = train_test_split(X,y, test_size=test_size, stratify=y, random_state=seed)
    y_train = X_train['diagnosis'].reset_index(drop=True)
    X_train = X_train.drop(['diagnosis', 'condition'], axis=1).reset_index(drop=True)
    y_test = X_test['diagnosis'].reset_index(drop=True)
    X_test = X_test.drop(['diagnosis', 'condition'], axis=1).reset_index(drop=True)

    X_train_index = X_train.index.tolist()
    X_test_index = X_test.index.tolist()

    #print(X_train.shape, X_train.Country.unique())
    #print(X_test.shape, X_test.Country.unique())

    # Standarize Train set
    cols = X_train.columns[3:]
    strata1_values = X_train[strata1].unique()

    for value in strata1_values:
        # controles del site
        train_centro = train_controles.loc[train_controles[strata1] == value]
        if len(train_centro)>0:
            #media y desvío del site
            train_mean = train_centro[cols].mean(skipna=True)
            train_std = train_centro[cols].std(skipna=True)
            for j in range(len(cols)):
                X_train[str(cols[j])+'_z'] = np.nan
                X_train[str(cols[j])+'_z'] = (X_train[cols[j]] - train_mean[j]) / train_std[j]

        else:
            # print(value, "(train set) Failed, using other sites data")
            train_otros_centros = train_controles.loc[train_controles[strata1] != value]
            train_mean_oc = train_otros_centros[cols].mean(skipna=True)
            train_std_oc = train_otros_centros[cols].std(skipna=True)
            for j in range(len(cols)):
                X_train[str(cols[j])+'_z'] = np.nan
                X_train[str(cols[j])+'_z'] = (X_train[cols[j]] - train_mean_oc[j]) / train_std_oc[j]



    # Standarize Test set
    for value in strata1_values:
        # controles del site
        test_centro = test_controles.loc[test_controles[strata1] == value]
        if len(test_centro)>0:
            #media y desvío del site
            test_mean = test_centro[cols].mean(skipna=True)
            test_std = test_centro[cols].std(skipna=True)
            for j in range(len(cols)):
                X_test[str(cols[j])+'_z'] = np.nan
                X_test[str(cols[j])+'_z'] = (X_test[cols[j]] - test_mean[j]) / test_std[j]

        else:
            # print(value, "(test set) Failed, using other sites data")
            test_otros_centros = test_controles.loc[test_controles[strata1] != value]
            test_mean_oc = test_otros_centros[cols].mean(skipna=True)
            test_std_oc = test_otros_centros[cols].std(skipna=True)
            for j in range(len(cols)):
                X_test[str(cols[j])+'_z'] = np.nan
                X_test[str(cols[j])+'_z'] = (X_test[cols[j]] - test_mean_oc[j]) / test_std_oc[j]


    # train_inf = np.isinf(X_train.select_dtypes(include=np.number)).any().sum()
    # test_inf = np.isinf(X_test.select_dtypes(include=np.number)).any().sum()
    # check_inf =train_inf+test_inf
    # print(check_inf)
    # print(np.isinf(X_test.select_dtypes(include=np.number)).any())
    # print(np.isinf(X_train.select_dtypes(include=np.number)).any())
        # train = pd.concat([X_train, y_train], axis=1)
        # train.replace([np.inf, -np.inf], 0, inplace=True)
        # test = pd.concat([X_test, y_test], axis=1)
        # test.replace([np.inf, -np.inf], 0, inplace=True)
    return X_train, X_test, y_train, y_test
        # return X_train, X_test, y_train, y_test

###################################   Confidence Inntervals

def confidence_interval(p, n, type='gaussian', conf=0.95, outcome='interval'):
    """
    Function to calculate confidence intervals (CI) for 0-1 range metrics, e.i. Accuracy , Recall, ROC AUC, etc

    Args:
        p (float, np.array): Proportion for calculating the CI, between 0 an 1. For non-parametric an array of n>6 scores or similar must be passed.
        n (int): number of observed cases or sample.
        type (str): Distribution assumption. Valids inputs are "gaussian" and 'non-parametric'. Defaults to 'gaussian'. Use statsmodels.stats.proportion.proportion_confint for Bernoulli distributions.
        conf (float): Level of confidence. Valids inputs are .90, .95, .98, and .99. Defaults to 95.


    Returns:
    lower and upper margins and prints results

    Examples:
        lower, upper = CI(p=0.9, n=1000)
        >>> CI = [0.88140580735821, 0.9185941926417901]

        accuracy_scores = np.array([0.96,0.94,0.93,0.91,0.89,0.885,0.88])
        a, b = CI(n=100, p=accuracy_scores, type='non-parametric', conf=0.90)
        >>> 2.5th percentile = 0.8808, 97.5th percentile = 0.957


    """
    from numpy import sqrt, percentile, round

    critical_val = 1.960


    if conf not in (.90,.95,.98,.99):
        print("Warning! Wrong input. Using 95% confidence.\nValid values are 90, 95, 98 or 99")

    if conf == .90:
        critical_val = 1.645
    elif conf == .98:
        critical_val = 2.326
    elif conf == .99:
        critical_val = 2.576
    else:
        critical_val = critical_val

    if type == 'gaussian':
        interval = critical_val * sqrt( p * (1-p) / n)
        lower = round((p -interval), 4)
        upper = round((p + interval),4)
#        print(f"CI = [{lower}, {upper}]")
        if outcome == 'interval':
            return interval
        else:
            return lower, upper

    else:
        alpha = 1-conf
        lower_p = alpha/2 *100
        lower_percentile = round(max(0.0, percentile(p, lower_p)), 4)
        upper_p = (1 - (alpha/2))*100
        upper_percentile = round(min(1.0, percentile(p, upper_p)), 4)
        print(f"2.5th percentile = {lower_percentile}, 97.5th percentile = {upper_percentile}")
        return lower_percentile, upper_percentile

###################################   Normality tests
def ntest(data, test='shapiro', plot=False):
    """
    Description:

    Function to perform normality test with 95% confidence. Applies scipy implementation of Shapiro-Wilk,  D’Agostino’s K-squared test, Chi-square, Jarque-Bera and Statsmodels Lilliefors.

    Parameters:
    data: (float) numpy array or list like of values to testing
    test: (str) valid values are : "shapiro" (Default), "dagostino", "chi" "Jarquebera", and 'lilliefors'
    plot: (bool) If True returns Histogram and qq plots. Deault is set to False

     Returns:
     Statistic, p-value and interpretation
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import shapiro, normaltest, anderson, chisquare, jarque_bera
    from statsmodels.stats.diagnostic import lilliefors

    if str(type(data)) == "<class 'numpy.ndarray'>":
        data = data
    else:
        data = np.array(data)
    
    data = data[~np.isnan(data)]

    if plot == True:
        print('not implemented yet!')
    else:
        pass

    if test in ('shapiro', 'dagostino', 'chi', 'jarquebera', 'lilliefors'):
        test = test
    else:
        print(f"Valid tests are 'shapiro', 'dagostino', 'chi', 'jarquebera' or 'lilliefors'.\nSetting parameter to default value.")
        test ='shapiro'

    if test == 'shapiro':
        stat, p = shapiro(data)
        print(f"statistic = {stat}, p-value = {p}")
        if p > 0.05:
            print("Can't Reject H0. Probably Normal Distribution")
        else:
            print("Reject H0. Probably NOT Normal Distribution")

    elif test == 'dagostino':
        stat, p = normaltest(data)
        print(f"statistic = {stat}, p-value = {p}")
        if p > 0.05:
            print("Can't Reject H0. Probably Normal Distribution")
        else:
            print("Reject H0. Probably NOT Normal Distribution")

    elif test == 'chi':
        stat, p = chisquare(data)
        print(f"statistic = {stat}, p-value = {p}")
        if p > 0.05:
            print("Can't Reject H0. Probably Normal Distribution")
        else:
            print("Reject H0. Probably NOT Normal Distribution")

    elif test == 'lilliefors':
        stat, p = lilliefors(data)
        print(f"statistic = {stat}, p-value = {p}")
        if p > 0.05:
            print("Can't Reject H0. Probably Normal Distribution")
        else:
            print("Reject H0. Probably NOT Normal Distribution")
    elif test == 'jarquebera':
        stat, p = jarque_bera(data)
        print(f"statistic = {stat}, p-value = {p}")
        if p > 0.05:
            print("Can't Reject H0. Probably Normal Distribution")
        else:
            print("Reject H0. Probably NOT Normal Distribution")
    else:
        print("error")


####################### ECDF

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


########################## Bootstrap replicate
def bootstrap_replicate_1d(data, func):
    """Generate bootstrap replicate of 1D data."""
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)


######################### Draw Bootstrap replicate

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size=size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

######################### A function to do pairs bootstrap (slope and intercept for regression model)

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression.

# Generate replicates of slope and intercept using pairs bootstrap:

bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy, fertility, size=1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5, 97.5]))

# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, normed=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()

"""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps
