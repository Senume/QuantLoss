import numpy as np

def forwPass(X, w):
    '''
    Functionality module to do a forward propagation through given weights and input data.
    
    Parameters
    ----------
    X : numpy ndarray
       input data
    w : numpy ndarray
       weights to be used for forward propagation
    Returns
    -------
    numpy ndarray
       output of forward propagation

    '''
    return X@w.T

def lploss(processed_y, true_y, p = 2):
    '''
    Functionality module to calculate the loss of the model.
    
    Parameters
    ----------
    processed_y : numpy ndarray
        output of forward propagation on quantized weights
    true_y : numpy ndarray
        output of forward propagation on orginal weights
    p : int (default = 2)
        dimension space of lp norm

    '''

    # Checking parameter instance type
    if not isinstance(processed_y, np.ndarray):
        raise TypeError("'processed_y' must be a numpy ndarray.")
    
    if not isinstance(true_y, np.ndarray):
        raise TypeError("'true_y' must be a numpy ndarray.")
    
    if not isinstance(p, int):
        raise TypeError("'p' must be an integer.")

    return (np.sum(np.abs(true_y - processed_y)**p))**(1/p)

# Finding losses for different range threshold
def findLossPerThreshold(X, w, true_y, callbacklossfn, step = 0.001, p = 2 ):
    '''
    Functionality module to calculate the loss of the weight for varying thresholds.

    Parameters
    ----------
    X : numpy ndarray
        input data for the forward popogation
    w : numpy ndarray
        copy of the weights analysis
    true_y : numpy ndarray
        output of forward propagation on orginal weights
    callbacklossfn : function
        loss function to calculate the loss of original output with output of pruned weights outcome
    step : float (default = 0.001)
        step size to create evenly spaced valued over min and max of the original weights
    p : int (default = 2)
        dimension space of lp norm

    Returns
    -------
    (list, list)
        holds the tuple of thresholds and respective loss values (magnitude, loss)
    '''
    # Check parameters instance
    if len(X.shape) != 2 or len(w.shape) != 2:
        raise ValueError("'X', 'w' must be a 2D array")
    
    if not callable(callbacklossfn):
        raise TypeError("'callbacklossfn' must be a callback loss function.")

    # Min and max range value of weight
    local_min = np.min(w)
    local_max = np.max(w)

    # Number of point instances between min and max
    points = int((local_max - local_min)/step)
    print("Local minimum: ", local_min, " Local max: ", local_max, "Points: ", points)

    # Initialize the range of threshold values temporary variable to hold the loss
    mags = list(np.linspace(local_min, local_max, num= points, endpoint=False))
    losses = []

    # For each range of points as thresholds, compute the loss    
    for mag in mags:
        w[(w < mag)] = 0
        
        y = forwPass(X, w)
        loss = callbacklossfn(y, true_y, p)
        
        losses.append(loss)

    return mags, losses

def SmoothRollingAverage(samples, window_size=5, mode='same'):
    """
    Smoothing given data via rolling average.

    Parameters
    ----------
    samples : numpy ndarray (1D)
        data samples on which smoothing is applied
    window_size : int (default = 5)
        convolution of kernel size
    mode : str (default ='same')
        covolution mode type

    Returns
    -------
    numpy ndarray (1D)
        returns smooth data
    """
    if not isinstance(samples, np.ndarray):
        raise TypeError("'samples' must be a numpy ndarray.")

    kernel = np.ones(window_size)/window_size
    conv_output = np.convolve(samples, kernel, mode)
    return conv_output

def findRanges(True_Index, Slope, Threshold):
    """
    Functionality to extracted region based on binary indicator array

    Parameters
    ----------
    True_Index : numpy ndarray (1D)
        Array which holds the the importance of elements at corresponding indices, represented as 0 or 1
    Slope : numpy ndarray (1D)
        Array of slopes values 
    Threshold : numpy ndarray (1D)
        Array of thresholds values

    Returns
    -------
    list(tuple)
        returns the list of extracted region as tuple, containing the subarray of slope values and thresholds values
    """

    # Holdding all the region of interest
    ListOfRegions = []

    # Temporary variable to hold the region of iteration
    SlopeRegion = []
    ThresholdRegion = []

    # For each individual element in the Slope 
    for index, value in enumerate(True_Index):

        # Case when the element is saved as a region
        if value == 1:
            SlopeRegion.append(Slope[index])
            ThresholdRegion.append(Threshold[index])
        
        # When the region is interupted, saving the observed region
        elif value == 0 and SlopeRegion and ThresholdRegion:
            ListOfRegions.append((np.array(SlopeRegion), np.array(ThresholdRegion)))
            SlopeRegion = []
            ThresholdRegion = []
    # Termination case if last region is not interrupted and iterated till end
    if SlopeRegion and ThresholdRegion:
        ListOfRegions.append((np.array(SlopeRegion), np.array(ThresholdRegion)))

    # Returning the list of regions
    return ListOfRegions

def findLargestRegion(ListOfRegions, Range):
    """
    Function helps is finding the first two largest ranges in given list of ranges.

    Parameters
    ----------
    ListOfRegions : list(tuple)
        Contains the list of ranges in tuple (slopes values array, thresholds values array)
    Range : float
        Magnitude of difference between min max thresholds values

    Returns
    -------
    list(tuple)
        returns only the first two largest regions.

    """
    # Temporary variable to have the index of the largest ranges
    #(ratio of weight range coverage, index)
    FirstRegionIndex = (-1, -1)
    SecondRegionIndex = (-1,-1)

    # For each individual element in the ListOfRegions
    for index, (SlopeRegion, ThresholdRegion) in enumerate(ListOfRegions):
        subRange = np.max(ThresholdRegion) - np.min(ThresholdRegion)

        # Checking percentage of region it convers
        if subRange/Range > FirstRegionIndex[0]:
            SecondRegionIndex = FirstRegionIndex
            FirstRegionIndex = (subRange/Range, index)
        elif subRange/Range > SecondRegionIndex[0]:
            SecondRegionIndex = (subRange/Range, index)

    print("Ratio of first region range coverage:", FirstRegionIndex[0], "Region of selection index: ", FirstRegionIndex[1])
    print("Ratio of second region range coverage:", SecondRegionIndex[0], "Region of selection index: ", SecondRegionIndex[1])
    
    # returns the region
    return [ListOfRegions[FirstRegionIndex[1]], ListOfRegions[SecondRegionIndex[1]]]

def findSlope(X, Range):
    '''
    For given time-series signal, it finds the slope of the signal at each point

    Parameters
    ----------
    X : numpy ndarray (1D)
        Time series signal
    Range : numpy ndarray (1D)
        Range points of the signal

    Returns
    -------
    (numpy ndarray (1D), numpy ndarray (1D))
        returns the slope of the signal at each point with adjusted range points
    '''
    slope = X[1:] - X[:-1]

    return (slope, Range[:-1])

def findThreshold(signal, sensitivity=0.5):
    '''
    Functionality to find the cut threshold value for given signal.

    Parameter
    ---------
    signal : numpy ndarray (1D)
        To which threshold  need to be calculated.
    sensitivity : float (default = 0.5)
        hyper parameter to adjust the threshold strength.

    Returns
    -------
    float
        Returns the threshold

    '''
    return sensitivity*np.mean(signal)