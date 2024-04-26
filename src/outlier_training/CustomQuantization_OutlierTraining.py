import numpy as np                                  # type: ignore
import matplotlib.pyplot as plt                     # type: ignore

from utils_tensor import *

class CustomQuantization:

    def __init__(self):

        self.quant_weights = None
        self.pruneIndexWeight = None
        self.outlierIndex = None
        self.mapping = None

        self.firstRange = None
        self.secondRange = None

    def extractRange(self, original_weights, input_x = None, smoothing_window_size = 25, sensitivity = 0.5, save_plot=False, plot_path=None, step = 500):
        '''
        Extraction of the configuration and required processes are performed. Then the quantization process is performed.

        Parameters
        ----------
        original_weights : numpy ndarray
            original weight of the dense layer
        input_x : numpy ndarray (2D)
            sample on which quantization is performed
        smoothing_window_size : int
            size of the smoothing window for smoothrollingAverage
        sensitivity : float
            factor impact of the threshold for finding suitable region
        save_plot : bool
            if true, saves the possible weight region plot, else none
        plot_path : str
            location of the plot to saved in
        '''

        if input_x is None:
            input_x = torch.randn(1, original_weights.shape[1])
        
        if len(input_x.shape) != 2:
            raise ValueError("'input_x' must be a 2D array")

        # Extraction of original weights forward-pass output
        True_Y = original_weights@input_x.T

        # Finding the range loss quantity
        Range, Loss = findLossPerThreshold(input_x, original_weights, True_Y, lploss, step=step)

        # Finding slope of it loss variation and smoothing the slope signal
        Slope, Range = findSlope(Loss, Range)
        smooth_Slope = SmoothRollingAverage(Slope, window_size=smoothing_window_size)

        # Threshold to select the region of interest
        cut_Threshold = findThreshold(smooth_Slope, sensitivity)

        # original_weight range of acceptence
        accepted_Index = smooth_Slope > cut_Threshold

        if save_plot:
            if isinstance(plot_path, str):
                plt.plot(Range, smooth_Slope, Range, accepted_Index)
                plt.title('REGION OF POSSIBLE SELECTION RANGE PLOT')
                plt.xlabel('thresholds')
                plt.ylabel('smoothened slope')
                plt.savefig(plot_path)
                plt.close()
            else:
                raise ValueError("'plot_path' must be a string")

        # Finding the suitable range of weight values
        Ranges = findRanges(accepted_Index, smooth_Slope, Range)
        R = torch.max(Range) - torch.min(Range)

        # Selected Region range and updatation
        [FirstRange, SecondRange] = findLargestRegion(Ranges, R)

        #  Updation
        self.firstRange = (torch.min(FirstRange[1]), torch.max(FirstRange[1]))
        self.secondRange = (torch.min(SecondRange[1]), torch.max(SecondRange[1]))

        # Rearanging based on the location of the selected region
        if self.firstRange[1] > self.secondRange[0]:
            self.firstRange, self.secondRange = self.secondRange, self.firstRange

        print('First Region Range: ', self.firstRange)
        print('Second Region Range', self.secondRange)

    def extractOutlierIndex(self, original_weights):
        '''
        Extraction of outlier indices from original weights

        Parameters
        ----------
        original_weights : numpy ndarray
                weight matrix of the layer

        '''
        Index = findOutliers(original_weights)
        self.outlierIndex = Index

    def proceedQuantization(self, original_weight):
        '''
            Prceeds with setting quantizing configuration and quantization of weights
            
            Parameters
            ----------
            original_weight : numpy ndarray
                weight matrix which needs to be quantized
        '''
        if original_weight is None:
            raise ValueError("Original weight cannot be None")

        # Finding the level mapping 
        self.quantizeConfig()

        # Quantization of the weights
        self.quantize(original_weight)

        # Finding the pruning index from the matrix
        self.pruneIndex(original_weight, (self.firstRange[1], self.secondRange[0]))

        # Finding outlier index from the weight matrix
        self.extractOutlierIndex(original_weight)

    def quantizeConfig(self):
        """
        Functions helps in finding the levels and returns the level mapping and and region ranges

        Parameters
        ----------
        Ranges : list(tuple)
            Selected region of to quantize for.

        Returns
        -------
        (dict, [tuples])
            Returns the level mapping and region ranges
        """

        [FirstRange, SecondRange] = [self.firstRange, self.secondRange]

        # First Range
        FirstStartThreshold, FirstEndThreshold = FirstRange[0], FirstRange[1]

        possible_levels = torch.linspace(FirstStartThreshold, FirstEndThreshold, steps= 4)
        FirstLevelMapping = dict(zip(range(2), possible_levels[1:3]))

        # Second Range
        SecondStartThreshold, SecondEndThreshold = SecondRange[0], SecondRange[1]

        possible_levels = torch.linspace(SecondStartThreshold, SecondEndThreshold, steps=4)
        SecondLevelMapping = dict(zip(range(2, 4), possible_levels[1:3]))

        # Merging levels maps
        LevelMapping = FirstLevelMapping | SecondLevelMapping
        
        # Updation
        self.mapping = LevelMapping

    def quantize(self, original_weight):
        '''
            Functionality module for quantizing the weights
            
            Parameters
            ----------
            original_weight : numpy ndarray
                weight matrix which needs to be quantized
            self.mapping : dict
                quantization points

        '''
        M1 = original_weight - self.mapping[0]
        M2 = original_weight - self.mapping[1]
        M3 = original_weight - self.mapping[2]
        M4 = original_weight - self.mapping[3]

        Subs = torch.dstack((np.abs(M1),np.abs(M2),np.abs(M3),np.abs(M4)))
        Quant = torch.argmin(Subs,axis=2)

        self.quant_weights = Quant

    def dequantize(self):

        """
        Functionality module for dequantizing the weights
            
        Returns
        -------
        torch tensor (2D)
        Dequantized weights

        """
        level0 = (self.quant_weights == 0)*self.mapping[0]
        level1 = (self.quant_weights == 1)*self.mapping[1]
        level2 = (self.quant_weights == 2)*self.mapping[2]
        level3 = (self.quant_weights == 3)*self.mapping[3]

        dequantize_weight = level0 + level1 + level2 + level3
        PruningMatrix = self.pruneIndexWeight != 1

        return (dequantize_weight* PruningMatrix).clone().to(torch.float32)

    def pruneIndex(self, original_weight, Limits):
        """
        For given weight range, finds the index of pruning weights and returns the corresponding index as sparse
        
        Parameters
        ----------
        original_weight : numpy ndarray
            original layer weight matrix
        range : tuple
            holds the limits of the pruning weights range (default around zero)

        Returns
        -------
        numpy sparse ndarray
        Pruning weights indexes

        """
        # Pruning conditions 
        LeftCase = original_weight > Limits[0]
        RightCase = original_weight < Limits[1]
        Index = LeftCase * RightCase
        
        # Updation
        self.pruneIndexWeight = Index
