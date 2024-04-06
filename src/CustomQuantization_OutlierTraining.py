import numpy as np                                  # type: ignore
from scipy import sparse                            # type: ignore

from utils import *

class CustomQuantization:

    def __init__(self):

        self.quant_weights = None
        self.pruneIndexWeight = None
        self.outlierIndex = None
        self.mapping = None

        self.firstRange = None
        self.secondRange = None

    def extractRange(self, original_weights, input_x = None):
        '''
        Extraction of the configuration and required processes are performed. Then the quantization process is performed.

        Parameters
        ----------
        original_weights : numpy ndarray
            original weight of the dense layer
        '''

        if input_x is None:
            input_x = np.random.randn(1, original_weights.shape[1])

        # Extraction of original weights forward-pass output
        True_Y = forwPass(input_x, original_weights)

        # Finding the range loss quantity
        Range, Loss = findLossPerThreshold(input_x, original_weights, True_Y, lploss)
        Range, Loss = np.array(Range), np.array(Loss)

        # Finding slope of it loss variation and smoothing the slope signal
        Slope, Range = findSlope(Loss, Range)
        smooth_Slope = SmoothRollingAverage(Slope, window_size=20)

        # Threshold to select the region of interest
        cut_Threshold = findThreshold(smooth_Slope)

        # original_weight range of acceptence
        accepted_Index = smooth_Slope > cut_Threshold

        # Finding the suitable range of weight values
        Ranges = findRanges(accepted_Index, smooth_Slope, Range)
        R = np.max(Range) - np.min(Range)

        # Selected Region range and updatation
        [FirstRange, SecondRange] = findLargestRegion(Ranges, R)

        #  Updation
        self.firstRange = (np.min(FirstRange[1]), np.max(FirstRange[1]))
        self.secondRange = (np.min(SecondRange[1]), np.max(SecondRange[1]))

    def extractOutlierIndex(self, original_weights):
        '''
        Extraction of outlier indices from original weights

        Parameters
        ----------
        original_weights : numpy ndarray
                weight matrix of the layer

        '''
        StandardDeviation = np.std(original_weights)

        Index = findOutlinear(original_weights, StandardDeviation)
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
        print(FirstRange, SecondRange)

        # First Range
        FirstStartThreshold, FirstEndThreshold = FirstRange[0], FirstRange[1]

        possible_levels = np.linspace(FirstStartThreshold, FirstEndThreshold, num=4)
        FirstLevelMapping = dict(zip(range(2), possible_levels[1:3]))

        # Second Range
        SecondStartThreshold, SecondEndThreshold = SecondRange[0], SecondRange[1]

        possible_levels = np.linspace(SecondStartThreshold, SecondEndThreshold, num=4)
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

        Subs = np.dstack((np.abs(M1),np.abs(M2),np.abs(M3),np.abs(M4)))
        Quant = np.argmin(Subs,axis=2)

        self.quant_weights = Quant

    def dequantize(self):

        """
        Functionality module for dequantizing the weights
            
        Returns
        -------
        numpy ndarray
        Dequantized weights

        """
        level0 = (self.quant_weights == 0)*self.mapping[0]
        level1 = (self.quant_weights == 1)*self.mapping[1]
        level2 = (self.quant_weights == 2)*self.mapping[2]
        level3 = (self.quant_weights == 3)*self.mapping[3]

        dequantize_weight = level0 + level1 + level2 + level3
        PruningMatrix = (self.pruneIndexWeight.toarray() != 1)

        return np.multiply(dequantize_weight, PruningMatrix)

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
        Index = LeftCase & RightCase

        # Conversion to sparse matrix
        SparseIndex = sparse.coo_matrix(Index)
        
        # Updation
        self.pruneIndexWeight = SparseIndex
