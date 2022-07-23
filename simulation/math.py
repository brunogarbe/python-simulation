# Abs	Output absolute value of input
# Add	Add or subtract inputs
# Algebraic Constraint	Constrain input signal
# Assignment	Assign values to specified elements of signal
# Bias	Add bias to input
# Complex to Magnitude-Angle	Compute magnitude and/or phase angle of complex signal
# Complex to Real-Imag	Output real and imaginary parts of complex input signal
# Divide	Divide one input by another
# Dot Product	Generate dot product of two vectors
# Find Nonzero Elements	Find nonzero elements in array
# Gain	Multiply input by constant
# Magnitude-Angle to Complex	Convert magnitude and/or a phase angle signal to complex signal
# Math Function	Perform mathematical function
# MinMax	Output minimum or maximum input value
# MinMax Running Resettable	Determine minimum or maximum of signal over time
# Permute Dimensions	Rearrange dimensions of multidimensional array dimensions
# Polynomial	Perform evaluation of polynomial coefficients on input values
# Product of Elements	Copy or invert one scalar input, or collapse one nonscalar input
# Product, Matrix Multiply	Multiply and divide scalars and nonscalars or multiply and invert matrices
# Real-Imag to Complex	Convert real and/or imaginary inputs to complex signal
# Reshape	Change dimensionality of signal
# Rounding Function	Apply rounding function to signal
# Sign	Indicate sign of input
# Sine Wave Function	Generate sine wave, using external signal as time source
# Slider Gain	Vary scalar gain using slider
# Sqrt	Calculate square root, signed square root, or reciprocal of square root
# Squeeze	Remove singleton dimensions from multidimensional signal
# Trigonometric Function	Specified trigonometric function on input
# Unary Minus	Negate input
# Vector Concatenate, Matrix Concatenate	Concatenate input signals of same data type to create contiguous output signal
# Weighted Sample Time Math	Support calculations involving sample time

from .core import BasicBlock
from scipy.interpolate import interp1d

#from enum import Enum
import numpy as np

# Bias	Add bias to input

class Gain(BasicBlock):
    
    def __init__(self, gain=[1], name=None):
        super().__init__(name=name)

        self.is_direct_feedthrough = True
        
        for _ in range(len(gain)):
            #print('!')
            self.add_input()
            self.add_output()
            
        self.gain = np.array(gain)

    def output(self, t, x, u):
        return self.gain * u


class Sum(BasicBlock):
    
    def __init__(self, coef=[1, -1], name=None):
        super().__init__(name=name)

        self.is_direct_feedthrough = True

        for _ in range(len(coef)):
            self.add_input()
        self.add_output('output')

        self.coefs = np.array(coef)

    def output(self, t, x, u):
        #print(u, np.sum(self.coefs * u))
        return np.sum(self.coefs * u)

class Interpolate1D(BasicBlock):
    
    def __init__(self, x, y, kind='linear', name=None):
        super().__init__(name=name)

        self.is_direct_feedthrough = True

        self.add_input()
        self.add_output('output')

        self.interp_func = interp1d(x, y, kind='cubic')

    def output(self, t, x, u):
        return self.interp_func(u)