
import numpy

def gaussian2D(shape, sigma = 1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = numpy.ogrid[-m:m+1,-n:n+1]

    h = numpy.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return h

def gaussianMargin2D(margin, sigma = 1):
    left, top, right, bottom = margin
    y, x = numpy.ogrid[-top:bottom + 1, -left:right + 1]
    
    h = numpy.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return h