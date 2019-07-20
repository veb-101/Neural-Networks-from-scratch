import numpy as np


class Conv3x3(object):
    # 3x3 filter Convolution layer

    def __init__(self, num_filters):
        self.num_filters = num_filters
        # (.../9) to reduce variance of out initial filters
        self.filters = np.random.randn(num_filters, 3, 3, ) / 9

    def iterate_regions(self, image):
        '''
        Generate all possible 3x3 image regions using valid padding
        Xavier initialization
        - image is a 2d numpy array

        conv output dimension formula : ((n + 2p -f)/s +1)x((n + 2p -f)/s +1)
        n = numOfPixels in width or height of image
        p = padding
        f = filter size
        s = strides

        Here 28 x 28 = ((28 + 2*0 - 3)/1 + 1) x ((28 + 2*0 - 3)/1 + 1)
                     = 26 x 26
                     = (Height - 2) x (Width - 2)  
        '''

        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:i+3, j:j+3]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs forward pass using the given input
        Returns a 3-d numpy array (h, w, num_filters)
        - input is a 2d numpy array
        '''
        self.last_input = input

        h, w = input.shape

        output = np.zeros(shape=(h-2, w-2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    def backprop(self, dL_dout, learning_rate):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        dL_dfilters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                dL_dfilters[f] += dL_dout[i, j, f] * im_region
        '''
        At each pass we update each filter (f, f) by multiply the image region
        with that filter's output which simultaneously update all the weights in
        the filter.
        We do this for each filter
        # Take one output unit error for a filter
        # Multiply it by the image region to give all the gradients of the weights
        of the filter wrt. that image region (we update the matrix of zeros for first filter and so on...)
        # Take output unit error at the same postion as first for another filter 
        eg. filter 1 output errors: dE/doutF(1)11, dE/doutF(1)12, dE/doutF(1)21, dE/doutF(1)22
        filter 2 output errors:  dE/doutF(2)11, dE/doutF(2)12, dE/doutF(2)21, dE/doutF(2)22
        take dE/doutF(1)11 * image region which gives filter1 weights error wrt to first image region
        take dE/doutF(2)11 * image region which gives filter2 weights error wrt to first image region

        # Continuously adding all the errors wrt to all image regions in the end
        gives the final dE/dFilter(1) matrix wrt to the whole matrix
        similarly for other filters
        '''

        # Update filters
        self.filters -= learning_rate * dL_dfilters

        # We aren't returning anything here since we use Conv3x3 as
        # the first layer in our CNN. Otherwise, we'd need to return
        # the loss gradient for this layer's inputs, just like every
        # other layer in our CNN.
        return None
