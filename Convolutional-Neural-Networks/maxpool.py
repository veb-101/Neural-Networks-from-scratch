import numpy as np


class MaxPool2(object):
    '''
    A Max Pooling layer using stride of 2
    '''

    def iterate_regions(self, image):
        '''
        Generate  non-overlapping 2x2 image regions to pool over
        - image is a 2d numpy array
        '''

        h, w, _ = image.shape

        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2 + 2), (j*2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros(shape=(h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    '''
    A Max Pooling layer can’t be trained because it doesn’t actually have any weights, 
    but we still need to implement a backprop() method for it to calculate gradients

    All the other values that are not maximum in the 2x2 block don't affect the output
    only the max value does.
    So the appropriate gradients are passed along only to those postions (units/neurons)
    which contributed (had) the max input value in that 2x2 region
    '''

    def backprop(self, dL_dout):
        '''
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - dL_dout is the loss gradient for this layer's outputs.
        '''

        dL_dinput = np.zeros(shape=self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):

            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            dL_dinput[i*2 + i2, j*2 + j2,
                                      f2] = dL_dout[i, j, f2]
        return dL_dinput
