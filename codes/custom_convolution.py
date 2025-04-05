

# Optimized CNN Layer Function
def cnn_layer(image_volume, filter, pad=1, step=1):
    k_size = filter.shape[1]  # Kernel size
    n_filters = filter.shape[0]  # Number of filters

    # Compute output dimensions
    width_out = (image_volume.shape[0] - k_size + 2 * pad) // step + 1
    height_out = (image_volume.shape[1] - k_size + 2 * pad) // step + 1

    # Ensure correct padding
    image_padded = np.pad(image_volume, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

    # Prepare output feature maps
    feature_maps = np.zeros((width_out, height_out, n_filters))

    # Apply convolution across filters
    for i in range(n_filters):
        convolved_image = np.sum(
            [
                convolution_2d(image_padded[:, :, j], filter[i, :, :, j], pad=0, step=step)
                for j in range(image_volume.shape[-1])
            ],
            axis=0
        )

        # Ensure the output shape matches expected shape
        feature_maps[:, :, i] = convolved_image[:width_out, :height_out]  # Trim excess padding

    return feature_maps



def convolution_loops(image, filter, output_image, step):
    output_height, output_width = output_image.shape

    for i in range(0, output_height, step):
        for j in range(0, output_width, step):
            patch_from_image = image[i:i+filter.shape[0], j:j+filter.shape[1]]

            # Manual element-wise multiplication
            element_wise_result = np.zeros_like(patch_from_image)
            for m in range(filter.shape[0]):
                for n in range(filter.shape[1]):
                    # element_wise_result[m, n] = patch_from_image[m, n] * filter[m, n]
                    # element_wise_result[m, n] = Binary_Multiply8_half_approx(int(patch_from_image[m, n]), int(filter[m, n]))
                    element_wise_result[m, n] = Binary_Multiply8_full_approx(int(patch_from_image[m, n]), int(filter[m, n]))

            # Summing the element-wise multiplied values
            output_image[i, j] = np.sum(element_wise_result)

    return output_image


class CustomConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding=1, predefined_kernel=None, **kwargs):
        super(CustomConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.predefined_kernel = predefined_kernel  # New parameter

    def build(self, input_shape):
        # Initialize trainable filters (compatible with NumPy-based function)
        if self.predefined_kernel is not None:
            # Convert to NumPy if it's a Tensor
            if isinstance(self.predefined_kernel, tf.Tensor):
                self.predefined_kernel = self.predefined_kernel.numpy()
            kernel_initializer = tf.constant_initializer(self.predefined_kernel)
        else:
            kernel_initializer = "glorot_uniform"

        self.kernel = self.add_weight(
            shape=(self.filters, self.kernel_size, self.kernel_size, input_shape[-1]),
            initializer=kernel_initializer,
            trainable=True,
        )

    def call(self, inputs):
        def numpy_convolution(inputs_np, kernel_np):
            """ Wrapper function to apply NumPy convolution per batch """
            batch_size = inputs_np.shape[0]
            output_list = []

            for i in range(batch_size):
                output = cnn_layer(inputs_np[i], kernel_np, pad=self.padding, step=self.strides)
                output_list.append(output)

            return np.array(output_list, dtype=np.float32)  # Convert back to NumPy array

        # Use tf.py_function to execute NumPy-based convolution in TensorFlow
        output = tf.py_function(func=numpy_convolution, inp=[inputs, self.kernel], Tout=tf.float32)

        # Calculate output shape based on input shape
        batch_size = tf.shape(inputs)[0]
        height = (tf.shape(inputs)[1] - self.kernel_size + 2 * self.padding) // self.strides + 1
        width = (tf.shape(inputs)[2] - self.kernel_size + 2 * self.padding) // self.strides + 1

        # Ensure proper shape so Flatten() can work
        output = tf.reshape(output, (batch_size, height, width, self.filters))

        return output

# **Define a predefined 3x3 kernel (Example: Edge Detection)**
predefined_kernel = np.array([
    [[[1]], [[0]], [[0]]],
    [[[0]], [[ 1]], [[0]]],
    [[[0]], [[0]], [[1]]]
], dtype=np.float32)  # Shape: (3, 3, 1)

# Expand dimensions to match expected shape (filters, kernel_size, kernel_size, input_channels)
predefined_kernel = np.expand_dims(predefined_kernel, axis=0)  # Add filters dimension

# **Using the custom layer in a Keras model**
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    CustomConv2D(filters=1, kernel_size=3, strides=1, padding=1, predefined_kernel=predefined_kernel),  # Using predefined kernel
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Print model summary
model.summary()
