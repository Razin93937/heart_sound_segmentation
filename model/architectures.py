from tensorflow.keras import layers, models
from tcn import TCN  # pip: keras-tcn


def TCNGRUModel(
    time_steps: int,
    n_classes: int = 3,
    base_channels: int = 256,
    noise_std: float = 0.01,
):
    """Build the TCN + BiGRU segmentation model."""
    inputs = layers.Input(shape=(time_steps, 1), name="waveform")
    x = layers.GaussianNoise(noise_std, name="gaussian_noise")(inputs)

    # TCN block 1
    x = TCN(
        base_channels,
        kernel_size=2,
        activation="relu",
        return_sequences=True,
        name="tcn_1",
    )(x)
    # Reshape & Pool (time/4, channels/4)
    x = layers.Reshape((time_steps, base_channels, 1), name="reshape_1")(x)
    x = layers.MaxPooling2D(pool_size=(4, 4), name="pool_1")(x)
    x = layers.Reshape((time_steps // 4, base_channels // 4), name="reshape_2")(x)

    # TCN block 2 (half channels)
    x = TCN(
        base_channels // 2,
        kernel_size=2,
        activation="relu",
        return_sequences=True,
        name="tcn_2",
    )(x)
    x = layers.Reshape((time_steps // 4, (base_channels // 2), 1), name="reshape_3")(x)
    x = layers.MaxPooling2D(pool_size=(4, 4), name="pool_2")(x)
    x = layers.Reshape((time_steps // 16, (base_channels // 8)), name="reshape_4")(x)

    # BiGRU layers
    x = layers.Bidirectional(
        layers.GRU(64, return_sequences=True, activation="tanh"), name="bgru_1"
    )(x)
    x = layers.Bidirectional(
        layers.GRU(64, return_sequences=True, activation="tanh"), name="bgru_2"
    )(x)

    # Per-step classifier
    x = layers.TimeDistributed(layers.Dense(128, activation="relu"), name="td_dense_1")(
        x
    )
    outputs = layers.TimeDistributed(
        layers.Dense(n_classes, activation="softmax"), name="td_out"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="HeartSoundSegmentation")
    return model
