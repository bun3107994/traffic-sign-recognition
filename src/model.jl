using Flux

function build_model()
    model = Chain(
        # Block 1
        Conv((3,3), 3=>32, relu, pad=1),
        BatchNorm(32),
        MaxPool((2,2)),

        # Block 2
        Conv((3,3), 32=>64, relu, pad=1),
        BatchNorm(64),
        MaxPool((2,2)),

        # Block 3
        Conv((3,3), 64=>128, relu, pad=1),
        BatchNorm(128),
        MaxPool((2,2)),

        # Block 4 - new extra block
        Conv((3,3), 128=>256, relu, pad=1),
        BatchNorm(256),

        # Flatten and classify
        Flux.flatten,
        Dense(4096, 512, relu),
        Dropout(0.3),
        Dense(512, 128, relu),
        Dropout(0.2),
        Dense(128, 43)
    )
    return model
end