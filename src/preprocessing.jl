using Statistics

const IMG_SIZE = 32

function normalize(arr)
    return (arr .- 0.5f0) ./ 0.5f0
end

function preprocess_images(images)
    println("Preprocessing images...")
    processed = [normalize(img) for img in images]
    println("Done! $(length(processed)) images preprocessed.")
    return processed
end

function one_hot(labels)
    n = length(labels)
    encoded = zeros(Float32, 43, n)
    for (i, label) in enumerate(labels)
        encoded[label + 1, i] = 1.0f0
    end
    return encoded
end

function prepare_batches(images, labels; batch_size=16)
    println("Preparing batches...")
    n = length(images)
    batches = []
    for i in 1:batch_size:n
        batch_end = min(i + batch_size - 1, n)
        batch_size_actual = batch_end - i + 1
        X = zeros(Float32, IMG_SIZE, IMG_SIZE, 3, batch_size_actual)
        for (j, idx) in enumerate(i:batch_end)
            img = images[idx]
            X[:,:,:,j] = permutedims(img, (2, 3, 1))
        end
        Y = one_hot(labels[i:batch_end])
        push!(batches, (X, Y))
    end
    println("Done! $(length(batches)) batches of size $batch_size created.")
    return batches
end