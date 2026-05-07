using Flux
using Statistics

function train_model!(model, train_batches; epochs=30)
    opt = Adam(0.0001)
    opt_state = Flux.setup(opt, model)

    history = Dict(
        "train_loss"     => Float64[],
        "train_accuracy" => Float64[]
    )

    println("Starting training for $epochs epochs...")
    println("="^50)

    for epoch in 1:epochs
        # Reduce learning rate at epoch 15 and 25
        if epoch == 15
            Flux.adjust!(opt_state, 0.00005)
            println("Learning rate reduced to 0.00005")
        elseif epoch == 25
            Flux.adjust!(opt_state, 0.00001)
            println("Learning rate reduced to 0.00001")
        end

        Flux.trainmode!(model)
        epoch_loss = 0.0

        for (X, Y) in train_batches
            loss, grads = Flux.withgradient(model) do m
                predictions = m(X)
                Flux.logitcrossentropy(predictions, Y)
            end
            Flux.update!(opt_state, model, grads[1])
            epoch_loss += loss
        end

        # Calculate accuracy in test mode
        Flux.testmode!(model)
        correct = 0
        total = 0
        for (X, Y) in train_batches
            preds = Flux.onecold(model(X))
            actual = Flux.onecold(Y)
            correct += sum(preds .== actual)
            total += length(actual)
        end

        avg_loss = epoch_loss / length(train_batches)
        accuracy = correct / total * 100

        push!(history["train_loss"], avg_loss)
        push!(history["train_accuracy"], accuracy)

        println("Epoch $epoch | Loss: $(round(avg_loss, digits=4)) | Accuracy: $(round(accuracy, digits=2))%")
    end

    println("="^50)
    println("Training complete!")
    return model, history
end