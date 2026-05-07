using Pkg
Pkg.activate(".")

using CSV, DataFrames, Plots, Images, Flux
using ImageTransformations

include("src/data_loading.jl")
include("src/preprocessing.jl")
include("src/model.jl")
include("src/train.jl")
include("src/evaluate.jl")

function predict_sign(model, img_array)
    Flux.testmode!(model)
    img_norm = (img_array .- 0.5f0) ./ 0.5f0
    X = permutedims(img_norm, (2, 3, 1))
    X = reshape(X, 32, 32, 3, 1)
    output = model(X)
    class_idx = Flux.onecold(output)[1]
    confidence = maximum(Flux.softmax(output))
    return class_idx, CLASS_NAMES[class_idx], confidence
end

function run_demo(model, n_samples=5)
    println("="^60)
    println("TRAFFIC SIGN RECOGNITION - LIVE DEMO")
    println("="^60)

    test_csv = CSV.read("data/Test.csv", DataFrame)
    plots_list = []

    for i in 1:n_samples
        row = test_csv[rand(1:nrow(test_csv)), :]
        img_path = joinpath("data", row[:Path])
        actual_class = row[:ClassId]

        img = load_image(img_path)
        arr = image_to_array(img)
        pred_idx, pred_name, confidence = predict_sign(model, arr)
        actual_name = CLASS_NAMES[actual_class + 1]

        println("\nSample $i:")
        println("  Actual:     Class $actual_class | $actual_name")
        println("  Predicted:  Class $(pred_idx-1) | $pred_name")
        println("  Confidence: $(round(confidence * 100, digits=2))%")
        if pred_idx - 1 == actual_class
            println("  Result:     CORRECT")
        else
            println("  Result:     WRONG")
        end

        display_img = load(img_path)
        display_img = imresize(display_img, (100, 100))
        correct = pred_idx - 1 == actual_class ? "Correct" : "Wrong"
        p = plot(display_img,
            title = "$correct: $pred_name\n($(round(confidence*100, digits=1))%)",
            titlefontsize = 7,
            axis = false,
            ticks = false,
            border = :none
        )
        push!(plots_list, p)
    end

    final_plot = plot(plots_list...,
        layout = (1, n_samples),
        size   = (250 * n_samples, 300)
    )
    display(final_plot)
    savefig(final_plot, "results/plots/demo_predictions.png")
    println("\nDemo plot saved to results/plots/demo_predictions.png")
end