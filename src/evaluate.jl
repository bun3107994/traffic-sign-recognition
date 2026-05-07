using Flux
using Statistics
using CSV
using DataFrames

# Class names for all 43 traffic sign classes
const CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50",
    "Speed limit 60", "Speed limit 70", "Speed limit 80",
    "End speed limit 80", "Speed limit 100", "Speed limit 120",
    "No passing", "No passing >3.5t", "Right of way",
    "Priority road", "Yield", "Stop", "No vehicles",
    "No vehicles >3.5t", "No entry", "General caution",
    "Dangerous curve left", "Dangerous curve right", "Double curve",
    "Bumpy road", "Slippery road", "Road narrows right",
    "Road work", "Traffic signals", "Pedestrians",
    "Children crossing", "Bicycles crossing", "Ice/snow",
    "Wild animals", "End restrictions", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left",
    "Roundabout", "End no passing", "End no passing >3.5t"
]

# Overall accuracy
function evaluate_model(model, test_batches)
    Flux.testmode!(model)
    correct = 0
    total = 0

    println("Evaluating model on test data...")
    for (X, Y) in test_batches
        preds = Flux.onecold(model(X))
        actual = Flux.onecold(Y)
        correct += sum(preds .== actual)
        total += length(actual)
    end

    accuracy = correct / total * 100
    println("="^50)
    println("Test Accuracy: $(round(accuracy, digits=2))%")
    println("="^50)
    return accuracy
end

# Per-class accuracy
function evaluate_per_class(model, test_batches)
    Flux.testmode!(model)
    class_correct = zeros(Int, 43)
    class_total   = zeros(Int, 43)

    for (X, Y) in test_batches
        preds  = Flux.onecold(model(X))
        actual = Flux.onecold(Y)
        for (p, a) in zip(preds, actual)
            class_total[a] += 1
            if p == a
                class_correct[a] += 1
            end
        end
    end

    println("\n" * "="^60)
    println("PER-CLASS ACCURACY")
    println("="^60)
    println("$(rpad("Class", 6)) $(rpad("Sign Name", 28)) $(rpad("Correct", 10)) $(rpad("Total", 8)) Accuracy")
    println("-"^60)

    results = []
    for i in 1:43
        if class_total[i] > 0
            acc = class_correct[i] / class_total[i] * 100
            push!(results, (i-1, CLASS_NAMES[i], class_correct[i], class_total[i], acc))
            println("$(rpad(string(i-1), 6)) $(rpad(CLASS_NAMES[i], 28)) $(rpad(string(class_correct[i]), 10)) $(rpad(string(class_total[i]), 8)) $(round(acc, digits=1))%")
        end
    end

    # Save to CSV
    df = DataFrame(
        ClassId   = [r[1] for r in results],
        ClassName = [r[2] for r in results],
        Correct   = [r[3] for r in results],
        Total     = [r[4] for r in results],
        Accuracy  = [round(r[5], digits=2) for r in results]
    )
    CSV.write(joinpath(@__DIR__, "..", "results", "metrics", "per_class_accuracy.csv"), df)
    println("\nSaved to results/metrics/per_class_accuracy.csv")
    return df
end

# Bias analysis
function analyze_bias(per_class_df)
    println("\n" * "="^60)
    println("BIAS ANALYSIS")
    println("="^60)

    # Sort by accuracy
    sorted = sort(per_class_df, :Accuracy)

    println("\n5 WORST performing classes:")
    println("-"^60)
    for row in eachrow(first(sorted, 5))
        println("  Class $(row.ClassId) | $(row.ClassName) | $(row.Accuracy)% ($(row.Total) samples)")
    end

    println("\n5 BEST performing classes:")
    println("-"^60)
    for row in eachrow(last(sorted, 5))
        println("  Class $(row.ClassId) | $(row.ClassName) | $(row.Accuracy)% ($(row.Total) samples)")
    end

    # Check class imbalance effect
    println("\nCLASS IMBALANCE CHECK:")
    println("-"^60)
    avg_acc_small = mean(per_class_df[per_class_df.Total .< 100, :Accuracy])
    avg_acc_large = mean(per_class_df[per_class_df.Total .>= 100, :Accuracy])
    println("  Avg accuracy (< 100 samples):  $(round(avg_acc_small, digits=2))%")
    println("  Avg accuracy (>= 100 samples): $(round(avg_acc_large, digits=2))%")

    if avg_acc_small < avg_acc_large - 10
        println("  ⚠ BIAS DETECTED: Model performs worse on underrepresented classes")
    else
        println("  ✓ No strong imbalance bias detected")
    end

    # Save bias report
        open(joinpath(@__DIR__, "..", "results", "metrics", "bias_report.txt"), "w") do f
        write(f, "BIAS ANALYSIS REPORT\n")
        write(f, "="^60 * "\n\n")
        write(f, "5 Worst Classes:\n")
        for row in eachrow(first(sorted, 5))
            write(f, "  Class $(row.ClassId) | $(row.ClassName) | $(row.Accuracy)% ($(row.Total) samples)\n")
        end
        write(f, "\n5 Best Classes:\n")
        for row in eachrow(last(sorted, 5))
            write(f, "  Class $(row.ClassId) | $(row.ClassName) | $(row.Accuracy)% ($(row.Total) samples)\n")
        end
        write(f, "\nClass Imbalance Check:\n")
        write(f, "  Avg accuracy (< 100 samples):  $(round(avg_acc_small, digits=2))%\n")
        write(f, "  Avg accuracy (>= 100 samples): $(round(avg_acc_large, digits=2))%\n")
    end
    println("\nSaved to results/metrics/bias_report.txt")
end

using Plots

function plot_training(history)
    epochs = 1:length(history["train_loss"])

    # Loss plot
    p1 = plot(epochs, history["train_loss"],
        title  = "Training Loss",
        xlabel = "Epoch",
        ylabel = "Loss",
        label  = "Train Loss",
        color  = :blue,
        lw     = 2,
        marker = :circle
    )

    # Accuracy plot
    p2 = plot(epochs, history["train_accuracy"],
        title  = "Training Accuracy",
        xlabel = "Epoch",
        ylabel = "Accuracy (%)",
        label  = "Train Accuracy",
        color  = :green,
        lw     = 2,
        marker = :circle
    )

    # Combine both plots side by side
    combined = plot(p1, p2, layout=(1,2), size=(900, 400))

    # Save to results/plots/
    savefig(combined, "results/plots/training_history.png")
    println("Plot saved to results/plots/training_history.png")

    return combined
end