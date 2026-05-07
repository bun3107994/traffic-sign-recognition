using Pkg
Pkg.activate(".")

mkpath("results/metrics")
mkpath("results/plots")

include("src/data_loading.jl")
include("src/preprocessing.jl")
include("src/model.jl")
include("src/train.jl")
include("src/evaluate.jl")

println("="^50)
println("TRAFFIC SIGN RECOGNITION - MAIN PIPELINE")
println("="^50)

# Step 1 - Load training data
println("\n[1/6] Loading training data...")
train_images, train_labels = load_train_data()

# Step 2 - Preprocess and batch training data
println("\n[2/6] Preprocessing training data...")
train_images_processed = preprocess_images(train_images)
train_batches = prepare_batches(train_images_processed, train_labels)

# Step 3 - Load and prepare test data
println("\n[3/6] Loading test data...")
test_images, test_labels = load_test_data()
test_images_processed = preprocess_images(test_images)
test_batches = prepare_batches(test_images_processed, test_labels)

# Step 4 - Build and train model
println("\n[4/6] Training model...")
model = build_model()
model, history = train_model!(model, train_batches, epochs=30)

# Step 5 - Evaluate on test data
println("\n[5/6] Evaluating on test data...")
evaluate_model(model, test_batches)
per_class_df = evaluate_per_class(model, test_batches)
analyze_bias(per_class_df)

# Step 6 - Plot training history
println("\n[6/6] Saving plots...")
plot_training(history)

println("\nAll done!")