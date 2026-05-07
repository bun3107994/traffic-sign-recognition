using Pkg
Pkg.activate(".")

println("Loading packages...")
include("src/data_loading.jl")
include("src/preprocessing.jl")
include("src/model.jl")
include("src/evaluate.jl")
include("demo.jl")

println("Loading saved model...")
using BSON: @load
@load "results/metrics/trained_model_v2.bson" model
println("="^50)
println("Model loaded! Ready to demo.")
println("="^50)

# Run demo automatically with 5 samples
run_demo(model, 5)