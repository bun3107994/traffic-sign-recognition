using Images
using ImageTransformations
using CSV
using DataFrames

const DATA_DIR  = joinpath(@__DIR__, "..", "data")
const TRAIN_DIR = joinpath(DATA_DIR, "Train")
const TEST_DIR  = joinpath(DATA_DIR, "Test")

function load_image(path::String)
    img = load(path)
    img = imresize(img, (32, 32))
    img = convert(Array{RGB{Float32}}, img)
    return img
end

function image_to_array(img)
    arr = zeros(Float32, 3, 32, 32)
    for i in 1:32, j in 1:32
        arr[1,i,j] = Float32(img[i,j].r)
        arr[2,i,j] = Float32(img[i,j].g)
        arr[3,i,j] = Float32(img[i,j].b)
    end
    return arr
end

function load_train_data()
    csv_path = joinpath(DATA_DIR, "Train.csv")
    df = CSV.read(csv_path, DataFrame)
    images = []
    labels = []
    println("Loading training data...")
    for row in eachrow(df)
        img_path = joinpath(DATA_DIR, row[:Path])
        img = load_image(img_path)
        arr = image_to_array(img)
        push!(images, arr)
        push!(labels, row[:ClassId])
    end
    println("Done! Loaded $(length(images)) training images.")
    return images, labels
end

function load_test_data()
    csv_path = joinpath(DATA_DIR, "Test.csv")
    df = CSV.read(csv_path, DataFrame)
    images = []
    labels = []
    println("Loading test data...")
    for row in eachrow(df)
        img_path = joinpath(DATA_DIR, row[:Path])
        img = load_image(img_path)
        arr = image_to_array(img)
        push!(images, arr)
        push!(labels, row[:ClassId])
    end
    println("Done! Loaded $(length(images)) test images.")
    return images, labels
end