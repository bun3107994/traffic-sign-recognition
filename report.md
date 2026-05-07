# Traffic Sign Recognition using Convolutional Neural Networks

**Student:** Alhagie Kijera 
**Course:** Machine Learning
**Date:** April 2026

---

## 1. Introduction

Traffic sign recognition is a fundamental computer vision task with
direct applications in road safety and intelligent transportation
systems. The ability to automatically identify and classify road signs
from images is a critical component of autonomous vehicles and advanced
driver assistance systems (ADAS). A vehicle that misidentifies a Stop
sign or a Speed Limit sign could pose serious risks to passengers and
other road users.

This project develops a deep learning system capable of classifying 43
types of German traffic signs using the German Traffic Sign Recognition
Benchmark (GTSRB) dataset. A Convolutional Neural Network (CNN) is
designed, implemented, trained, and evaluated using the Julia
programming language and the Flux.jl deep learning framework.

The project went through two major iterations. The first model used
28x28 pixel images and achieved 65.1% test accuracy after 20 epochs.
The second improved model used 32x32 pixel images, a deeper
architecture, learning rate scheduling, and 30 epochs — achieving
76.4% test accuracy, an improvement of 11.3 percentage points.

The central research question guiding this project is: *Can a CNN
trained on low-resolution traffic sign images generalise reliably across
all 43 sign classes, and what factors influence its reliability?*

Beyond achieving high classification accuracy, this project places
particular emphasis on understanding the trustworthiness and limitations
of the model. The evaluation examines not only overall performance but
also per-class accuracy, failure patterns, and potential biases.

---

## 2. Problem Statement

The task addressed in this project is a supervised multi-class image
classification problem. Given a photograph of a traffic sign, the model
must correctly assign it to one of 43 predefined categories. These
categories span four main sign types: speed limit signs, warning signs,
prohibitory signs, and mandatory signs.

The input to the model is a 32x32 pixel RGB image, and the output is a
probability distribution over 43 classes. The predicted class is the
one with the highest probability.

This problem presents several practical challenges:

**Visual similarity:** Many traffic signs share the same outer shape.
All warning signs in the German system use a triangular shape, and all
prohibitory signs use a circular shape with a red border. Signs within
the same shape family differ only in their internal symbol, which can
be difficult to distinguish at low resolution.

**Resolution constraints:** Images are resized to 32x32 pixels to
reduce memory requirements. This causes loss of fine detail, making it
harder to distinguish signs with similar shapes but different internal
symbols. This is particularly problematic for speed limit signs, where
the only distinguishing feature is a number.

**Class imbalance:** The GTSRB dataset is not uniformly distributed.
Some sign classes have over 2,000 training examples while others have
fewer than 200. The bias analysis in this project confirms that this
imbalance affects model performance — classes with fewer samples
perform significantly worse.

**Real-world variability:** Traffic sign images are captured under
varying lighting conditions, angles, distances, and weather conditions.
This variability introduces noise that the model must learn to handle.

---

## 3. Dataset

The **German Traffic Sign Recognition Benchmark (GTSRB)** dataset is
used in this project. It was originally introduced at the International
Joint Conference on Neural Networks (IJCNN) 2011 and is one of the most
widely used benchmarks for traffic sign classification research.

### 3.1 Dataset Statistics

| Property | Value |
|---|---|
| Total images | ~51,839 |
| Training images | 39,209 |
| Test images | 12,630 |
| Number of classes | 43 |
| Image format | PNG (varying sizes) |
| Input size used | 32x32 pixels |

### 3.2 Class Distribution

The dataset exhibits significant class imbalance. Some sign categories
such as Speed Limit 50 km/h contain over 2,000 training samples, while
others such as Dangerous Curve Left contain fewer than 250. The bias
analysis confirms that this imbalance has a measurable impact on model
performance — classes with fewer than 100 test samples achieve an
average accuracy of 64.58% compared to 77.9% for larger classes.

### 3.3 Data Source

The dataset was obtained from Kaggle:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

---

## 4. Methodology

### 4.1 Data Loading

Images are loaded from the file system using their paths specified in
the Train.csv and Test.csv label files. Each image is loaded using
Julia's Images.jl library and converted to RGB format to ensure
consistent 3-channel input regardless of the original image format.
The RGB conversion is critical — without it, test images loaded in a
different format caused a large gap between training and test accuracy
in early experiments.

### 4.2 Preprocessing

All images undergo the following preprocessing steps:

**Resizing:** Each image is resized to 32x32 pixels using bilinear
interpolation. This was increased from 28x28 in the second model
iteration to preserve more detail in signs that differ only in internal
symbols.

**Channel extraction:** RGB pixel values are extracted into a
(3, 32, 32) Float32 array representing red, green, and blue channels.

**Normalisation:** Pixel values are normalised from [0, 1] to [-1, 1]
using the formula: `normalised = (x - 0.5) / 0.5`. This centres the
data around zero for more efficient training.

**Batching:** Images are grouped into batches of 16 for memory-
efficient training on CPU hardware.

### 4.3 Model Architecture

The final model is a 4-block Convolutional Neural Network implemented
using Flux.jl. The architecture was developed iteratively — the initial
3-block model was expanded to 4 blocks in the second iteration to
extract richer features from the larger 32x32 inputs.

#### Convolutional Blocks

Each convolutional block consists of a convolutional layer with 3x3
filters and same padding, batch normalisation, ReLU activation, and
max pooling (except Block 4 which has no pooling to preserve spatial
information):

| Block | Input | Filters | Output |
|---|---|---|---|
| Block 1 | 32x32x3 | 32 | 16x16x32 |
| Block 2 | 16x16x32 | 64 | 8x8x64 |
| Block 3 | 8x8x64 | 128 | 4x4x128 |
| Block 4 | 4x4x128 | 256 | 4x4x256 |

#### Classifier

After the convolutional blocks the feature maps are flattened into a
4,096-dimensional vector (4 x 4 x 256 = 4,096). This passes through:

- Dense(4096 → 512, relu) + Dropout(0.3)
- Dense(512 → 128, relu) + Dropout(0.2)
- Dense(128 → 43)

#### Design Decisions

**BatchNorm after every conv layer:** Stabilises training and allows
faster convergence. Without it early experiments showed unstable loss.

**Two dropout layers with different rates:** Dropout(0.3) after the
first dense layer provides strong regularisation. Dropout(0.2) after
the second provides lighter regularisation closer to the output.

**No softmax in model:** Logit cross-entropy loss handles softmax
internally in a numerically stable way.

**Learning rate scheduling:** The learning rate starts at 0.0001,
drops to 0.00005 at epoch 15, and to 0.00001 at epoch 25. This allows
the model to make large updates early and fine-tune carefully later.

### 4.4 Training

| Parameter | Version 1 | Version 2 |
|---|---|---|
| Image size | 28x28 | 32x32 |
| Architecture | 3 blocks | 4 blocks |
| Loss function | Logit cross-entropy | Logit cross-entropy |
| Optimizer | Adam 0.0001 | Adam with scheduling |
| Epochs | 20 | 30 |
| Batch size | 16 | 16 |

---

## 5. Results

### 5.1 Training Progress

| Epoch | Loss | Training Accuracy |
|---|---|---|
| 1 | 1.862 | 5.59% |
| 5 | 0.327 | 11.89% |
| 10 | 0.095 | 32.01% |
| 15 | 0.118 | 51.14% |
| 20 | 0.028 | 69.11% |
| 25 | 0.110 | 75.59% |
| 30 | 0.023 | 83.6% |

The small loss spikes at epochs 15 and 25 correspond to the learning
rate reductions. This is expected behaviour — when the learning rate
changes, the optimizer momentarily loses its momentum before recovering.

### 5.2 Overall Performance Comparison

| Metric | Version 1 | Version 2 | Improvement |
|---|---|---|---|
| Training Accuracy | 74.46% | 83.6% | +9.14% |
| Test Accuracy | 65.1% | 76.4% | +11.3% |
| Generalisation Gap | 9.36% | 7.2% | Better |

The improved model not only achieves higher accuracy but also
generalises better — the gap between training and test accuracy
reduced from 9.36% to 7.2%.

### 5.3 Per-Class Performance

**Best performing classes:**

| Class | Sign | Correct | Total | Accuracy |
|---|---|---|---|---|
| 32 | End restrictions | 60 | 60 | 100.0% |
| 16 | No vehicles >3.5t | 149 | 150 | 99.33% |
| 33 | Turn right ahead | 203 | 210 | 96.67% |
| 20 | Dangerous curve right | 86 | 90 | 95.56% |
| 13 | Yield | 671 | 720 | 93.19% |

**Worst performing classes:**

| Class | Sign | Correct | Total | Accuracy |
|---|---|---|---|---|
| 0 | Speed limit 20 | 4 | 60 | 6.67% |
| 39 | Keep left | 32 | 90 | 35.56% |
| 30 | Ice/snow | 66 | 150 | 44.0% |
| 3 | Speed limit 60 | 203 | 450 | 45.11% |
| 19 | Dangerous curve left | 28 | 60 | 46.67% |

---

## 6. Trustworthiness and Bias Analysis

### 6.1 Class Imbalance Bias — Confirmed

The bias analysis reveals a significant finding in the improved model:

| Group | Average Test Accuracy |
|---|---|
| Classes with < 100 test samples | 64.58% |
| Classes with >= 100 test samples | 77.9% |
| **Gap** | **13.32%** |

This 13.32% gap confirms that class imbalance is a real source of bias
in the model. Classes with fewer training and test samples consistently
perform worse. This was less visible in the first model (6% gap) but
becomes clearer in the improved model, suggesting that the larger
architecture is better at utilising data from well-represented classes
while still struggling with underrepresented ones.

For a real-world deployment, this bias is concerning. If a particular
sign type is rare in the training data — perhaps because it appears
less frequently on roads in the dataset's geographic region — the model
will be less reliable for exactly those signs.

### 6.2 Speed Limit Sign Confusion

A persistent failure pattern across both model versions is the
difficulty distinguishing between speed limit signs. Speed limit 20
achieves only 6.67% accuracy despite the improved model. Speed limit
60 achieves 45.11%. These signs all share identical circular shapes
with red borders — the only difference is the number displayed.

At 32x32 pixels, numbers are small and the model appears to struggle
to distinguish between similar-looking digits. For example:
- 20 vs 120 (similar digit shapes)
- 60 vs 80 (both round digits)

This suggests that even with 32x32 resolution, number recognition
remains a significant challenge for this architecture.

### 6.3 Improvement in Triangular Warning Signs

Comparing the two model versions reveals significant improvement in
the triangular warning sign category:

| Sign | Version 1 | Version 2 | Change |
|---|---|---|---|
| Double curve | 0.0% | 57.8% | +57.8% |
| Slippery road | 10.0% | 88.7% | +78.7% |
| General caution | 27.4% | 76.2% | +48.8% |
| Road narrows right | 21.11% | 76.7% | +55.6% |

The increased resolution from 28x28 to 32x32 pixels, combined with
the deeper architecture, enabled the model to learn the internal
symbols of triangular warning signs much more effectively. This
confirms that resolution was the primary limiting factor for this
class of signs.

### 6.4 Overconfident Predictions

A consistent observation during the demo is that the model makes
wrong predictions with high confidence. For example, Road Work was
predicted as Wild Animals with 98.98% confidence. This overconfidence
is a common problem with neural networks and is particularly dangerous
in safety-critical applications because it provides no warning signal
when the model is wrong.

A future improvement would be to use Monte Carlo Dropout or temperature
scaling to produce better-calibrated uncertainty estimates.

### 6.5 Suitability for Real-World Deployment

Based on the analysis, the current model at 76.4% test accuracy is
still not suitable for safety-critical deployment in autonomous
vehicles. The key concerns are:

- 23.6% of test images are still misclassified
- Speed limit 20 achieves only 6.67% — almost complete failure
- Overconfident wrong predictions with no uncertainty signal
- Confirmed class imbalance bias
- No testing under varying weather or lighting conditions
- Trained and tested on a single benchmark from one country

However, the model demonstrates that the approach is fundamentally
sound and that further improvements — more data, higher resolution,
data augmentation — could push accuracy into a deployable range.

---

## 7. Discussion

### 7.1 Key Lessons Learned

**Learning rate matters enormously.** The initial learning rate of
0.001 produced unstable training with oscillating accuracy. Reducing
to 0.0001 with scheduling produced stable, consistent improvement.
This was the most impactful single change made during the project.

**Resolution directly affects which classes can be learned.** The
jump from 28x28 to 32x32 pixels produced dramatic improvements on
triangular warning signs (+57% on some classes) while having less
impact on classes that were already well-learned. This confirms that
the triangular sign failures in Version 1 were primarily a resolution
problem, not a model capacity problem.

**Bigger models reveal bias more clearly.** The class imbalance gap
grew from 6% in Version 1 to 13% in Version 2. The more capable model
is better at utilising data from well-represented classes, making the
imbalance effect more visible.

**Accuracy alone is not enough.** The per-class analysis reveals
failures that would be invisible in the overall accuracy number. A
model with 76.4% overall accuracy that completely fails on Speed
Limit 20 (6.67%) is not as reliable as the headline number suggests.

### 7.2 Future Improvements

**Data augmentation:** Applying random rotations, brightness changes,
and zoom during training would expose the model to more variation and
likely improve generalisation, particularly for underrepresented classes.

**Higher resolution:** Increasing input to 64x64 pixels would
significantly help with speed limit number recognition. This was
limited by available CPU RAM in this project but would be feasible
with GPU hardware.

**Weighted loss function:** Using class-weighted loss would force the
model to pay more attention to underrepresented classes during
training, reducing the imbalance bias.

**Transfer learning:** Using a pre-trained backbone such as ResNet-18
would provide much richer initial features and likely push accuracy
above 90%.

---

## 8. Conclusion

This project successfully implemented and improved a CNN for German
traffic sign classification using Julia and Flux.jl. Through two
iterations of development — improving image resolution, deepening the
architecture, and adding learning rate scheduling — test accuracy was
pushed from 65.1% to 76.4%, exceeding the 75% target.

The trustworthiness analysis reveals two key findings. First, class
imbalance is a confirmed source of bias — underrepresented classes
perform 13.32% worse on average. Second, resolution is the primary
limiting factor for visually similar signs — increasing from 28x28
to 32x32 pixels dramatically improved performance on triangular
warning signs while speed limit signs remain difficult.

The project demonstrates that understanding a model's failures is
as important as improving its accuracy. A model that achieves 76.4%
overall but completely fails on Speed Limit 20 (6.67%) is not
uniformly reliable across all classes — a critical consideration for
any safety-critical application.

---

## 9. References

Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2011).
*The German Traffic Sign Recognition Benchmark: A multi-class
classification competition*. Proceedings of IJCNN, 1453-1460.

Innes, M. et al. (2018). *Fashionable Modelling with Flux*.
arXiv:1811.01457.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
*Gradient-based learning applied to document recognition*.
Proceedings of the IEEE, 86(11), 2278-2324.

GTSRB Dataset:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign