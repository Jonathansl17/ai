# Convolutional Neural Networks

Two image classification problems solved with the same kind of model, a small
convolutional neural network trained from scratch in Keras. In the first, what
separates the classes is an object on the face. In the second, it is the face
together with the way the photograph was taken, and most of the work goes into
finding out which of the two the network is really using.

| Notebook | Model | Data | Task |
|---|---|---|---|
| [`face-mask-cnn-classification.ipynb`](face-mask-cnn-classification.ipynb) | CNN of four convolutional blocks, global average pooling and one logit, 388 673 parameters | 7553 photographs of faces | Decide whether the face in a photograph is wearing a mask |
| [`professional-model-face-classification.ipynb`](professional-model-face-classification.ipynb) | Same architecture on 112 by 112 inputs, with a best-epoch checkpoint | 4600 close-up portraits of women | Decide whether a portrait shows a professional model or a common person |

## Problem

A support vector machine or a random forest reads a row of numbers, and a
photograph is not one. The [HOG and colour SVM](../hog-svm-image-classification)
in this repository solves that by designing 1821 numbers per image by hand. A CNN
removes that step. It slides small grids of weights, the filters, over the
picture, the first layers learn edges and patches of colour, the later ones
combine them into eyes, lips or the border of a mask, and nobody decides by hand
what to measure.

Both notebooks use the same network: four blocks of 3 by 3 convolutions with 32,
64, 128 and 256 filters, ReLU, max pooling and dropout, then global average
pooling, which reduces each of the 256 final feature maps to its mean, and a
single output neuron with no activation. That neuron returns a logit, its
weighted sum before any activation, and the loss is binary cross-entropy computed
from it. The optimizer is Adam. Training photographs get data augmentation, a
random crop and a random horizontal flip every time they enter a batch, and
validation and test photographs always get the central crop.

- **Mask detection.** The response is whether the face is covered, with covered
  as the positive class. Photographs are read at 72 by 72 and cropped to 64 by 64.
  A fixed stratified split leaves 5287 photographs for training, 1133 for
  validation and 1133 for test, and the network trains for 30 epochs.
- **Professional models against common people.** The response is whether the
  portrait is of a professional model, the positive class. Photographs are read at
  128 by 128 and cropped to 112 by 112, a size chosen because 64 by 64 lost too
  much of the face. The dataset comes already split and is used as delivered. The
  network trains for 30 epochs at a learning rate of 0.001, a `ModelCheckpoint`
  keeps the epoch with the best validation accuracy, and the trained network is
  inspected through its first-layer filters and the activation maps of its second
  convolution.

## Dataset

Neither image collection is in this repository, since they weigh 163 MB and 67 MB.
Each notebook downloads and unpacks its own collection on the first run and skips
the download when the folder is already there.

| Collection | Class | Photographs | Share |
|---|---|---|---|
| Masks | Without a mask | 3828 | 50.7 % |
| Masks | With a mask | 3725 | 49.3 % |
| Portraits | Common person | 2150 | 50.0 % |
| Portraits | Professional model | 2150 | 50.0 % |

The mask photographs range from 26 by 37 to 5760 by 5412 pixels and show faces
alone, in crowds and next to product shots. The portraits are all 224 by 224 RGB
close-ups, already cropped and centred, split into 2000 per class for training
and 150 per class for validation and for test.

## Results

| Metric | Masks | Portraits |
|---|---|---|
| Test photographs | 1133 | 300 |
| Test accuracy | 0.9532 | 0.9767 |
| Area under the ROC curve | 0.9902 | 0.9968 |
| Always answering the larger class | 0.5066 | 0.5000 |
| Mistakes | 53 | 7 |
| Last validation accuracy | 0.9744 | 0.9267 |

**Masks.** The network lets 23 of 574 uncovered faces through as covered, 4.0 %,
the mistake the check exists to prevent, and stops 30 of 559 covered faces. The
SVM on hand-made features in this repository reached 0.9052 on the same
photographs with a different split, so the comparison is indicative rather than
exact. The mistakes cluster in faces seen from the side, small faces in busy
scenes, masks close to the colour of skin and hands held near the mouth. A few
confident mistakes on plain frontal faces have no visible cause, and the
exploration shows why that is worrying: the two classes differ in file size,
sharpness and background, so part of what the network learned may be the source
of each class rather than the mask.

**Portraits.** The accuracy is high, and the notebook does not take it at face
value. The model portraits are much sharper than the snapshots of common people,
with a median sharpness of 212.7 against 80.3, and a logistic regression that sees
only brightness, colour saturation and sharpness, with no face involved, already
scores **0.8333** on the test photographs. The CNN goes clearly beyond that: on
the 50 test photographs where those three traits point to the wrong class it is
still right 0.9400 of the time. But 3 of its 7 mistakes fall in that group, which
holds one test photograph in six, most mistakes are sharp snapshots taken as
portraits or soft portraits taken as snapshots, and its activation maps light up
the border of the frame and part of the background as well as the face. The
network learned the face and the camera together, and this data cannot separate
them. The test score is also 5 points above the validation score, 15 photographs
out of 300, so with parts this small the accuracy is only known to within a few
points.

In both cases the curves of training and validation rise together with no sign
of overfitting, and in the portraits the best validation epoch was the last one,
so the network had not finished improving.

## Techniques used

Convolutional neural networks in Keras, with 3 by 3 convolutions, ReLU, max
pooling, dropout, global average pooling and a single logit output trained with
binary cross-entropy from logits and the Adam optimizer. Image loading and
resizing to a common frame, image tensors stored as 8 bit integers, average image
per class and per-pixel difference maps, picture-quality traits (brightness,
saturation and Laplacian sharpness) with a logistic regression baseline built on
them, data augmentation with random crops and horizontal flips through a
`PyDataset` batch generator, fixed train, validation and test splits, learning
curves read for bias and variance, best-epoch checkpointing with
`ModelCheckpoint`, confusion matrix, per-class precision, recall and F1, ROC curve
and AUC, accuracy on the subset where a baseline is wrong, inspection of the
misclassified images, visualisation of learned filters and of activation maps,
and model persistence in the `.keras` format.

## Running it

From the repository root:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open a notebook from this folder and run the cells in order. The first data
cell downloads and unpacks the image collection, and skips the download when the
folder is already there. On Google Colab Keras runs on TensorFlow, and on a
machine without TensorFlow the notebook switches Keras to PyTorch on its own, so
it runs from a clone and on Colab with nothing to configure. A GPU makes training
take a couple of minutes, and on a CPU it takes considerably longer.
