# Improving GarbageClassification_CNN Training

## Short diagnosis

The current notebook is useful for showing an edge-friendly CNN, but it is too constrained for strong training on the public garbage dataset.

The model has only about 2k trainable parameters, uses grayscale 96x96 inputs, and relies on depthwise-separable convolution blocks from the start. On a 6-class real-world dataset with visually similar categories such as glass, plastic, metal, and paper, this is very likely to underfit.

The training log supports that diagnosis:

- Chance accuracy for 6 balanced classes is about 16.7%.
- The largest class baseline is probably around 23-24%, and the model starts around 23.5%.
- Training accuracy and validation accuracy both stay low and improve slowly.
- There is no obvious sign of validation accuracy collapsing while training accuracy rises, so overfitting is not the first problem.

So the first lesson for students is: before changing optimizers or activations, compare train accuracy vs. validation accuracy.

If both are poor, the model is not learning enough. If training is high and validation is poor, the model is overfitting.

## What I would change first

### 1. Keep the final activation and loss

For this notebook, `Dense(num_classes, activation='softmax')` with `sparse_categorical_crossentropy` is correct.

Do not switch the final activation to sigmoid unless this becomes a multi-label problem where one image can belong to several classes at the same time.

For single-label multi-class classification:

```python
out = layers.Dense(num_classes, activation='softmax')(x)
loss = 'sparse_categorical_crossentropy'
```

This part is not the main cause of the weak result.

### 2. Use RGB instead of grayscale

Garbage categories often depend on color and material appearance:

- cardboard vs. paper
- clear glass vs. plastic
- metal cans vs. plastic containers

Converting to grayscale removes useful information. For a teaching experiment, this is one of the cleanest changes to demonstrate.

Change:

```python
tf.io.decode_image(img, channels=1, expand_animations=False)
input_shape=(96, 96, 1)
```

to:

```python
tf.io.decode_image(img, channels=3, expand_animations=False)
input_shape=(96, 96, 3)
```

Expected lesson: input representation can matter as much as model architecture.

### 3. Increase model capacity before changing optimizer

The current model has about 1,966 parameters. That is extremely small for a real-world 6-class image dataset.

A simple teaching progression:

```python
def build_small_cnn(input_shape=(96, 96, 3), num_classes=6):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inp, out)
```

This is less edge-minimal, but it is better for teaching the difference between a learning baseline and a deployable tiny model.

Expected lesson: if both training and validation accuracy are low, increase representational capacity.

### 4. Add BatchNormalization after convolutions

Batch normalization often makes CNN training more stable and less sensitive to initialization and learning rate.

Prefer this style:

```python
x = layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
```

instead of:

```python
x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
```

Expected lesson: activation choice is usually less important than stable optimization and good feature scaling.

### Batch Normalization
- Batch normalization is a technique used in neural networks to stabilize and speed up training.
- It normalizes the inputs of a layer across a mini-batch, so they have mean near 0 and variance near 1.
- For each feature/channel it computes:
    - mean of the batch
    - variance of the batch
    - normalized value: (x - mean) / sqrt(variance + epsilon)
- Then it applies two learned parameters:
    - gamma (scale)
    - beta (shift)

*This lets the layer learn the best scale and offset after normalization.*

### Formulas
```
y = W * x + b or y = W * x
y_norm = (y - μ) / sqrt(σ² + ε)
y_bn = γ * y_norm + β
z = max(0, y_bn)
```

### Why it helps
- Reduces internal covariate shift: the distribution of activations changes less during training.
- Makes gradient descent more stable.
- Often allows higher learning rates.
- Can act as a mild regularizer.

### 5. Add data augmentation, but not as the first fix

Augmentation helps generalization, but if the tiny model is already underfitting, augmentation may make learning even harder at first.

Use augmentation after the model can fit the training set better:

```python
augment = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])
```

Then place it near the input:

```python
x = augment(inp)
```

Expected lesson: augmentation fights overfitting. It does not usually fix underfitting by itself.

### 6. Use a learning-rate schedule

Adam at `1e-3` is a reasonable starting optimizer. I would not make optimizer choice the first change.

A better teaching step is to keep Adam and add callbacks:

```python
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=callbacks,
    verbose=2,
)
```

Expected lesson: learning rate is often more important than optimizer brand.

### 7. Check class imbalance and confusion matrix

The dataset is not perfectly balanced. The `trash` class is usually much smaller than classes like paper, glass, and plastic.

Have students compute:

```python
from collections import Counter

print(Counter([y for _, y in train_items]))
print(Counter([y for _, y in val_items]))
```

Then inspect a confusion matrix:

```python
y_true = []
y_pred = []

for xb, yb in val_ds:
    pred = model.predict(xb, verbose=0).argmax(axis=1)
    y_true.extend(yb.numpy())
    y_pred.extend(pred)

cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES)
print(cm.numpy())
```

Expected lesson: overall accuracy can hide class-specific failure.

## Suggested teaching sequence

Run these as separate experiments, changing one thing at a time:

1. Baseline: current tiny grayscale model.
2. RGB only: same model, but 3-channel input.
3. More capacity: RGB plus a larger standard CNN.
4. BatchNorm: larger CNN with Conv-BN-ReLU blocks.
5. LR callbacks: add `ReduceLROnPlateau` and `EarlyStopping`.
6. Augmentation: add moderate image augmentation.
7. Edge compression: once accuracy is acceptable, shrink the model again using depthwise separable convolutions, fewer channels, and int8 quantization.

This order teaches a practical workflow:

1. Prove the dataset is learnable.
2. Build a model that learns.
3. Improve generalization.
4. Compress for deployment.

## Recommended architecture for the teaching baseline

This version is still simple enough for students to understand, but much more realistic than the current 2k-parameter model.

```python
from tensorflow.keras import layers, models

def conv_bn_relu(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_teaching_cnn(input_shape=(96, 96, 3), num_classes=6):
    inp = layers.Input(shape=input_shape)

    x = layers.Rescaling(1.0)(inp)

    x = conv_bn_relu(x, 32)
    x = conv_bn_relu(x, 32)
    x = layers.MaxPooling2D()(x)

    x = conv_bn_relu(x, 64)
    x = conv_bn_relu(x, 64)
    x = layers.MaxPooling2D()(x)

    x = conv_bn_relu(x, 128)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inp, out, name='teaching_cnn')
```

Compile:

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
```

## Deployment note

The current notebook is titled as an OpenMV-friendly int8 model, so there is a real tradeoff:

- The current model is deployment-friendly but too weak for the dataset.
- A larger RGB CNN is better for teaching successful learning.
- After students understand the successful model, you can introduce edge constraints and show how accuracy drops as the model is compressed.

That makes the weak baseline useful: it becomes a teaching example of the accuracy/latency/model-size tradeoff.

## My prioritized recommendation

If I had to pick only three changes:

1. Switch from grayscale to RGB.
2. Increase CNN capacity substantially.
3. Add BatchNorm and learning-rate callbacks.

I would keep:

- ReLU or explicit `layers.ReLU()`
- Adam as the initial optimizer
- Softmax final activation
- Sparse categorical cross-entropy

Those choices are already appropriate for the problem.


# Assignment
## Optimize your Garbage Classification CNN with these instructions and report your results.