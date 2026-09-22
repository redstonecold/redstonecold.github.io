---
layout: post
title: "Deep Learning for Computer Vision"
date: 2026-09-22 09:00:00 +0900
description: "Image 표현부터 CNN, Training, Transfer Learning, Object Detection, Segmentation까지 Computer Vision의 핵심 흐름을 정리한다."
tags: [ai, deep-learning, computer-vision, cnn, study]
categories: ["AI/DeepLearning"]
math: true
---

<style>
article h2 { margin-top: 3.5rem; }
article h3 { margin-top: 2.5rem; }
article img { max-width: 100%; height: auto; }
</style>

> Image를 숫자로 표현한다 → CNN이 Feature를 학습한다 → Loss로 학습 상태를 평가한다 → 목적에 맞게 Classification, Detection, Segmentation을 수행한다.

## 초록

Computer Vision은 Image에서 의미 있는 정보를 찾는 분야이다. Deep Learning Model이 Image를 처리하려면 먼저 Pixel을 Tensor로 표현해야 한다. CNN은 가까운 Pixel의 관계를 이용해 Feature를 학습한다. 학습한 Feature는 Image Classification, Object Detection, Image Segmentation에 활용할 수 있다.

이 글은 OpenCV를 이용한 전처리부터 Transfer Learning과 실제 Vision Task까지 하나의 흐름으로 연결한다. 코드에 등장하는 Function, Class, Argument의 역할도 함께 설명한다.

![Computer Vision 전체 흐름](/assets/img/computer-vision/01-overview.svg)

## 2.1 OpenCV & Image Representation

### OpenCV와 TensorFlow/Keras의 역할

| 도구 | 역할 |
| --- | --- |
| OpenCV | Image Loading, Resizing, 색상 변환, Edge 검출 등 Image 처리 |
| NumPy | Pixel을 다차원 Array로 저장하고 수치 계산 수행 |
| TensorFlow | Tensor 연산, 자동 미분, GPU 계산, Model Training 수행 |
| Keras | TensorFlow의 기능으로 Layer와 Model을 쉽게 구성하는 High-level API |

`cv2`는 Python용 OpenCV Module이다. Image 객체가 아니다. `cv2.imread()`, `cv2.resize()`처럼 Image 처리 Function을 제공한다. OpenCV로 읽은 Image는 일반적으로 NumPy의 `ndarray`이다.

```python
import cv2

image = cv2.imread("cat.jpg")
print(type(image))
# <class 'numpy.ndarray'>
```

`cv2.imread()`는 파일에서 Image를 읽는 Function이다. Color Image를 기본적으로 BGR 순서로 읽는다.

### Pixel, Shape, Channel

Pixel은 Digital Image를 구성하는 최소 단위이다. Image Shape은 일반적으로 다음 순서로 표현한다.

$$
(\text{Height},\text{Width},\text{Channels})
$$

```text
(480, 640, 3)
  ↑    ↑    └─ B, G, R 또는 R, G, B의 세 Channel
  │    └────── Width
  └─────────── Height
```

Channel은 Pixel 하나가 가진 정보의 종류이다. Grayscale은 밝기값 하나를 가지므로 Channel이 1개이다. RGB는 Red, Green, Blue 값이 있으므로 Channel이 3개이다. OpenCV는 Color Image를 BGR 순서로 읽는다.

8-bit는 가능한 상태가 다음과 같이 256개이다.

$$
2^8=256
$$

따라서 정수 범위는 0부터 255까지이다. 258개가 아니다.

### Loading, Resizing, Normalization

`cv2.cvtColor()`는 Color 표현을 변환하는 Function이다. `cv2.resize()`는 Image의 Width와 Height를 바꾸는 Function이다. `astype()`은 NumPy Array의 Data Type을 변환하는 Method이다.

```python
image_bgr = cv2.imread("cat.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
resized = cv2.resize(image_rgb, (224, 224))
normalized = resized.astype("float32") / 255.0
```

`cv2.resize()`의 크기 Argument는 `(Width, Height)` 순서이다. `astype("float32")`는 정수 Pixel을 소수 계산이 가능한 32-bit 실수로 바꾼다. 255로 나누면 Pixel 값은 0–1 범위가 된다.

$$
x_{\text{normalized}}=\frac{x}{255}
$$

Pixel 값 204는 0.8이 된다. 이는 색상 또는 밝기의 정규화된 값이다. 예측 확률이 아니다.

### Edge, Corner, Color Histogram

| Feature | 의미 |
| --- | --- |
| Edge | 한 방향으로 밝기나 색상이 급격히 변하는 경계선 |
| Corner | 여러 방향의 Edge가 만나거나 방향이 크게 바뀌는 지점 |
| Color Histogram | 밝기 또는 색상값별 Pixel 수의 분포 |

`cv2.Canny()`는 Canny Edge Detection을 수행하는 Function이다. 낮은 Threshold와 높은 Threshold를 이용해 이어진 Edge를 선택한다.

```python
gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 100, 200)
```

`cv2.GaussianBlur()`는 작은 Noise를 줄인다. `cv2.Canny()`에서 200보다 강한 변화는 Edge로 선택한다. 100–200 사이는 강한 Edge와 연결될 때 선택한다. Threshold는 Image에 맞춰 조정하는 Hyperparameter이다.

Color Histogram은 색의 양을 알려 주지만 위치는 알려 주지 않는다. 같은 색상 비율을 가진 서로 다른 Image는 비슷한 Histogram을 가질 수 있다.

### Tensor와 Batch Dimension

Tensor는 여러 차원의 숫자 배열을 포함하는 넓은 개념이다. Vector는 1차원 Tensor이다.

| Rank | 이름 | 예시 Shape |
| ---: | --- | --- |
| 0 | Scalar | `()` |
| 1 | Vector | `(784,)` |
| 2 | Matrix | `(28, 28)` |
| 3 | Color Image | `(224, 224, 3)` |
| 4 | Image Batch | `(32, 224, 224, 3)` |

Batch Dimension은 한 번에 처리하는 Image 수이다. `np.expand_dims()`는 NumPy Array에 크기 1인 Dimension을 추가하는 Function이다.

```python
import numpy as np

image_batch = np.expand_dims(normalized, axis=0)
```

```text
(224, 224, 3) → (1, 224, 224, 3)
```

## 2.2 From MLP to Deep Learning Vision

### Handcrafted Features와 Learned Features

Handcrafted Features(수작업 특징)는 사람이 Edge, Corner, Histogram처럼 사용할 특징을 직접 선택한다. Learned Features(학습된 특징)는 Model이 Training Data에서 필요한 표현을 학습한다. 이를 Representation Learning(표현 학습)이라고 한다.

```text
앞쪽 Layer: Edge와 색상 변화
중간 Layer: Texture와 모양의 일부
뒤쪽 Layer: 복잡한 Object Feature
```

### Supervised Learning과 Dataset 분할

Supervised Learning(지도 학습)은 Input과 정답 Label을 함께 사용한다.

| Dataset | 역할 |
| --- | --- |
| Training Set | Weight와 Bias 학습 |
| Validation Set | Hyperparameter와 학습 종료 시점 선택 |
| Test Set | 선택이 끝난 Model의 최종 평가 |

### Flatten과 MLP

Dense Layer는 Image 한 장을 일반적으로 1차원 Vector로 받는다. Flatten은 값을 삭제하지 않고 Shape만 바꾼다.

$$
(28,28,1)\rightarrow(784,)
$$

Batch가 32개이면 다음과 같다.

$$
(32,28,28,1)\rightarrow(32,784)
$$

`keras.Sequential`은 Layer를 순서대로 쌓아 Model을 만드는 Class이다. `Dense`의 첫 번째 Argument인 `units`는 Neuron 수이다.

```python
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation="relu"),
    keras.layers.Dense(units=10, activation="softmax")
])
```

Softmax는 Class별 Logit을 합이 1인 확률로 변환한다.

$$
p_j=\frac{e^{z_j}}{\displaystyle\sum_{k=1}^{K}e^{z_k}}
$$

One-hot Label을 사용하는 Categorical Cross-Entropy는 다음과 같다.

$$
L=-\sum_{j=1}^{K}y_j\log(p_j)
$$

### Training과 Inference

```text
Training:
Input → Prediction → Loss → Backpropagation → Parameter Update

Inference:
새 Input → Prediction
```

`np.argmax()`는 가장 큰 값이 있는 Index를 반환한다. `axis=1`은 Batch의 각 행에서 Class 방향으로 가장 큰 값의 위치를 찾는다.

```python
predicted_class = np.argmax(prediction, axis=1)
```

## 2.3 Convolutional Neural Networks (CNN)

MLP는 Flatten 과정에서 공간 구조를 직접 사용하기 어렵다. CNN은 작은 영역을 확인하는 Kernel을 Image 전체에서 반복해 사용한다.

![RGB Filter와 CNN의 Feature 추출 과정](/assets/img/computer-vision/02-cnn.svg)

### Convolution, Kernel, Filter, Feature Map

RGB Input의 Channel은 3개이다. Filter 하나는 Channel별 Kernel을 하나씩 가진다. `3 × 3` Filter 하나의 Shape은 `(3, 3, 3)`이다.

```text
Filter 1
├── Red용   3 × 3 Kernel
├── Green용 3 × 3 Kernel
└── Blue용  3 × 3 Kernel
```

한 위치에서 세 Kernel의 27개 곱셈 결과를 모두 더하고 Bias를 더하면 Output 값 하나가 된다. Filter 하나가 모든 위치를 이동하면 Feature Map 하나가 만들어진다. Filter가 32개이면 서로 다른 Feature Map 32개가 만들어진다.

$$
(H,W,3)\rightarrow(H_{out},W_{out},32)
$$

전체 Kernel Weight의 Shape은 다음과 같다.

$$
(3,3,3,32)
$$

### Local Receptive Field와 Weight Sharing

Local Receptive Field(국소 수용 영역)는 Output 한 값이 Kernel 크기만큼의 Input 영역만 확인한다는 뜻이다. 같은 Kernel Weight를 모든 위치에서 사용하는 것이 Weight Sharing이다. Parameter 수를 줄이고 같은 Feature를 여러 위치에서 찾게 한다.

### Stride와 Padding

Stride는 Kernel이 한 번에 이동하는 Pixel 수이다. Padding은 Image 가장자리에 값을 추가한다. Zero Padding은 Image 밖을 0으로 가정해 Kernel을 가장자리에도 놓게 한다.

한 방향의 Output 크기는 다음과 같다.

$$
O=\left\lfloor\frac{I-K+2P}{S}\right\rfloor+1
$$

| 기호 | 의미 |
| --- | --- |
| I | Input 크기 |
| K | Kernel 크기 |
| P | 한쪽 Padding 크기 |
| S | Stride |
| O | Output 크기 |

Padding을 포함한 길이는 `I + 2P`이다. Kernel 전체가 배열 안에 있어야 하므로 Kernel 길이 `K`를 뺀다. Stride 1에서는 시작 위치 0도 포함하므로 1을 더한다.

길이 7에 Kernel 3을 놓으면 시작점은 0, 1, 2, 3, 4이다.

```text
[0 1 2] 3 4 5 6
 0 [1 2 3] 4 5 6
 0 1 [2 3 4] 5 6
 0 1 2 [3 4 5] 6
 0 1 2 3 [4 5 6]
```

### ReLU와 Pooling

ReLU는 Non-linearity를 추가한다.

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

Max Pooling은 작은 영역의 최댓값을 남긴다. 값이 0으로 수렴하는 과정이 아니다. Height와 Width를 줄여 계산량을 낮추고 강한 Feature 반응을 요약한다.

```text
Convolution → Feature 탐지
ReLU       → Non-linearity 추가
Pooling    → 공간 크기와 계산량 감소
```

Pooling을 지나치게 반복하면 값보다 **세밀한 위치 정보**가 사라진다. Classification에서는 어느 정도의 위치 변화에 둔감해지는 장점이 있다. Detection과 Segmentation에서는 고해상도 Feature를 보존하거나 복원하는 구조가 필요하다.

### Convolutional Layers와 Classification Head

```text
Input Image
→ Conv + ReLU + Pooling
→ Conv + ReLU + Pooling
→ Global Average Pooling
→ Dense
→ Softmax
```

`GlobalAveragePooling2D`는 Feature Map마다 모든 위치의 평균을 하나 계산한다.

$$
(7,7,64)\rightarrow(64,)
$$

`7 × 7` 값 49개를 평균 내면 Channel마다 값 하나가 남는다. Channel이 64개이므로 길이 64인 Vector가 된다.

## 2.4 Training & Generalization

### SGD와 Adam

SGD는 Stochastic Gradient Descent(확률적 경사하강법)의 약자이다. 일부 Training Data로 Gradient를 계산하고 Parameter를 수정한다.

```text
새 Parameter
= 현재 Parameter
− Learning Rate × Gradient
```

Adam은 최근 Gradient의 방향과 크기에 관한 정보를 사용해 Parameter별 업데이트 크기를 조절한다.

| SGD | Adam |
| --- | --- |
| 단순하고 현재 Gradient 중심 | 과거 Gradient 정보도 사용 |
| Learning Rate 설정에 민감할 수 있음 | 초기 Training이 비교적 안정적인 경우가 많음 |
| 좋은 Generalization을 보이는 경우가 있음 | 빠르게 수렴하는 경우가 많음 |

### Learning Rate, Epoch, Batch Size

| 용어 | 의미 |
| --- | --- |
| Learning Rate | Parameter를 한 번에 움직이는 크기 |
| Batch Size | 업데이트 한 번에 사용하는 Example 수 |
| Iteration | Batch 하나로 Parameter를 한 번 업데이트 |
| Epoch | 전체 Training Dataset을 한 번 처리 |

Learning Rate가 너무 크면 Minimum을 넘어 Gradient 부호가 바뀌고 반대편으로 다시 이동할 수 있다. 이 과정이 반복되면 진동하며, 폭이 커지면 발산할 수도 있다.

### Loss, Accuracy, Overfitting

![Training과 Validation 곡선](/assets/img/computer-vision/03-training.svg)

| 상태 | Training | Validation |
| --- | --- | --- |
| 정상 학습 | Loss 감소, Accuracy 증가 | 함께 개선 |
| Underfitting | 성능이 충분히 개선되지 않음 | 함께 낮음 |
| Overfitting | 계속 개선 | Loss가 다시 증가하거나 Accuracy 정체 |

Generalization(일반화)은 Training에 사용하지 않은 Data에서도 좋은 Prediction을 만드는 능력이다. Loss는 예측 확률까지 반영하지만 Accuracy는 정답 여부만 센다. 따라서 두 지표가 항상 같은 방향으로 움직이지는 않는다.

### Overfitting을 줄이는 방법

| 방법 | 역할 |
| --- | --- |
| Data Augmentation | Label을 유지하는 회전, 이동, 밝기 변화 등으로 Training Image 다양화 |
| Dropout | Training 중 일부 Activation을 무작위로 0으로 만들어 특정 Neuron 의존 감소 |
| Early Stopping | Validation 성능이 개선되지 않으면 Training 중단 |

`EarlyStopping`은 Validation 지표를 관찰하는 Keras Callback Class이다.

```python
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
```

## 2.5 Transfer Learning

Transfer Learning(전이 학습)은 대규모 Dataset에서 학습한 Feature Extraction 능력을 새로운 문제에 재사용한다.

![Transfer Learning 단계](/assets/img/computer-vision/04-transfer.svg)

### Pretrained Model과 ImageNet

Pretrained Model은 이미 Training된 Model이다. ImageNet으로 학습한 CNN은 Edge, Texture, Shape 같은 일반적인 시각 Feature를 가진다.

`MobileNetV2()`는 MobileNetV2 Model을 만드는 Keras Function이다.

```python
base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
```

- `weights="imagenet"`: ImageNet Weight를 불러온다.
- `include_top=False`: 기존 ImageNet Classification Head를 제외한다.
- `input_shape`: 새로운 Image Shape을 지정한다.

### Feature Extraction과 Layer Freezing

```python
base_model.trainable = False
```

`trainable=False`는 Base Model의 Weight를 업데이트하지 않는다. 먼저 새 Classification Head만 학습해 무작위 초기 Weight가 Pretrained Feature를 크게 손상시키지 않게 한다.

### Classification Head 교체

```python
inputs = keras.Input(shape=(224, 224, 3))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

`keras.Model()`은 Input Tensor와 Output Tensor를 연결해 Model을 만드는 Class이다. `base_model(x, training=False)`는 Batch Normalization을 포함한 Base Model을 Inference 방식으로 실행한다.

### Fine-tuning

Fine-tuning은 새 Head를 먼저 학습한 뒤 Base Model의 일부 Layer를 다시 학습 가능하게 만드는 과정이다.

```python
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

마지막 20개 Layer만 학습한다. Pretrained Weight를 조금씩 조정하기 위해 작은 Learning Rate를 사용한다. `trainable`을 바꾼 뒤에는 `compile()`을 다시 호출한다.

### Model-specific Input Preprocessing

Model마다 Training 당시 사용한 Input 변환이 다르다. MobileNetV2의 `preprocess_input()`은 0–255 Pixel을 -1부터 1 범위로 변환한다. Image를 먼저 255로 나눈 뒤 같은 전처리를 다시 적용하면 Model이 기대한 값과 달라진다.

## 2.6 Image Classification & Object Detection

| Image Classification | Object Detection |
| --- | --- |
| Image 전체의 Class 예측 | 여러 Object의 Class와 위치 예측 |
| 위치를 출력하지 않음 | Bounding Box 출력 |
| 일반적으로 한 Prediction | 여러 Detection 가능 |

![Classification, Detection, IoU와 NMS](/assets/img/computer-vision/05-detection.svg)

### Class, Bounding Box, Confidence

Bounding Box는 `(x_min, y_min, x_max, y_max)` 또는 `(x_center, y_center, width, height)`로 표현할 수 있다. Confidence Score는 Model이 Detection을 얼마나 강하게 선택했는지 나타낸다. 실제 성공 확률과 정확히 일치한다는 보장은 없다.

### YOLO

YOLO는 You Only Look Once의 약자이며 대표적인 One-stage Object Detector이다. 하나의 Network가 Bounding Box, Object 존재 여부, Class를 함께 예측한다.

```text
Training:
Image + Ground-truth Box/Class
→ YOLO → Loss → Backpropagation → Parameter Update

Inference:
새 Image → 학습된 YOLO → Box/Class/Confidence → NMS
```

YOLO는 Training과 Inference에 모두 사용한다. 이미 학습된 Weight를 받아 Detection만 수행할 수도 있다.

### Intersection over Union

IoU는 두 Box가 얼마나 겹치는지 나타낸다.

$$
\operatorname{IoU}=\frac{\text{Intersection Area}}{\text{Union Area}}
$$

$$
\text{Union}=\text{Area A}+\text{Area B}-\text{Intersection}
$$

IoU가 0이면 전혀 겹치지 않는다. 1이면 두 Box가 완전히 같다.

### Non-Maximum Suppression

NMS는 같은 Object에 대한 중복 Box를 제거한다.

1. Confidence가 낮은 Box를 제거한다.
2. Confidence가 가장 높은 Box를 선택한다.
3. 다른 Box와 IoU를 계산한다.
4. IoU Threshold보다 크게 겹치는 낮은 Confidence Box를 제거한다.
5. 남은 Box에서 반복한다.

일반적인 NMS는 Class별로 수행한다. 사람과 자전거가 같은 위치에 있어도 Class가 다르면 둘 다 남길 수 있다.

## 2.7 Image Segmentation & Applications

Segmentation은 Pixel마다 Class를 예측한다.

| Task | 질문 | Output |
| --- | --- | --- |
| Classification | Image 전체가 무엇인가? | Class |
| Detection | 무엇이 어디에 있는가? | Class와 Box |
| Segmentation | 각 Pixel은 무엇인가? | Pixel-wise Mask |

### Semantic과 Instance Segmentation

Semantic Segmentation은 같은 Class의 모든 Pixel을 같은 값으로 표시한다. Instance Segmentation은 같은 Class의 Object도 개별 Instance로 구분한다.

```text
사람 두 명

Semantic: Person 영역 전체
Instance: Person 1 영역 + Person 2 영역
```

### U-Net

![U-Net과 Background Replacement](/assets/img/computer-vision/06-unet.svg)

U-Net은 Encoder, Bottleneck, Decoder, Skip Connection으로 구성된다.

| 부분 | 역할 |
| --- | --- |
| Encoder | Downsampling하며 Feature와 Context 추출 |
| Bottleneck | 가장 넓은 범위의 의미 정보 표현 |
| Decoder | Upsampling하며 Pixel 위치 복원 |
| Skip Connection | Encoder의 고해상도 위치 정보를 Decoder에 전달 |

Skip Connection은 같은 Height와 Width를 가진 Feature를 Channel 방향으로 이어 붙인다.

```text
Encoder Feature: (64, 64, 128)
Decoder Feature: (64, 64, 128)
Concatenation:   (64, 64, 256)
```

같은 위치에서 `[e1, e2, e3]`과 `[d1, d2]`를 더하지 않고 `[e1, e2, e3, d1, d2]`로 연결한다.

### Background Replacement

Background Replacement에는 원본 Image, Segmentation Mask, 새 Background Image가 필요하다. 세 자료의 Height와 Width를 맞춘다.

```text
결과 Pixel
= Mask × 원본 Pixel
+ (1 − Mask) × 새 Background Pixel
```

Mask가 1이면 원본 Foreground Pixel이 남는다. Mask가 0이면 준비한 새 Background Pixel이 들어간다. 0과 1 사이의 Soft Mask를 사용하면 머리카락 같은 경계를 부드럽게 합성할 수 있다.

## 마무리

Computer Vision의 출발점은 Image를 Tensor로 정확히 표현하는 것이다. CNN은 Local Receptive Field와 Weight Sharing을 이용해 공간적 Feature를 학습한다. Training에서는 Loss와 Validation 성능을 함께 확인해야 한다. Pretrained Model은 이 Feature 학습을 재사용한다.

마지막 Output의 형태에 따라 Task가 달라진다. Classification은 Image 전체의 Class를, Detection은 Object의 Class와 Box를, Segmentation은 Pixel별 Class를 예측한다. 이 차이를 이해하면 새로운 Vision Model을 볼 때 Input, Feature Extractor, Head, Loss, Output의 역할을 구분할 수 있다.
