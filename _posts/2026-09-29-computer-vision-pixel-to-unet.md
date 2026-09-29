---
layout: post
title: "Computer Vision 발표: Pixel에서 U-Net까지"
date: 2026-09-29 09:00:00 +0900
description: "Pixel과 Image Shape부터 CNN의 Feature 학습, Generalization, Segmentation Mask, U-Net까지 발표 흐름에 맞춰 정리한다."
tags: [ai, deep-learning, computer-vision, cnn, segmentation, study]
categories: ["AI/DeepLearning"]
math: true
---

<style>
article h2 { margin-top: 3.5rem; }
article h3 { margin-top: 2.3rem; }
article img { max-width: 100%; height: auto; }
</style>

> Image를 숫자 배열로 표현 → CNN으로 Feature 추출 → Validation으로 Generalization 확인 → U-Net으로 Pixel Mask 생성

## 발표 흐름

1. Pixel, Image Shape, Height × Width × Channels
2. Image Loading, Resizing, Normalization
3. Convolution, Kernel / Filter, Feature Map
4. Local Receptive Field, Weight Sharing
5. Overfitting and Generalization
6. Pixel-wise Prediction and Segmentation Mask
7. U-Net · Encoder, Decoder, Skip Connection

![발표 전체 흐름](/assets/img/computer-vision/07-presentation-flow.svg)

**그림 읽기:** Image의 숫자 표현에서 시작해 전처리, CNN, Generalization, Segmentation으로 이어지는 발표 순서.

## 1. Pixel, Image Shape, Height × Width × Channels

**Tensor(텐서):** 숫자를 여러 차원으로 배열한 자료 구조. 여기서는 Image Pixel을 `(Height, Width, Channels)` 형태로 저장한 숫자 배열이라는 의미만 사용.

**Axis(축):** Shape에 적힌 각 위치의 번호. 이 글에서 사용하는 HWC 순서에서는 Axis 0이 Height, Axis 1이 Width, Axis 2가 Channels.

| 차원 | 이름 | 형태 | Image에서의 예 |
| --- | --- | --- | --- |
| 0차원 | Scalar(스칼라) | 숫자 하나 | Pixel의 Channel 값 하나 |
| 1차원 | Vector(벡터) | 숫자가 한 방향으로 배열 | RGB Pixel `[R, G, B]` |
| 2차원 | Matrix(행렬) | 행과 열로 배열 | Grayscale Image `(H, W)` |
| 3차원 | 3D Tensor(3차원 텐서) | 행 × 열 × Channel | Color Image `(H, W, C)` |
| 4차원 | 4D Tensor(4차원 텐서) | 여러 3D Tensor의 묶음 | Image Batch `(N, H, W, C)` |

> 이 표에서는 숫자가 배열된 방향의 수를 Tensor의 차원이라고 부름. 발표의 핵심은 3차원 Color Image Shape인 `(H, W, C)`.

**Pixel:** Digital Image를 구성하는 최소 단위

**Image:** Pixel을 Height와 Width 방향으로 배열한 숫자 Tensor

Model이 보는 것은 고양이, 사람, 자동차라는 의미가 아니라 각 Pixel에 저장된 숫자.

### Pixel 값

**Grayscale Pixel:** 밝기값 하나

```text
0   : 검은색
128 : 중간 회색
255 : 흰색
```

**RGB Pixel:** 색상 강도 세 개

```text
Pixel = [Red, Green, Blue]
```

### 8-bit Image

**8-bit Image:** Pixel의 값 하나를 8개의 bit로 표현하는 Image

Grayscale Image에서는 Pixel 하나의 밝기를 8-bit로 표현.

RGB Image에서는 Pixel 하나가 가진 Red, Green, Blue의 색상 강도를 각각 8-bit로 표현.

$$
2^8=256
$$

```text
2의 8제곱 = 256
```

가능한 값의 개수는 256개. 0부터 시작하므로 실제 범위는 0–255.

```text
00000000 = 0
11111111 = 255
```

```text
Grayscale Pixel = 밝기값 1개 = 8-bit
RGB Pixel       = [R, G, B]  = 8-bit + 8-bit + 8-bit = 24-bit
```

RGB Pixel의 예:

```text
[255,   0,   0] → Red
[  0, 255,   0] → Green
[  0,   0, 255] → Blue
[255, 255, 255] → White
[  0,   0,   0] → Black
```

### Image Shape

**Image Shape:** Tensor의 각 Dimension 크기

$$
(\text{Height},\text{Width},\text{Channels})
$$

```text
Image Shape = (Height, Width, Channels)
```

```text
(480, 640, 3)
  ↑    ↑    └─ Pixel마다 저장된 값 3개
  │    └────── 가로 640 Pixel
  └─────────── 세로 480 Pixel
```

| Dimension | 핵심 의미 |
| --- | --- |
| Height | 세로 방향 Pixel 수 |
| Width | 가로 방향 Pixel 수 |
| Channels | 같은 공간 위치에 저장하는 정보 성분의 수 |

**Channel:** Image를 구성하는 정보 성분 하나

각 Channel은 `Height × Width` 크기의 2차원 값 배열. 여러 Channel을 같은 위치에 겹쳐 하나의 Image를 표현.

```text
Grayscale: 밝기 Channel 1개
RGB:       Red, Green, Blue Channel 3개
BGR:       Blue, Green, Red Channel 3개
```

RGB Image의 한 좌표에는 각 Channel에서 값 하나씩 대응하므로 `[R, G, B]` 세 값이 존재.

> CNN의 중간 Feature Map에서 Channel은 색상이 아니라 서로 다른 Feature를 나타냄.

### RGB와 BGR

**RGB:** Red, Green, Blue 순서

**BGR:** Blue, Green, Red 순서

OpenCV의 기본 Color Image 순서는 BGR. RGB 기반 화면이나 Library에 전달할 때 Channel 순서 변환 필요.

![Pixel과 Image Shape](/assets/img/computer-vision/08-image-shape.svg)

**그림 읽기:** RGB Pixel의 세 값이 Channel Dimension을 형성. Image Shape의 마지막 3이 Red, Green, Blue에 대응.

## 2. Image Loading, Resizing, Normalization

**Image 전처리:** 원본 Image를 Model이 받을 수 있는 Shape과 값의 범위로 변환하는 과정

```text
Image Loading
→ Color Conversion
→ Resizing
→ Data Type 변환
→ Normalization
```

![Image 전처리 단계와 Shape 변화](/assets/img/computer-vision/12-preprocessing-pipeline.svg)

**그림 읽기:** Image를 읽은 뒤 Color 순서와 크기를 맞추고, Pixel 값을 Model이 계산하기 좋은 범위로 변환.

### 도구별 역할

| 도구 | 핵심 역할 |
| --- | --- |
| OpenCV | Image Loading, Resizing, 색상 변환 |
| NumPy | Pixel을 Array로 저장하고 수치 계산 |
| TensorFlow | Tensor 연산, 자동 미분, Training |
| Keras | Layer와 Model 구성을 위한 High-level API |

**`cv2`:** Python에서 OpenCV 기능을 제공하는 Module

`cv2` 자체가 Image 객체인 것은 아님. `cv2.imread()`, `cv2.resize()` 같은 Function의 모음.

**NumPy:** 다차원 Array와 수치 계산을 제공하는 Library

OpenCV로 읽은 Image의 일반적인 Type은 NumPy의 `ndarray`.

### Image Loading

**`cv2.imread()`:** 파일 경로에서 Image를 읽는 Function

```python
import cv2

image_bgr = cv2.imread("cat.jpg")
```

```python
print(type(image_bgr))
print(image_bgr.shape)
print(image_bgr.dtype)
```

```text
<class 'numpy.ndarray'>
(480, 640, 3)
uint8
```

**`uint8`:** 0–255를 저장하는 부호 없는 8-bit 정수 Type

### Color Conversion

**`cv2.cvtColor()`:** Image의 색상 표현을 변환하는 Function

```python
image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)
```

Channel 수는 3개로 유지. BGR에서 RGB로 순서만 변경.

### Resizing

**`cv2.resize()`:** Image의 Width와 Height를 바꾸는 Function

```python
resized = cv2.resize(image_rgb, (224, 224))
```

주의할 순서:

```text
cv2.resize Argument: (Width, Height)
Image Shape:          (Height, Width, Channels)
```

Model은 정해진 Input Shape을 사용하므로 크기가 다른 Image를 같은 크기로 통일.

### Data Type 변환

**`.astype()`:** NumPy Array의 Data Type을 변환하는 Method

```python
float_image = resized.astype("float32")
```

`uint8`에서 `float32`로 변환 → 소수 계산 가능. TensorFlow Image 계산에 흔히 사용하는 Type.

### Normalization

**Normalization:** Pixel 값을 일정한 범위로 변환하는 과정

0–255 값을 0–1로 바꾸는 기본식:

$$
x_{normalized}=\frac{x}{255}
$$

```text
정규화된 Pixel 값 = 원래 Pixel 값 ÷ 255
```

```python
normalized = resized.astype("float32") / 255.0
```

Pixel 204의 변환:

$$
\frac{204}{255}=0.8
$$

```text
204 ÷ 255 = 0.8
```

여기서 0.8의 의미: 해당 Channel의 정규화된 색상 강도 또는 밝기. Class 확률은 아님.

### 전체 코드

```python
import cv2

image_bgr = cv2.imread("cat.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
resized = cv2.resize(image_rgb, (224, 224))
normalized = resized.astype("float32") / 255.0
```

**Model-specific Preprocessing:** Pretrained Model이 학습할 때 사용한 Input 변환 규칙

Model마다 Training 때 사용한 Pixel 범위가 다름. 어떤 Model은 `0–1`을 사용하고, 어떤 Model은 값의 중심이 0이 되도록 `-1–1`을 사용.

Inference에서도 Input Image의 Pixel 값을 Training 때 사용한 Pixel 값의 범위와 같게 변환해야 함.

```text
직접 만든 Model이 0–1로 학습됨
→ Pixel ÷ 255 사용

Pretrained Model이 -1–1로 학습됨
→ 해당 Model의 preprocess_input() 사용
```

`Pixel ÷ 255`를 적용한 뒤 `preprocess_input()`을 또 적용하면 값을 두 번 변환하게 됨. 따라서 두 방법을 연속으로 사용하지 않고, Model이 요구하는 전처리 방법 하나만 사용.

## 3. Convolution, Kernel / Filter, Feature Map

**Convolution:** 작은 Weight 배열을 Image 위에서 이동시키며 지역 Pattern을 계산하는 연산

**Kernel:** 한 Channel의 작은 영역에 적용되는 Weight 행렬

**Filter:** 모든 Input Channel에 대응하는 Kernel 묶음

**Feature Map:** Filter가 각 위치에서 계산한 결과를 모은 Output

![Convolution, Kernel, Filter, Feature Map의 관계](/assets/img/computer-vision/17-convolution-terms.svg)

**그림 읽기:** RGB Image의 같은 위치에서 Red, Green, Blue 영역을 선택. 각 영역에 해당 Channel의 Kernel을 적용하고 결과를 모두 더해 Output 값 하나 생성. Filter가 Image 전체를 이동하면 이 값들이 모여 Feature Map 하나 완성.

```text
Channel별 작은 Weight 행렬 = Kernel
RGB Channel용 Kernel 3개의 묶음 = Filter 1개
Filter 1개가 Image 전체를 이동한 결과 = Feature Map 1개
```

### 한 Channel의 Convolution

$$
z=\sum_i\sum_j X_{i,j}K_{i,j}+b
$$

```text
z = 현재 Image 영역과 Kernel의 같은 위치끼리 곱한 값의 합 + Bias
```

| 기호 | 의미 |
| --- | --- |
| X | 현재 Kernel이 보는 Input 영역 |
| K | Kernel Weight |
| b | Bias |
| z | 현재 위치의 Output 값 |

계산 예시:

$$
X=
\begin{bmatrix}
1&2\\
4&5
\end{bmatrix},
\qquad
K=
\begin{bmatrix}
1&0\\
0&-1
\end{bmatrix}
$$

```text
Input 영역 X = [[1, 2],     Kernel K = [[ 1,  0],
                [4, 5]]                 [ 0, -1]]
```

$$
(1\times1)+(2\times0)+(4\times0)+(5\times-1)=-4
$$

```text
(1 × 1) + (2 × 0) + (4 × 0) + (5 × -1) = -4
```

계산 결과 -4: Feature Map의 현재 위치에 저장되는 값.

![Convolution 한 위치의 계산](/assets/img/computer-vision/13-convolution-calculation.svg)

**그림 읽기:** Input 영역과 Kernel을 같은 위치끼리 곱한 뒤 모두 더해 Feature Map의 값 하나 생성.

### RGB Image에서 Filter 하나

**RGB Input:** Channel 3개

**`3 × 3` Filter 하나:** Channel별 `3 × 3` Kernel 3개

```text
Filter 1
├── Red용   3 × 3 Kernel
├── Green용 3 × 3 Kernel
└── Blue용  3 × 3 Kernel
```

Filter 하나의 Shape:

$$
(3,3,3)
$$

```text
Filter Shape = (Kernel Height 3, Kernel Width 3, Input Channels 3)
```

한 위치의 곱셈 수:

$$
3\times3\times3=27
$$

```text
3 × 3 × 3 = 27번의 곱셈
```

```text
Red 영역   × Red Kernel
+ Green 영역 × Green Kernel
+ Blue 영역  × Blue Kernel
+ Bias
= Output 값 하나
```

Filter 하나가 모든 위치를 이동 → Feature Map 하나.

### `filters=32`

**`filters=32`:** 서로 다른 Filter 32개를 학습하겠다는 설정

Filter 하나가 Channel 하나만 담당하는 구조가 아님. 32개 Filter 각각이 RGB 세 Channel을 모두 확인.

```text
같은 RGB 영역
├── Filter 1  → Output 값 1
├── Filter 2  → Output 값 2
├── Filter 3  → Output 값 3
│
└── Filter 32 → Output 값 32
```

Output Shape:

$$
(H,W,3)\rightarrow(H_{out},W_{out},32)
$$

```text
Input (H, W, 3) → Output (H_out, W_out, 32)
```

전체 Kernel Weight Shape:

$$
(3,3,3,32)
$$

```text
전체 Kernel Weight Shape = (3, 3, 3, 32)
```

마지막 32: Filter 수 = Feature Map 수 = Output Channel 수.

![RGB Channel과 32개 Filter](/assets/img/computer-vision/02-cnn.svg)

**그림 읽기:** Filter 하나가 RGB 세 Channel을 동시에 계산해 Feature Map 하나 생성. Filter 32개는 Feature Map 32개 생성.

### Keras 코드

**`keras.layers.Conv2D`:** 2차원 Image에 Convolution을 수행하는 Layer Class

```python
conv = keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu"
)
```

| Argument | 핵심 의미 |
| --- | --- |
| `filters=32` | Filter와 Output Channel 32개 |
| `kernel_size=(3, 3)` | Channel별 Kernel 크기 |
| `activation="relu"` | Convolution 결과에 ReLU 적용 |

어떤 Filter가 Edge, Texture, Shape에 반응할지는 Training으로 결정.

## 4. Local Receptive Field, Weight Sharing

**Local Receptive Field:** Output 한 값이 직접 확인하는 Input의 지역 영역

**Weight Sharing:** 같은 Kernel Weight를 모든 공간 위치에서 반복 사용

### Local Receptive Field

`3 × 3` Kernel의 한 번 계산 범위: 공간 위치 9개. RGB라면 각 위치의 세 Channel을 동시에 확인.

```text
전체 Image
┌─────────────────────┐
│  ┌───────┐          │
│  │ 3 × 3 │ → 이동   │
│  └───────┘          │
│                     │
└─────────────────────┘
```

핵심: Kernel이 Image 전체를 한 번에 보는 것이 아니라, Kernel 크기만큼 확인하며 이동.

![Local Receptive Field와 Weight Sharing](/assets/img/computer-vision/09-local-weight-sharing.svg)

**그림 읽기:** 작은 Kernel이 위치를 옮겨도 같은 Weight 사용. 노란 영역이 Output 한 값의 Local Receptive Field.

### Network Depth와 Receptive Field

첫 Layer: 원본 Image의 작은 영역 확인.

다음 Layer: 앞 Layer의 여러 Feature Map 위치 확인.

결과: Layer가 깊어질수록 원본 Image의 더 넓은 영역에 간접적으로 영향받음.

![Layer가 깊어질수록 넓어지는 Receptive Field](/assets/img/computer-vision/14-receptive-field-depth.svg)

**그림 읽기:** 연속된 `3 × 3` Convolution을 지나면 뒤쪽 값 하나가 참고하는 원본 영역이 단계적으로 확대.

```text
앞쪽 Layer: Edge, 밝기 변화
중간 Layer: Texture, 곡선, 모서리
뒤쪽 Layer: Object 일부, 복잡한 Shape
```

### Weight Sharing

동일한 Filter를 여러 위치에서 사용.

```text
같은 Filter
→ Image 왼쪽 위
→ Image 중앙
→ Image 오른쪽 아래
```

고양이 귀가 왼쪽에 있든 오른쪽에 있든 같은 Filter로 탐지 가능.

### Parameter 수 비교

`224 × 224 × 3` Image를 Dense Neuron 하나에 연결할 때의 Weight 수:

$$
224\times224\times3=150{,}528
$$

```text
224 × 224 × 3 = 150,528개 Weight
```

RGB Image의 `3 × 3` Filter 하나가 가진 Weight 수:

$$
3\times3\times3=27
$$

```text
3 × 3 × 3 = 27개 Weight
```

Bias 하나를 포함한 Parameter: 28개.

같은 28개 Parameter를 여러 위치에서 공유.

**Weight Sharing의 효과:** Parameter 감소, 위치가 달라도 같은 Feature 탐지, 공간 구조 유지

## 5. Overfitting and Generalization

**Overfitting(과적합):** Training Data의 일반 Pattern뿐 아니라 Noise와 우연한 차이까지 지나치게 학습한 상태

**Generalization(일반화):** Training에 사용하지 않은 Data에서도 유용한 Prediction을 만드는 능력

```text
좋은 Training 성능
+ 좋은 Validation/Test 성능
= 좋은 Generalization
```

### Dataset의 역할

| Dataset | 핵심 역할 |
| --- | --- |
| Training | Weight와 Bias 학습 |
| Validation | Hyperparameter, Model, 종료 시점 선택 |
| Test | 모든 선택이 끝난 Model의 최종 평가 |

Validation Data: Parameter 업데이트에는 사용하지 않지만 Model 선택에는 사용.

Test Data: 최종 평가용. 반복해서 확인하며 설정을 바꾸면 공정한 평가가 어려움.

### Loss Curve 해석

![Training과 Validation Loss](/assets/img/computer-vision/03-training.svg)

**그림 읽기:** Training Loss는 계속 감소하지만 Validation Loss가 다시 증가하는 구간에서 Overfitting 가능성 확인.

| 관찰 | 가능한 해석 |
| --- | --- |
| 두 Loss가 함께 감소 | Training과 Validation에서 함께 개선 |
| 두 Loss가 모두 높음 | Underfitting 가능성 |
| Training Loss만 계속 감소 | Training Data에 더 밀착 |
| Validation Loss가 다시 증가 | Overfitting 가능성 |

판단 기준: 한 Epoch의 작은 변화가 아니라 여러 Epoch의 추세.

### Loss와 Accuracy의 차이

**Accuracy:** 정답 여부만 계산

**Cross-Entropy Loss:** 정답 Class에 부여한 확률까지 반영

```text
정답 Class: 1

Prediction A: [0.10, 0.90] → 정답
Prediction B: [0.40, 0.60] → 정답
```

두 Prediction의 Accuracy는 동일. A가 정답에 더 높은 확률을 주므로 Loss는 더 작음.

### Overfitting을 줄이는 방법

| 방법 | 핵심 설명 |
| --- | --- |
| 더 다양한 Data | 우연한 특징을 외우기 어렵게 구성 |
| Data Augmentation | Label을 유지하는 회전, 이동, 밝기 변화 추가 |
| Model Capacity 감소 | 불필요하게 큰 Layer와 Neuron 축소 |
| Weight Decay | 지나치게 큰 Weight에 불이익 적용 |
| Dropout | Training 중 일부 Activation을 무작위로 0 처리 |
| Early Stopping | Validation 성능이 나빠지기 전에 중단 |

**`EarlyStopping`:** Validation 지표를 관찰하는 Keras Callback

```python
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
```

```text
monitor="val_loss"       : Validation Loss 관찰
patience=3               : 3 Epoch 동안 개선이 없으면 중단
restore_best_weights=True: 가장 좋은 시점의 Weight 복원
```

## 6. Pixel-wise Prediction and Segmentation Mask

**Pixel-wise Prediction:** Image의 각 Pixel마다 Class 또는 Foreground 확률을 예측하는 방식

**Segmentation Mask:** 각 Pixel의 최종 Class 정보를 저장한 Image

### Vision Task 비교

| Task | 핵심 질문 | Output |
| --- | --- | --- |
| Classification | Image 전체가 무엇인가? | Class Label |
| Detection | 무엇이 어디에 있는가? | Class와 Bounding Box |
| Segmentation | 각 Pixel은 무엇인가? | Pixel-wise Mask |

### Binary Segmentation Mask

```text
0 = Background
1 = Person
```

```text
0 0 0 0 0
0 0 1 1 0
0 1 1 1 0
0 0 1 0 0
0 0 0 0 0
```

![Pixel-wise Prediction과 Segmentation Mask](/assets/img/computer-vision/10-segmentation-mask.svg)

**그림 읽기:** Model이 Pixel마다 Score를 계산하고, Class 결정 과정을 거쳐 원본과 같은 공간 크기의 Mask 생성.

Shape 비교:

```text
Input Color Image: (Height, Width, 3)
Class Mask:        (Height, Width)
```

Input의 3: RGB Channel 수.

Mask의 값: 색상이 아니라 Class 번호.

### Binary Pixel Prediction

Pixel마다 Logit 하나 계산 → Sigmoid로 Foreground 확률 생성.

$$
p=\sigma(z)
$$

```text
Foreground 확률 p = Sigmoid(z)
```

```text
p가 0.5 이상 → 1, Foreground
p가 0.5 미만 → 0, Background
```

0–1이라는 범위만으로 자동으로 확률이 되는 것은 아님. Output을 Foreground 사건으로 정의하고 그 목적에 맞는 Loss로 Training할 때 Foreground 확률로 해석.

### Multi-class Pixel Prediction

Pixel마다 Class별 Logit 계산 → Softmax로 Class별 확률 생성.

```text
한 Pixel의 Prediction

Background: 0.02
Road:       0.05
Car:        0.88
Person:     0.03
Building:   0.02
```

가장 높은 확률의 Car → 해당 Pixel의 최종 Class.

### Semantic과 Instance Segmentation

**Semantic Segmentation:** 같은 Class의 모든 Pixel을 같은 Class 영역으로 표시

**Instance Segmentation:** 같은 Class의 Object도 개별 Instance로 구분

![Semantic Segmentation과 Instance Segmentation 비교](/assets/img/computer-vision/15-semantic-instance.svg)

**그림 읽기:** Semantic Mask는 같은 Class의 두 Object에 같은 값을 사용. Instance Mask는 Object마다 별도 ID 부여.

```text
사람 두 명

Semantic: Person 영역 전체
Instance: Person 1 영역 + Person 2 영역
```

## 7. U-Net · Encoder, Decoder, Skip Connection

**U-Net:** Encoder에서 압축한 Feature를 Decoder에서 Pixel 해상도로 복원하는 Segmentation 구조

**구성:** Encoder → Bottleneck → Decoder + Skip Connection

![U-Net의 Encoder, Decoder, Skip Connection](/assets/img/computer-vision/16-unet-architecture.svg)

**그림 읽기:** Encoder에서 공간 크기를 줄이고 Channel을 늘린 뒤 Decoder에서 공간 크기 복원. 같은 해상도의 Encoder Feature는 Skip Connection으로 전달.

### Encoder

**Encoder:** Downsampling하면서 Feature와 넓은 Context를 학습하는 부분

```text
256 × 256 × 3
→ 128 × 128 × 64
→ 64 × 64 × 128
→ 32 × 32 × 256
→ 16 × 16 × 512
```

Height와 Width: 감소.

Channel 수: 증가 가능.

앞쪽 Feature: Edge와 Texture.

뒤쪽 Feature: Object 일부와 복잡한 의미.

장점: 계산량 감소, 더 넓은 Pattern 학습.

손실: 정확한 위치와 경계 정보 감소.

### Bottleneck

**Bottleneck:** Encoder와 Decoder 사이의 가장 압축된 Feature

```text
작은 공간 해상도
+ 많은 Feature Channels
+ 넓은 Context
```

### Decoder

**Decoder:** Feature Map의 Height와 Width를 키워 Pixel 위치에 맞는 Mask를 복원하는 부분

```text
16 × 16 × 512
→ 32 × 32 × 256
→ 64 × 64 × 128
→ 128 × 128 × 64
→ 256 × 256 × Classes
```

단순 확대만으로 정확한 경계 복원은 어려움. Encoder의 Downsampling 과정에서 세밀한 위치 정보가 감소했기 때문.

### Skip Connection

**Skip Connection:** Encoder의 고해상도 Feature를 같은 공간 크기의 Decoder에 직접 전달하는 연결

```text
Encoder Feature: 어디에 있는가?
Decoder Feature: 무엇인가?
```

U-Net의 일반적인 결합 방식: Channel Concatenation.

```text
Encoder Feature: (64, 64, 128)
Decoder Feature: (64, 64, 128)

Concatenation:   (64, 64, 256)
```

Height와 Width는 유지. Channel만 128 + 128 = 256.

### Channel Concatenation

같은 Pixel 위치의 Encoder 값:

```text
[e1, e2, e3]
```

같은 위치의 Decoder 값:

```text
[d1, d2]
```

더하기가 아니라 순서대로 연결:

```text
[e1, e2, e3, d1, d2]
```

3 Channels + 2 Channels → 5 Channels.

다음 Convolution Layer: Encoder의 위치 정보와 Decoder의 의미 정보를 함께 사용.

### 최종 Output

Binary Segmentation:

```text
Output Shape = (Height, Width, 1)
```

Class가 5개인 Multi-class Segmentation:

```text
Output Shape = (Height, Width, 5)
```

각 Pixel의 Class Score → Sigmoid 또는 Softmax → 최종 Segmentation Mask.

## 발표 마무리

```text
1. Image: Pixel이 H × W × C로 배열된 구조
2. Preprocessing: Load → Resize → Normalize
3. Convolution: Channel별 Kernel 계산 → Feature Map
4. CNN: Local Receptive Field + Weight Sharing
5. Evaluation: Validation으로 Overfitting과 Generalization 확인
6. Segmentation: Pixel마다 Class Prediction
7. U-Net: 의미 Feature + 위치 Feature → 정밀한 Mask
```

발표를 관통하는 질문:

- Input의 Shape과 Pixel 범위는 무엇인가?
- Filter 하나가 몇 개 Channel을 확인하는가?
- 같은 Weight를 왜 여러 위치에서 사용하는가?
- Training 성능이 새로운 Data에서도 유지되는가?
- 최종 Output이 Image Class인가, Pixel Mask인가?
