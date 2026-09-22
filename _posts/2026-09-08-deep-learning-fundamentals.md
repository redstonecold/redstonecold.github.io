---
layout: post
title: "Deep Learning 기초: Neuron에서 경사하강법까지"
date: 2026-09-08 09:00:00 +0900
description: "간단한 예시와 수식으로 Neural Network의 계산과 학습 과정을 정리한다."
tags: [ai, deep-learning, study]
categories: ["AI/DeepLearning"]
math: true
---

<style>
article h2 { margin-top: 3.5rem; }
article h3 { margin-top: 2.5rem; }
</style>

## 초록

Neural Network는 Input을 받아 Output을 계산한다. 학습은 예측이 정답에 가까워지도록 Weight(가중치)와 Bias를 조정하는 과정이다.

손글씨 숫자 분류를 예로 들어 계산 흐름을 살펴본다. Neuron의 계산을 먼저 살펴본다. 이어서 Cost(Loss)와 Gradient를 정리한다. 각 수식 아래에서 기호와 계산 순서를 설명한다.

## 1. Neuron과 Activation

여기서 Neuron은 Artificial Neuron(인공 뉴런)을 뜻한다. 생물학적 Neuron의 작동을 그대로 재현한 것은 아니다. Input에 Weight를 곱하고 Bias와 Activation Function을 적용하는 계산 단위이다. Neuron의 출력은 숫자 하나로 표현한다. 이 숫자가 Activation(활성값)이다. Activation은 해당 Neuron의 반응 정도를 나타낸다.

여기서 다룰 예시는 손글씨 숫자 분류이다. Input 이미지는 28 × 28 Pixel로 구성된다. 따라서 Input에는 784개의 Neuron을 둔다. 각 Neuron은 Pixel 하나의 값을 전달한다. Pixel 값을 0부터 1 사이로 정규화했다면 0.8은 정해진 기준에서 해당 Pixel의 검은 정도가 0.8이라는 뜻이다. Input Layer에서는 이 Pixel 값을 Input Activation이라고도 부른다.

Output에는 10개의 Neuron을 둔다. 각각 숫자 0부터 9에 대응한다. 가장 큰 Activation을 가진 Output Neuron에 해당하는 숫자를 모델의 예측으로 선택한다.

같은 0.8이라도 Layer에 따라 의미가 다르다. Input Activation 0.8은 Pixel의 검은 정도를 나타낸다. Hidden Activation 0.8은 여러 Pixel을 종합한 뒤 해당 Neuron이 보인 반응의 크기이다. Hidden Activation이 0과 1 사이라는 이유만으로 “어떤 Pattern이 존재할 확률이 80%”라는 뜻은 아니다. Output은 문제와 학습 방법에 따라 특정 Class의 예측 확률로 해석할 수 있다.

![Pixel에서 숫자 예측까지](/assets/img/deep-learning-fundamentals/pixels-to-prediction.png){: style="max-width: 100%; height: auto;"}

**그림 설명.** 그림의 격자와 Neuron 수는 이해를 위해 단순화했다. 실제 Input은 28 × 28 Pixel이다. 왼쪽의 이미지를 Pixel별 밝기 값으로 펼친다. Network는 이 값들을 받아 숫자별 Output을 계산한다. 주황색 막대가 가장 큰 숫자 3이 예측 결과이다.

## 2. Hidden Layer와 패턴

Input과 Output 사이에는 Hidden Layer가 있다. 이 예시에는 설명을 쉽게 하기 위해 Hidden Layer를 두 개 사용한다. 간단한 MLP 실험에서는 한두 개를 사용하기도 하지만, Hidden Layer의 개수는 문제와 모델 구조에 따라 달라진다.

각 Layer는 이전 Layer의 Activation을 받아 계산한다. 앞쪽 Layer가 단순한 패턴을, 뒤쪽 Layer가 그 조합을 다룬다고 생각할 수 있다. 숫자 이미지에서는 짧은 획과 획의 조합을 예로 들 수 있다.

이는 구조를 이해하기 위한 직관이다. 개별 Neuron이 반드시 특정 획을 학습한다는 뜻은 아니다. 또한 Hidden Layer의 개수가 항상 두 개로 고정되는 것도 아니다.

## 3. Weight(가중치)와 Weighted Sum

Fully Connected Layer(완전연결 계층)의 각 Neuron은 이전 Layer의 모든 Activation을 받는다. 각 Activation에 서로 다른 Weight(가중치)를 곱한 뒤 그 결과를 모두 더한다.

$$
w_1a_1+w_2a_2+w_3a_3+\cdots+w_na_n
$$

이 값이 Weighted Sum(가중합)이다. 여기서 a는 이전 Layer에 있는 Neuron의 Activation이다. w는 각 연결의 Weight이다.

Weight는 이전 Layer의 각 Activation을 얼마나 강하게 반영할지 결정한다. Activation이 0 이상일 때 Positive Weight는 다음 Neuron의 Weighted Sum을 증가시키고, Negative Weight는 감소시킨다. Weight가 0이면 해당 Activation은 계산에 기여하지 않는다. 일반적으로 실제 기여의 방향과 크기는 Weight와 Activation을 곱한 값으로 결정된다.

예를 들어 Activation이 0.8이라면 Weight 0.5의 기여값은 0.4이고, Weight −0.5의 기여값은 −0.4이다.

![하나의 Neuron이 계산하는 순서](/assets/img/deep-learning-fundamentals/neuron-computation.png){: style="max-width: 100%; height: auto;"}

**그림 설명.** 각 Input에 Weight를 곱한 뒤 더한다. Bias를 더하고 Activation Function을 통과시키면 Output이 나온다. 화살표를 왼쪽부터 따라가 보자.

### 3.1 Linear Combination

이전 Layer의 각 Activation에 해당 Weight를 곱한 뒤 모두 더한 Weighted Sum은 Activation들의 Linear Combination(선형결합)이다.

$$
\text{Weighted Sum}=w_1a_1+w_2a_2+\cdots+w_na_n
$$

여기에 Bias를 더한 계산은 Affine Transformation(아핀 변환)이다.

$$
z=w_1a_1+w_2a_2+\cdots+w_na_n+b
$$

Affine Layer(아핀 계층)는 이전 Layer의 각 Activation에 Weight를 곱해 더하고 Bias를 추가하는 계산을 수행한다. 즉, Neuron이 Activation Function을 적용하기 전의 계산 부분이다. Deep Learning Library에서는 Bias가 포함되어 있어도 편의상 Linear Layer라고 부르기도 한다.

## 4. Sigmoid와 Bias

### 4.1 Sigmoid

Weighted Sum을 그대로 전달하지 않고 Activation Function에 넣는다. 먼저 살펴볼 함수는 Sigmoid이다.

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Sigmoid Function(시그모이드 함수)은 Neuron의 Weighted Sum과 Bias를 더한 값 z를 0과 1 사이의 Activation으로 변환한다. 큰 음수는 0에 가까워진다. 큰 양수는 1에 가까워진다. z가 0이면 결과는 0.5이다.

z가 같은 크기만큼 변해도 Sigmoid Activation은 항상 같은 크기로 변하지 않는다. z가 0 근처에서는 비교적 빠르게 변한다. z가 매우 크거나 작으면 1 또는 0 근처에서 천천히 변한다. 이런 성질을 Non-linearity(비선형성)라고 한다. 이 성질 덕분에 여러 Layer를 연결한 Network가 직선만으로 구분하기 어려운 복잡한 Pattern을 표현할 수 있다.

### 4.2 Bias

Weighted Sum에서 10을 빼는 경우를 생각해 보자.

$$
w_1a_1+w_2a_2+w_3a_3+\cdots+w_na_n-10
$$

여기서 −10이 더해지는 Bias이다. Weighted Sum이 10이면 Sigmoid에 들어가는 값은 0이다. 따라서 Activation은 0.5가 된다.

Weight는 이전 Layer의 각 Activation을 얼마나 강하게 반영할지 결정한다. Bias(편향)는 Weighted Sum 전체를 이동시켜 Neuron이 어느 정도의 신호부터 반응할지 조절한다.

Bias는 더하는 값으로 정의할 수 있다. 아래 행렬식에서는 Bias를 더한다. 이 표기에서 위 예시의 Bias는 −10이다.

### 4.3 Perceptron Model과 Threshold Function

Perceptron(퍼셉트론)은 여러 Input을 받아 두 Class 중 하나를 선택하는 가장 단순한 Artificial Neuron Model이다. 각 Input에 Weight를 곱해 더하고 Bias를 추가한다. 마지막으로 Threshold Function(임계 함수)을 적용해 0 또는 1을 출력한다.

$$
\hat{y}=
\begin{cases}
1 & z\geq 0\\
0 & z<0
\end{cases}
$$

z가 0 이상이면 1을 출력하고, 0보다 작으면 0을 출력한다. 예를 들어 “Spam인가?”, “이 이미지가 숫자 3인가?”처럼 두 가지 중 하나를 판단할 수 있다.

두 Input이 모두 1일 때만 1을 출력하는 AND 문제를 생각해 보자.

$$
z=x_1+x_2-1.5
$$

| x₁ | x₂ | z | Output |
| --- | --- | --- | --- |
| 0 | 0 | −1.5 | 0 |
| 0 | 1 | −0.5 | 0 |
| 1 | 0 | −0.5 | 0 |
| 1 | 1 | 0.5 | 1 |

Threshold Function은 경계에서 불연속적이다. Sigmoid처럼 부드럽게 변하지 않는다. 따라서 이 함수를 그대로 사용해 일반적인 Backpropagation으로 학습하는 것은 적절하지 않다. 고전적인 Perceptron은 별도의 Perceptron Learning Rule을 사용한다.

### 4.4 Decision Boundary와 Perceptron Learning Rule

Decision Boundary(결정 경계)는 예측 Class가 바뀌는 위치이다. Input이 두 개라면 데이터를 x₁과 x₂의 좌표로 나타낼 수 있다. Perceptron의 판단이 바뀌는 경계에서는 z가 0이다.

$$
w_1x_1+w_2x_2+b=0
$$

이 식은 좌표평면에서 직선이다. 직선의 한쪽에서는 0을, 반대쪽에서는 1을 출력한다. 앞의 AND 예제에서는 Decision Boundary가 x₁ + x₂ = 1.5이다. Input이 세 개이면 데이터를 3차원 공간에 나타낼 수 있고, 경계는 평면이 된다.

Perceptron Learning Rule(퍼셉트론 학습 규칙)은 Perceptron이 틀린 예측을 했을 때 Weight와 Bias를 수정하는 방법이다. 학습할 때 정답 y와 예측값 ŷ를 비교한다. 둘 다 0 또는 1로 표현하자.

| 결과 | y − ŷ | 업데이트 |
| --- | --- | --- |
| 정답과 예측이 같음 | 0 | 변경하지 않음 |
| 정답 1, 예측 0 | +1 | Input 방향으로 Weight를 더하고 Bias를 높임 |
| 정답 0, 예측 1 | −1 | Input 방향으로 Weight를 빼고 Bias를 낮춤 |

각 Weight와 Bias는 다음 규칙으로 수정한다.

$$
w_i\leftarrow w_i+\eta(y-\hat{y})x_i
$$

$$
b\leftarrow b+\eta(y-\hat{y})
$$

xᵢ는 i번째 Input이다. η는 Learning Rate(학습률)이다. Learning Rate는 한 번 틀렸을 때 값을 얼마나 크게 수정할지 정하는 Hyperparameter이다. 보통 0.1, 0.01, 0.001처럼 서로 다른 값을 실험하고 Validation Data의 결과를 비교해 선택한다. 너무 작으면 학습이 느리고, 너무 크면 적절한 값을 지나치며 학습이 흔들릴 수 있다.

예를 들어 Input이 (1, 0), 정답이 1, 예측이 0이고 Learning Rate가 0.1이라고 하자. y − ŷ는 1이다. 첫 Weight에는 0.1 × 1 × 1을 더한다. 두 번째 Weight는 Input이 0이므로 변하지 않는다. Bias에는 0.1을 더한다. 그 결과 같은 Input에서 z가 커지고, 다음에는 1로 판단하기 쉬워진다.

이것이 Perceptron Learning Rule이다. 선형 분리가 가능한 유한한 데이터에서는 표준 규칙이 유한 횟수의 오류 수정 후 분리 경계를 찾는다. 선형 분리가 불가능하면 수렴이 보장되지 않는다.

### 4.5 Single-Layer Perceptron의 한계: XOR Problem

XOR은 두 Input이 다를 때만 1을 출력한다.

| Input 1 | Input 2 | XOR |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

네 점을 정사각형 꼭짓점에 놓아 보자. 같은 Class가 대각선으로 마주 본다. 직선 하나로 두 Class를 나눌 수 없다. 따라서 단일 Perceptron은 XOR을 표현할 수 없다.

![단일 직선으로 분리할 수 없는 XOR](/assets/img/dl-concepts/xor.svg)

**그림 읽기.** 청록색 두 점은 1, 주황색 두 점은 0이다. 한쪽 색을 직선의 한 편에 모두 모으려 하면 다른 색이 함께 들어온다.

이 문제는 학습 시간을 늘리는 것만으로 해결되지 않는다. 모델이 만들 수 있는 Decision Boundary 자체에 한계가 있기 때문이다. Hidden Layer가 필요한 이유를 여기서 볼 수 있다.

## 5. Layer 전체의 계산

여러 Neuron의 계산을 한 번에 묶으면 다음과 같다.

$$
a^{(1)}=\sigma\left(Wa^{(0)}+b\right)
$$

이 식은 Layer 전체의 계산을 한 줄로 표현한다.

- a⁽⁰⁾: 이전 Layer의 Activation을 모은 vector이다.
- W: Weight(가중치)를 모은 행렬이다.
- b: Bias를 모은 vector이다.
- a⁽¹⁾: 계산을 마친 다음 Layer의 Activation이다.

먼저 W와 a⁽⁰⁾을 곱한다. 이어서 b를 더한다. 마지막으로 각 성분에 Sigmoid를 적용한다.

$$
\sigma\left(
\begin{bmatrix}
x\\y\\z
\end{bmatrix}
\right)
=
\begin{bmatrix}
\sigma(x)\\\sigma(y)\\\sigma(z)
\end{bmatrix}
$$

Sigmoid는 vector 전체를 숫자 하나로 바꾸지 않는다. x, y, z에 각각 적용된다.

이 계산을 Input에서 Output까지 반복한다. 이 과정이 Forward Propagation이다.

### 5.1 Multilayer Perceptron (MLP)

MLP는 Input Layer, 하나 이상의 Hidden Layer, Output Layer로 구성된 Feedforward Network이다. 보통 인접 Layer 사이를 완전연결한다. 각 Hidden Neuron은 Weight, Bias, 비선형 Activation Function으로 계산한다.

| Layer | 역할 |
| --- | --- |
| Input Layer | Pixel이나 측정값 같은 Input Data를 전달 |
| Hidden Layer | 학습한 변환으로 새로운 특징을 계산 |
| Output Layer | 예측값을 문제에 맞는 형태로 출력 |

이 글의 손글씨 Network가 MLP의 한 예이다. MLP라는 이름을 쓰더라도 Hidden Layer가 고전적인 Threshold Function을 사용해야 하는 것은 아니다. Sigmoid나 ReLU처럼 Gradient 기반 학습에 사용할 수 있는 함수를 쓴다.

### 5.2 Non-linearity와 Non-linear Representation

Non-linear Representation(비선형 표현)은 Input 사이의 복잡한 관계를 직선이나 평면에 제한되지 않는 형태로 나타내는 것이다.

먼저 Activation Function 없이 Affine Layer만 두 개 연결해 보자. 첫 번째 Layer와 두 번째 Layer가 다음과 같이 계산한다고 하자.

$$
z_1=2x+1
$$

$$
y=3z_1-4
$$

첫 번째 식을 두 번째 식에 넣으면 전체 Output은 다음과 같다.

$$
y=3(2x+1)-4=6x-1
$$

두 개의 Affine Layer를 사용했지만 하나의 Affine Layer와 같은 계산이다. Activation Function 없이 Affine Layer만 여러 개 연결하면 전체 계산은 다시 하나의 Affine Transformation으로 합쳐진다. 따라서 Hidden Layer를 추가해 Network를 깊게 만들어도 Decision Boundary는 여전히 직선이며 XOR을 구분할 수 없다.

Layer 사이에 Sigmoid나 ReLU 같은 Non-linear Activation Function을 넣으면 전체 계산을 하나의 Affine Transformation으로 합칠 수 없다. Hidden Layer는 원래 Input을 구분하기 쉬운 Intermediate Feature(중간 특징)로 변환할 수 있다.

XOR에서는 첫 번째 Hidden Neuron이 “두 Input 중 적어도 하나가 1인가?”를 판단하고, 두 번째 Hidden Neuron이 “두 Input이 모두 1인가?”를 판단한다고 생각할 수 있다.

| x₁ | x₂ | 적어도 하나가 1인가? | 둘 다 1인가? | XOR |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 | 1 |
| 1 | 0 | 1 | 0 | 1 |
| 1 | 1 | 1 | 1 | 0 |

첫 번째 조건만 참일 때 XOR은 1이다. Output Layer는 두 Intermediate Feature를 조합하여 XOR을 판단한다. 실제 Training에서 각 Hidden Neuron이 반드시 OR와 AND라는 이름 그대로 학습하는 것은 아니다. 이 예시는 Hidden Layer가 XOR을 해결하는 원리를 쉽게 보여 준다.

### 5.3 Representation Learning

Representation Learning(표현 학습)은 Network가 문제를 해결하는 데 필요한 특징을 데이터로부터 학습하는 과정이다. 손글씨 분류에서는 Pixel을 그대로 비교하는 대신 Hidden Layer가 선, 곡선, 모서리처럼 분류에 유용한 조합을 만들 수 있다. 즉, 원래 Input을 분류하기 쉬운 새로운 Representation(표현)으로 변환한다.

특징 하나가 Neuron 하나에 명확히 대응할 필요는 없다. 여러 Neuron의 Activation이 함께 하나의 특징을 표현할 수 있다. 따라서 모든 Hidden Neuron에 “가로획 감지기”처럼 이름을 붙일 수 있는 것은 아니다.

### 5.4 Universal Approximation과 Network Depth

Universal Approximation(보편 근사)은 MLP가 충분히 많은 Hidden Neuron을 사용하면 일정한 Input 범위 안에서 연속적인 함수의 모양을 원하는 수준까지 비슷하게 표현할 수 있다는 결과이다. Hidden Layer의 폭은 한 Hidden Layer에 들어 있는 Neuron의 개수이다. 근사는 완전히 같지는 않아도 차이를 원하는 만큼 작게 만든다는 뜻이다.

손글씨 분류에 연결하면, Neural Network가 Pixel 값과 숫자 Class 사이의 복잡한 관계를 표현할 능력이 있다는 뜻이다. 비슷한 손글씨 모양을 실제로 학습하는 과정은 Representation Learning에 해당한다.

Universal Approximation은 작은 Network가 모든 함수를 정확히 표현한다는 뜻이 아니다. 필요한 Neuron의 수, 적절한 Weight를 쉽게 찾을 수 있는지, 새로운 데이터에서도 잘 예측하는지는 보장하지 않는다.

Depth가 늘면 여러 단계의 특징을 조합할 수 있다. 어떤 함수는 얕고 매우 넓은 Network보다 깊은 Network에서 효율적으로 표현된다. 반면 계산 비용과 최적화의 어려움도 커진다. “깊을수록 항상 성능이 좋다”는 결론은 성립하지 않는다. [관련 설명: Deep Learning, Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html).

## 6. ReLU

Activation Function에는 ReLU도 있다. ReLU는 Rectified Linear Unit의 약자이다.

$$
\operatorname{ReLU}(a)=\max(0,a)
$$

a가 음수이면 0을 출력한다. 양수이면 a를 그대로 출력한다.

따라서 ReLU의 Activation은 1보다 클 수 있다. 모든 Activation이 0과 1 사이에 있는 것은 아니다. 범위는 사용하는 Activation Function에 따라 달라진다.

![Layer 사이의 흐름과 Activation Function](/assets/img/deep-learning-fundamentals/layers-and-activation.png){: style="max-width: 100%; height: auto;"}

**그림 설명.** 왼쪽 그림은 이해를 위해 Neuron 수를 줄인 구조이다. 정보는 왼쪽에서 오른쪽으로 전달된다. 오른쪽은 Sigmoid와 ReLU가 들어온 값을 바꾸는 방식이다.

## 7. Cost(Loss): 예측과 정답의 차이

Cost 또는 Loss는 Training을 마친 뒤에만 계산하는 값이 아니다. Training 과정에서 Forward Propagation으로 Prediction을 만든 다음, 예측이 정답에서 얼마나 벗어났는지 계산하는 값이다.

Training에서는 **Input → Forward Propagation → Prediction → Loss 계산 → Backpropagation → Parameter Update**의 순서를 반복한다.

여기서 Parameter(매개변수)는 Training으로 수정하는 Weight와 Bias를 뜻한다. Activation은 Input과 Parameter로 계산되는 중간 결과이며 직접 수정하는 Parameter가 아니다.

손글씨 숫자 0부터 9까지 분류한다면 Output Layer에는 10개의 Neuron이 있다. 각 Output은 해당 Output Neuron의 Activation, 즉 각 숫자에 대한 예측값이다. 최종 선택된 숫자 자체를 뜻하지 않는다.

정답이 숫자 3이라면 숫자 3의 목표값만 1이고 나머지는 0이다.

$$
\mathbf{y}=[0,0,0,1,0,0,0,0,0,0]
$$

정답 Class의 위치만 1로 나타내는 방식을 One-hot Encoding(원-핫 인코딩)이라고 한다.

학습 원리를 간단히 보기 위해 각 Output과 목표값의 차이를 제곱하고 모두 더해 보자. 아래는 숫자 3을 분류할 때의 계산 일부이다.

$$
\begin{aligned}
&(0.43-0.00)^2+\cdots\\
&\quad +(0.88-1.00)^2+\cdots
\end{aligned}
$$

따라서 정답에 해당하는 Output은 1에 가까워야 한다. 나머지 Output은 0에 가까워야 한다. 예측과 목표의 차이가 커질수록 Cost가 커진다.

전체 Cost를 계산할 때는 생략된 항을 포함해 10개 Output의 제곱오차를 모두 더한다.

### 7.1 Classification에 맞는 Output과 Loss

제곱오차는 학습 원리를 설명하기에 간단하다. 실제 분류에서는 Output의 의미에 맞춰 Loss Function을 선택한다.

| 문제 | Output 계산 | Loss |
| --- | --- | --- |
| Binary Classification(이진 분류) | 하나의 Logit을 Sigmoid로 변환하여 Class 1의 예측 확률을 계산 | Binary Cross-Entropy(이진 교차 엔트로피) |
| Multi-class Classification(다중 클래스 분류) | 각 Class의 Logit을 Softmax로 변환하여 Class별 예측 확률을 계산 | Categorical Cross-Entropy(범주형 교차 엔트로피) |

Logit(로짓)은 Output Layer에서 Weighted Sum과 Bias를 더한 값이다. 아직 Sigmoid나 Softmax를 적용하기 전의 점수이므로 음수나 1보다 큰 값도 가능하다. Logit 자체는 확률이 아니다.

Classification의 계산 순서는 **이전 Layer의 Activation → Weighted Sum과 Bias → Logit → Sigmoid 또는 Softmax → 예측 확률 → Cross-Entropy Loss**이다.

### 7.2 Binary Classification: Sigmoid

예를 들어 “고양이인가?”처럼 두 Class를 구분한다. Sigmoid Output이 0.8이면 모델이 Class 1에 부여한 확률은 0.8이다. Class 0에는 0.2가 대응한다. 실제 발생 빈도와 완벽히 일치하는 확률이라는 보장은 없다.

확률을 최종 Class로 바꾸려면 Threshold가 필요하다. 기본 예시로 0.5 이상이면 1로 분류할 수 있다. 놓치는 오류와 잘못 경고하는 오류의 비용에 따라 Threshold는 달라질 수 있다.

여기서 분류용 Threshold는 예측을 해석하는 단계이다. 학습 중에는 부드러운 Sigmoid Output으로 Loss와 Gradient를 계산한다.

### 7.3 Multi-class Classification: Softmax

숫자 0–9 중 하나를 고르는 문제에는 서로 배타적인 Class가 10개 있다. 하나의 이미지가 숫자 3과 숫자 7에 동시에 해당하지 않고 하나의 정답만 가진다는 뜻이다. Output Layer의 10개 Neuron은 각 숫자에 대한 Logit을 계산한다. 높은 Logit은 Network가 해당 Class라고 더 강하게 판단했다는 뜻이다.

Softmax Function(소프트맥스 함수)은 모든 Logit을 비교하여 합이 1인 Class별 예측 확률로 바꾼다. 계산은 **각 Logit에 지수함수 적용 → 그 결과의 전체 합으로 나누기**이다.

Class가 총 $K$개일 때, $j$번째 Class의 확률은 다음과 같다.

$$
p_j=\frac{e^{z_j}}{\displaystyle\sum_{k=1}^{K}e^{z_k}}
$$

- $z_j$: $j$번째 Class의 Logit
- $e^{z_j}$: Logit에 지수함수를 적용한 Positive 값
- $K$: 전체 Class의 수
- $p_j$: $j$번째 Class의 예측 확률

분모는 모든 Class의 $e^{z_k}$를 더한 값이다. 모든 Class가 같은 분모를 사용하므로 각 확률의 합은 1이 된다.

Class가 세 개이고 Logit이 (2, 1, 0)이라고 하자.

| Class | Logit | 지수함수 결과 | Softmax 확률 |
| --- | --- | --- | --- |
| Class 1 | 2 | 7.389 | 7.389 ÷ 11.107 ≈ 0.665 |
| Class 2 | 1 | 2.718 | 2.718 ÷ 11.107 ≈ 0.245 |
| Class 3 | 0 | 1 | 1 ÷ 11.107 ≈ 0.090 |

지수함수의 결과는 항상 Positive이므로 음수 Logit도 Positive 값으로 바뀐다. 세 지수함수 결과의 합은 11.107이다. 각 값을 이 합으로 나누면 Softmax 확률 (0.665, 0.245, 0.090)을 얻는다. 세 확률의 합은 1이며, 가장 높은 확률을 가진 Class 1을 최종 예측으로 선택한다.

Logit이 (0, 0, 0)처럼 모두 같으면 지수함수 결과도 모두 1이다. 따라서 세 Class의 확률은 각각 1/3이 된다. Network가 어느 Class도 더 강하게 선택하지 않은 상태이다.

Sigmoid를 각 Logit에 따로 적용하면 (0.8, 0.7, 0.6)처럼 합이 1보다 큰 결과가 나올 수 있다. Softmax는 한 Class의 확률이 높아지면 다른 Class에 배분되는 확률이 줄어들도록 모든 Logit을 함께 비교한다. 따라서 여러 Class 중 하나만 정답인 Multi-class Classification에 적합하다.

### 7.4 Binary Cross-Entropy와 Categorical Cross-Entropy

Cross-Entropy Loss(교차 엔트로피 손실)는 모델이 **정답에 얼마나 높은 확률을 주었는지** 평가한다. 정답에 높은 확률을 주면 Loss가 작다. 자신 있게 틀리면 Loss가 크게 증가한다.

#### 7.4.1 Binary Cross-Entropy(이진 교차 엔트로피)

Binary Classification에서는 Output Logit $z$에 Sigmoid를 적용한다.

$$
p=\sigma(z)
$$

여기서 $p$는 모델이 예측한 Class 1의 확률이다. 즉 $p=P(y=1\mid x)$로 해석한다. $x$는 Input이고, $y$는 정답이다.

정답 $y$는 0 또는 1이다. 개별 example의 Binary Cross-Entropy는 다음과 같다.

$$
L=-\left[y\log(p)+(1-y)\log(1-p)\right]
$$

- 정답이 $y=1$이면 $L=-\log(p)$만 남는다.
- 정답이 $y=0$이면 $L=-\log(1-p)$만 남는다.

따라서 정답이 1일 때는 $p$가 1에 가까울수록 좋다. 정답이 0일 때는 $p$가 0에 가까울수록 좋다.

예를 들어 정답이 1이고 $p=0.9$라면 Loss는 약 0.105이다. 같은 정답에 $p=0.1$을 예측하면 Loss는 약 2.303이다.

Sigmoid의 출력이 0과 1 사이라는 이유만으로 자동으로 확률이 되는 것은 아니다. Output Neuron을 “Class 1”로 정의하고, Binary Cross-Entropy로 학습할 때 $p$를 Class 1의 예측 확률로 해석한다.

#### 7.4.2 Categorical Cross-Entropy(범주형 교차 엔트로피)

서로 배타적인 Class가 여러 개이면 Softmax와 Categorical Cross-Entropy를 함께 사용한다. One-hot 정답을 $y_j$, Softmax가 만든 $j$번째 Class의 확률을 $p_j$라고 하자.

$$
L=-\sum_{j=1}^{K}y_j\log(p_j)
$$

- $K$: 전체 Class의 수
- $y_j$: $j$번째 위치의 One-hot 정답. 정답 Class만 1이고 나머지는 0이다.
- $p_j$: Softmax가 만든 $j$번째 Class의 예측 확률

정답 위치의 $y_j$만 1이므로 실제 계산에는 정답 Class의 확률만 남는다.

$$
L=-\log(p_{\text{correct}})
$$

예를 들어 정답이 “개”이고 One-hot 정답이 $(0,1,0)$이라고 하자. Softmax Prediction이 $(0.2,0.7,0.1)$이면 $p_{\text{correct}}=0.7$이다.

$$
L=-\log(0.7)\approx0.357
$$

다른 Class의 확률을 무시하는 것은 아니다. Softmax 확률의 합은 1이다. 다른 Class에 더 높은 확률을 주면 정답 Class의 확률이 작아지고 Loss가 커진다.

| 구분 | 예측 확률 | 정답 | Loss가 작아지는 경우 |
| --- | --- | --- | --- |
| Binary Cross-Entropy | Sigmoid가 만든 Class 1 확률 $p$ | $y\in\{0,1\}$ | 정답이 1이면 $p\to1$, 정답이 0이면 $p\to0$ |
| Categorical Cross-Entropy | Softmax가 만든 Class별 확률 $p_j$ | One-hot $y_j$ | 정답 Class의 $p_j\to1$ |

### 7.5 Training과 Inference(추론)

Training과 Inference는 Deep Learning Model을 학습하고 사용하는 두 단계이다. 둘 다 학습 과정이라는 뜻은 아니다.

| Training(학습) | Inference(추론) |
| --- | --- |
| Training Data와 정답을 사용 | 새로운 Input을 사용 |
| Forward Propagation과 Loss 계산 | Forward Propagation으로 Prediction 계산 |
| Backpropagation 수행 | Backpropagation을 수행하지 않음 |
| Weight와 Bias 수정 | Weight와 Bias를 수정하지 않음 |

Inference(추론)는 Training을 마친 Model에 새로운 Input을 넣어 Prediction을 얻는 과정이다. 정답을 입력하지 않아도 예측할 수 있다. Input 정규화 등 전처리는 Training 때의 기준을 유지한다.

## 8. Neural Network와 Cost Function

Neural Network와 Cost Function은 서로 다른 역할을 한다. Neural Network는 Input Data를 받아 Prediction을 만든다. Cost Function은 Prediction과 실제 정답을 비교해 Cost를 계산한다.

전체 계산 순서는 **Input Data → Neural Network → Prediction → Cost Function → Cost**이다.

Weight와 Bias가 바뀌면 Prediction이 달라지고, 그 결과 Cost도 달라진다. Training은 Cost가 작아지는 방향으로 Weight와 Bias를 수정하는 과정이다.

이 예시 Network에는 13,002개의 Weight와 Bias가 있다. 이 Parameter들은 Network 전체에 나뉘어 있으며, 하나의 Neuron 안에 들어 있는 것이 아니다. 13,002개는 이 예시만의 Parameter 수이며 모든 Neural Network가 같은 수를 갖는 것은 아니다.

Training Dataset은 Parameter를 조정하는 데 사용한다. Test Dataset은 학습에 사용하지 않은 데이터에서 성능을 확인하는 데 사용한다.

### 8.1 Model Capacity, Overfitting, Generalization

Model Capacity(모델 용량)는 모델이 얼마나 다양하고 복잡한 관계를 표현할 수 있는지를 뜻한다. Layer의 폭과 깊이, Activation Function 등이 영향을 준다. Parameter 수는 참고 지표이지만 Capacity를 완전히 설명하지는 않는다.

Capacity가 부족하면 Training Data에서 반복해서 나타나는 Input과 정답 사이의 Pattern을 충분히 학습하지 못할 수 있다. 반대로 모델이 Training Data에만 있는 잡음이나 우연한 차이까지 지나치게 맞추면 Overfitting(과적합)이 생길 수 있다.

Generalization(일반화)은 Training에 직접 사용하지 않은 새로운 데이터도 잘 예측하는 능력이다. Training Loss는 Training Data에서의 오차이다. Validation Loss는 Weight와 Bias를 학습하는 데 사용하지 않은 Validation Data에서의 오차이다.

학습 초반에는 두 Loss가 함께 작아질 수 있다. 하지만 Training Loss는 계속 작아지는데 Validation Loss가 일정 기간 커진다면, Model이 Training Data에 지나치게 맞춰져 새로운 데이터를 잘 예측하지 못할 가능성이 있다. 이때 Overfitting을 의심한다. 단 한 번의 작은 변화만으로 판단하지 않고 일정 기간의 추세와 데이터 분포도 함께 확인한다.

| 데이터 구분 | 용도 |
| --- | --- |
| Training | Weight와 Bias 학습 |
| Validation | 모델 크기, Learning Rate, 학습 종료 시점 선택 |
| Test | 선택을 마친 모델의 최종 성능 평가 |

Test 결과를 반복해서 보고 설정을 고르면 Test도 간접적으로 학습에 사용한 셈이 된다. 더 많은 적절한 데이터, Regularization, Early Stopping은 Overfitting을 줄이는 데 도움이 될 수 있다. Early Stopping은 보통 Validation 성능을 기준으로 학습을 멈춘다.

### 8.2 Overfitting을 줄이는 방법

Overfitting을 줄이는 핵심은 Training Data를 외우기 어렵게 만들고, 새로운 데이터에도 반복되는 Pattern을 배우게 하는 것이다.

1. **더 다양하고 충분한 Data를 사용한다.** 다양한 예시가 있으면 우연한 특징을 외우기 어려워진다.
2. **Data Augmentation(데이터 증강)을 적용한다.** 이미지의 회전, 이동, 밝기 변화처럼 정답을 유지하는 변형을 추가한다.
3. **Model Capacity를 줄인다.** Layer나 Neuron 수를 줄여 필요 이상으로 복잡한 Pattern을 표현하지 못하게 한다.
4. **L2 Regularization 또는 Weight Decay를 사용한다.** 지나치게 큰 Weight에 불이익을 준다.
5. **Dropout을 사용한다.** Training 중 일부 Activation을 무작위로 0으로 만들어 특정 Neuron에만 의존하지 않게 한다.
6. **Early Stopping을 사용한다.** Validation Loss가 더 이상 좋아지지 않거나 계속 커지면 학습을 멈춘다.

L2 Regularization을 적용한 Cost는 다음처럼 나타낼 수 있다.

$$
C_{\text{total}}=C_{\text{data}}+\lambda\sum_i w_i^2
$$

- $C_{\text{data}}$: Prediction과 정답에서 계산한 원래 Cost
- $w_i$: 각 Weight
- $\lambda$: 큰 Weight에 줄 불이익의 강도를 정하는 Hyperparameter

한 방법이 언제나 가장 좋은 것은 아니다. Validation Data로 효과를 확인하며 조합을 선택한다. Test Data는 선택이 끝난 뒤 최종 평가에 사용한다.

## 9. Gradient와 경사하강법(Gradient Descent)

### 9.1 Cost를 줄이는 방향

Weight와 Bias를 바꾸면 Prediction이 달라지고 Cost도 변한다. 학습에서는 Cost가 작아지는 방향을 찾는다.

먼저 하나의 Weight만 바꾸고 다른 Parameter는 고정한 Cost 그래프를 생각해 보자. 가로축은 Weight이고 세로축은 Cost인 2차원 곡선이다. 현재 위치의 기울기를 보면 어느 방향으로 움직일지 판단할 수 있다.

Weight 두 개를 함께 바꾸면 두 Weight와 Cost를 축으로 하는 3차원 표면으로 나타낼 수 있다. 실제 Neural Network의 Cost는 Network 전체의 많은 Weight와 Bias에 의해 결정되므로 고차원 함수이다. Parameter가 n개라면 다음과 같이 표현할 수 있다.

$$
C=C(\theta_1,\theta_2,\ldots,\theta_n)
$$

여기서 각 θ는 하나의 Weight 또는 Bias이다. 하나의 Weight에 대한 2차원 Cost 그래프는 이 고차원 함수에서 다른 Parameter를 고정하고 한 방향만 잘라 본 단면이다.

여러 Parameter를 함께 다룰 때 사용하는 것이 Gradient이다. Gradient는 각 Parameter에 대한 편미분을 모은 값이다. Cost Function이 미분 가능한 위치라면 Parameter가 몇 개이든 각 Parameter에 대한 편미분을 계산할 수 있다. ReLU가 0인 지점처럼 미분값이 하나로 정해지지 않는 곳에서는 실제 구현이 사용할 값을 정해 계산한다.

### 9.2 음의 Gradient

모든 Weight(가중치)와 Bias를 W vector로 묶어 표현하자. 이때 Cost를 줄이는 국소 방향은 다음과 같다.

$$
-\nabla C(\vec{W})
$$

여기서 W vector는 Bias까지 포함한다. 앞의 Layer 계산식에서 사용한 Weight 행렬 W와는 역할이 다르다.

Gradient는 Cost가 가장 빠르게 증가하는 국소 방향을 나타낸다. 음의 Gradient는 반대 방향이다. 이는 유클리드 거리 기준의 설명이다.

여기서 Parameter는 Activation이 아니라 Training으로 수정하는 Weight와 Bias이다. Gradient의 각 성분은 하나의 Parameter에 대응한다. 편미분이 Positive이면 그 Parameter를 증가시킬 때 Cost가 증가하므로, Cost를 줄이기 위해 Parameter를 감소시킨다. 편미분이 Negative이면 Parameter를 증가시킬 때 Cost가 감소하므로 Parameter를 증가시킨다. 편미분의 절댓값은 현재 위치에서 해당 Parameter의 변화에 Cost가 얼마나 민감한지를 나타낸다.

음의 Gradient 방향으로 조금 이동한다. 새 위치에서 다시 Gradient를 계산한다. 이 과정을 반복하는 방법이 경사하강법(Gradient Descent)이다.

한 번에 너무 멀리 이동하면 Cost가 오히려 커질 수 있다. 따라서 이동 크기도 중요하다.

### 9.3 두 변수로 살펴보기

다음 Cost Function을 생각해 보자.

$$
C(x,y)=\frac{3}{2}x^2+\frac{1}{2}y^2
$$

이 함수에서 x와 y는 조정할 두 좌표이다. 현재 위치를 (1, 1)로 두면 Gradient는 다음과 같다.

$$
\nabla C(1,1)=
\begin{bmatrix}
3\\1
\end{bmatrix}
$$

현재 위치에서 x에 대한 편미분은 3이다. y에 대한 편미분은 1이다. 따라서 음의 Gradient 방향에서는 두 값 모두 줄인다. x 방향의 변화량은 y 방향보다 크게 잡힌다.

이는 현재 위치의 국소적인 설명이다. x가 언제나 y보다 세 배 중요하다는 뜻은 아니다.

### 9.4 Local Minimum과 Global Minimum

Local Minimum(지역 최솟값)은 주변보다 낮지만 Cost Function 전체에서 가장 낮지는 않을 수 있는 지점이다. Global Minimum(전역 최솟값)은 전체에서 가장 낮은 지점이다.

Gradient Descent는 Cost Function 전체를 미리 보고 이동하지 않는다. 현재 위치의 Gradient만 사용해 Cost가 작아지는 방향으로 이동한다. 따라서 Local Minimum에 도착하거나 매우 평평한 구간에서 이동이 느려질 수 있으며, 항상 Global Minimum을 찾는 것은 아니다.

Gradient가 0인 지점도 반드시 Minimum은 아니다. 골짜기 바닥인 Minimum일 수 있지만, 산꼭대기인 Maximum이나 방향에 따라 올라가기도 하고 내려가기도 하는 Saddle Point(안장점)일 수도 있다.

또한 Training Loss가 작아져도 Overfitting으로 Validation 성능이 나빠질 수 있다. 따라서 학습 과정에서는 Training Loss뿐 아니라 Validation Loss와 Accuracy 같은 평가 성능도 함께 확인한다.

![예측 오차와 Cost를 줄이는 과정](/assets/img/deep-learning-fundamentals/loss-and-gradient.png){: style="max-width: 100%; height: auto;"}

**그림 설명.** 왼쪽은 숫자 0–4의 Output만 표시한 예시이다. 청록색은 Prediction, 주황색은 Target이다. Target이 0인 막대는 높이가 없어 보이지 않는다. 두 막대의 차이가 작아지는 것이 목표이다. 오른쪽은 하나의 Weight만 바꾼 단순한 Cost 그래프이다. 점과 화살표는 Cost가 낮아지는 쪽으로 조금씩 이동하는 과정을 보여 준다.

## 10. 이번 학습의 연결

Neuron은 Weighted Sum과 Bias를 계산한다. Activation Function은 그 결과를 변환한다. 이 계산이 Layer를 따라 반복되면 Output이 나온다.

Cost(Loss)는 Output과 정답의 차이를 나타낸다. Gradient는 Parameter를 어느 방향으로 조정할지 알려 준다. 경사하강법(Gradient Descent)은 이 정보를 이용해 Cost를 줄여 나간다.

여러 Layer의 Gradient 계산, Vanishing Gradient, Mini-batch 학습 단위는 [Backpropagation과 Chain Rule](/blog/2026/backpropagation/)에서 이어서 다룬다.

## 부록: Hidden Layer에서 Sigmoid보다 ReLU를 많이 사용하는 이유

Sigmoid와 ReLU는 모두 $z$를 Activation $a$로 바꾸는 Activation Function이다.

$$
a=f(z)
$$

Backpropagation에서는 뒤에서 온 Gradient $\partial C/\partial a$에 Activation Function의 미분값을 곱한다.

$$
\frac{\partial C}{\partial z}
=
\frac{\partial C}{\partial a}
\frac{\partial a}{\partial z}
$$

여기서 $f'(z)=\partial a/\partial z$이다. 즉 $z$가 조금 변할 때 Activation $a$가 얼마나 변하는지를 나타낸다.

Sigmoid의 미분값은 최대 0.25이다. $z$가 매우 크거나 작으면 미분값이 0에 가까워진다. 여러 Hidden Layer에서 작은 값이 반복해서 곱해지면 앞쪽 Layer에 도착하는 Gradient가 매우 작아질 수 있다.

$$
0.8\times0.1\times0.1\times0.1=0.0008
$$

![Sigmoid와 ReLU를 지날 때 Gradient 크기 비교](/assets/img/deep-learning-fundamentals/relu-vs-sigmoid-gradient.svg){: style="max-width: 100%; height: auto;"}

**그림 읽기.** 같은 Gradient 0.8이 출발해도 작은 Sigmoid 미분값을 반복해서 곱하면 빠르게 작아진다. ReLU의 Positive 영역에서는 미분값 1을 곱하므로 Activation Function 때문에 크기가 줄지 않는다. 숫자는 원리를 보여 주기 위한 예시이다.

ReLU는 다음과 같다.

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

$$
\frac{\partial a}{\partial z}
=
\begin{cases}
1, & z>0\\
0, & z<0
\end{cases}
$$

Positive 영역에서는 미분값이 1이다. 따라서 이 구간에서는 Activation Function 때문에 Gradient가 계속 작아지지 않는다. 계산도 단순하다. 이 두 이유로 깊은 Network의 Hidden Layer에서는 ReLU가 Sigmoid보다 자주 사용된다.

$z=0$에서는 수학적인 미분값이 하나로 정해지지 않는다. 실제 구현에서는 보통 0으로 정한다.

$\partial C/\partial z$를 구한 뒤에는 계산이 끝난 것이 아니다. 이 값으로 현재 Layer의 Weight와 Bias Gradient를 구하고, 이전 Layer로 Gradient를 전달한다.

$$
\frac{\partial C}{\partial W}=\frac{\partial C}{\partial z}a_{\text{prev}}^{T},
\qquad
\frac{\partial C}{\partial b}=\frac{\partial C}{\partial z},
\qquad
\frac{\partial C}{\partial a_{\text{prev}}}=W^{T}\frac{\partial C}{\partial z}
$$

ReLU도 한계가 있다. $z<0$인 구간에서는 미분값이 0이므로 Neuron이 계속 업데이트되지 않는 Dying ReLU가 생길 수 있다. 또한 Output Layer의 Activation은 문제에 맞춰 선택한다. Binary Classification에는 보통 Sigmoid를, 서로 배타적인 Multi-class Classification에는 Softmax를 사용한다.
