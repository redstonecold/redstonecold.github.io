---
layout: post
title: "Deep Learning 기초: Neuron에서 경사하강법까지"
date: 2026-09-08 09:00:00 +0900
description: "간단한 예시와 수식으로 Neural Network의 계산과 학습 과정을 정리한다."
tags: [ai, deep-learning, study]
categories: ["AI/DeepLearning"]
math: true
---

## 초록

Neural Network는 Input을 받아 Output을 계산한다. 학습은 예측이 정답에 가까워지도록 Weight(가중치)와 Bias를 조정하는 과정이다.

손글씨 숫자 분류를 예로 들어 계산 흐름을 살펴본다. Neuron의 계산을 먼저 살펴본다. 이어서 Cost(Loss)와 Gradient를 정리한다. 각 수식 아래에서 기호와 계산 순서를 설명한다.

## 1. Neuron과 Activation

여기서 Neuron은 Artificial Neuron을 뜻한다. 생물학적 Neuron의 작동을 그대로 재현한 것은 아니다. Input에 Weight를 곱하고 Bias와 Activation Function을 적용하는 계산 단위이다. Neuron의 출력은 숫자 하나로 표현한다. 이 숫자가 Activation이다. Activation은 해당 Neuron의 반응 정도를 나타낸다.

여기서 다룰 예시는 손글씨 숫자 분류이다. Input 이미지는 28 × 28 Pixel로 구성된다. 따라서 Input에는 784개의 Neuron을 둔다. 각 Neuron은 Pixel 하나의 값을 전달한다.

Output에는 10개의 Neuron을 둔다. 각각 숫자 0부터 9에 대응한다. Output이 큰 숫자를 모델의 예측으로 선택한다.

단, Activation이 0과 1 사이에 있다는 것만으로 확률이 되는 것은 아니다. 이 글에서는 우선 각 숫자에 대한 모델의 반응값으로 이해한다.

![Pixel에서 숫자 예측까지](/assets/img/deep-learning-fundamentals/pixels-to-prediction.png){: style="max-width: 100%; height: auto;"}

**그림 설명.** 그림의 격자와 Neuron 수는 이해를 위해 단순화했다. 실제 Input은 28 × 28 Pixel이다. 왼쪽의 이미지를 Pixel별 밝기 값으로 펼친다. Network는 이 값들을 받아 숫자별 Output을 계산한다. 주황색 막대가 가장 큰 숫자 3이 예측 결과이다.

## 2. Hidden Layer와 패턴

Input과 Output 사이에는 Hidden Layer가 있다. 이 예시에는 Hidden Layer가 두 개 있다.

각 Layer는 이전 Layer의 Activation을 받아 계산한다. 앞쪽 Layer가 단순한 패턴을, 뒤쪽 Layer가 그 조합을 다룬다고 생각할 수 있다. 숫자 이미지에서는 짧은 획과 획의 조합을 예로 들 수 있다.

이는 구조를 이해하기 위한 직관이다. 개별 Neuron이 반드시 특정 획을 학습한다는 뜻은 아니다. 또한 Hidden Layer의 개수가 항상 두 개로 고정되는 것도 아니다.

## 3. Weight(가중치)와 Weighted Sum

Neuron은 이전 Layer의 Activation에 Weight(가중치)를 곱한다. 그 결과를 모두 더한다.

$$
w_1a_1+w_2a_2+w_3a_3+\cdots+w_na_n
$$

이 값이 Weighted Sum이다. 여기서 a는 이전 Neuron의 Activation이다. w는 각 연결의 Weight(가중치)이다.

Weight(가중치)는 각 Activation이 계산에 기여하는 정도를 조절한다. 양수인지 음수인지에 따라서도 기여 방향이 달라진다.

![하나의 Neuron이 계산하는 순서](/assets/img/deep-learning-fundamentals/neuron-computation.png){: style="max-width: 100%; height: auto;"}

**그림 설명.** 각 Input에 Weight를 곱한 뒤 더한다. Bias를 더하고 Activation Function을 통과시키면 Output이 나온다. 화살표를 왼쪽부터 따라가 보자.

### 3.1 Linear Combination

각 Input에 계수를 곱해 더한 것이 Linear Combination이다. 앞의 Weighted Sum이 여기에 해당한다. Bias까지 더한 계산은 엄밀하게는 Affine Transformation이다. Bias를 포함해 편의상 “선형 Layer”라고 부르기도 한다.

## 4. Sigmoid와 Bias

### 4.1 Sigmoid

Weighted Sum을 그대로 전달하지 않고 Activation Function에 넣는다. 먼저 살펴볼 함수는 Sigmoid이다.

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Sigmoid는 값을 0과 1 사이로 바꾼다. 큰 음수는 0에 가까워진다. 큰 양수는 1에 가까워진다. x가 0이면 결과는 0.5이다.

이때 x는 Sigmoid에 들어가는 값이다. 이미지 전체를 뜻하는 기호는 아니다.

### 4.2 Bias

Weighted Sum에서 10을 빼는 경우를 생각해 보자.

$$
w_1a_1+w_2a_2+w_3a_3+\cdots+w_na_n-10
$$

여기서 −10이 더해지는 Bias이다. Weighted Sum이 10이면 Sigmoid에 들어가는 값은 0이다. 따라서 Activation은 0.5가 된다.

Bias는 Neuron이 반응하는 기준을 옮긴다. Sigmoid는 부드럽게 변하므로 특정 기준에서 갑자기 켜지는 스위치는 아니다.

Bias는 더하는 값으로 정의할 수 있다. 아래 행렬식에서는 Bias를 더한다. 이 표기에서 위 예시의 Bias는 −10이다.

### 4.3 Perceptron Model과 Threshold Function

Perceptron은 Weighted Sum과 Bias로 두 Class를 구분하는 모델이다. Sigmoid 대신 Threshold Function을 사용하면 판단이 0 또는 1로 바로 결정된다.

**Weighted Sum + Bias가 0 이상 → 1 / 0 미만 → 0**

Threshold Function은 경계에서 불연속적이다. Sigmoid처럼 부드럽게 변하지 않는다. 따라서 이 함수를 그대로 사용해 일반적인 Backpropagation으로 학습하는 것은 적절하지 않다. 고전적인 Perceptron은 별도의 Perceptron Learning Rule을 사용한다.

### 4.4 Decision Boundary와 Perceptron Learning Rule

Decision Boundary는 예측 Class가 바뀌는 경계이다. Input이 두 개라면 단일 Perceptron의 경계는 직선이다. Input이 세 개라면 평면이다.

학습할 때 정답 y와 예측값 ŷ를 비교한다. 둘 다 0 또는 1로 표현하자.

| 결과 | y − ŷ | 업데이트 |
| --- | --- | --- |
| 정답과 예측이 같음 | 0 | 변경하지 않음 |
| 정답 1, 예측 0 | +1 | Input 방향으로 Weight를 더하고 Bias를 높임 |
| 정답 0, 예측 1 | −1 | Input 방향으로 Weight를 빼고 Bias를 낮춤 |

정확한 규칙은 간단하다. 각 Weight에는 **Learning Rate × (y − ŷ) × 해당 Input**을 더한다. Bias에는 **Learning Rate × (y − ŷ)**를 더한다. Input이 음수라면 Weight 변화의 부호도 달라진다.

예를 들어 Input이 (1, 0), 정답이 1, 예측이 0이고 Learning Rate가 0.1이라고 하자. 첫 Weight와 Bias에는 0.1을 더한다. 두 번째 Weight는 그대로 둔다. 이 Input을 다시 만났을 때 1로 판단하기 쉬워진다.

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

비선형 Activation Function을 제거하고 Affine Layer만 여러 개 쌓아 보자. 결과는 다시 하나의 Affine Transformation으로 합쳐진다. 깊어졌다는 이유만으로 XOR을 구분할 수는 없다.

Hidden Layer의 비선형 변환은 이 제한을 벗어나게 한다. XOR에서는 “적어도 하나가 1인가?”와 “둘 다 1인가?” 같은 중간 특징을 만든 뒤 조합하면 된다. 첫 조건이 참이고 두 번째 조건이 거짓일 때 XOR은 1이다.

이처럼 원래 Input을 구분하기 쉬운 새 특징으로 바꾸는 것이 Non-linear Representation의 역할이다. 실제 학습에서는 사람이 이런 규칙을 직접 정하기보다 데이터로 Weight를 조정한다.

### 5.3 Representation Learning

모델이 문제에 필요한 특징 자체를 학습하는 것을 Representation Learning이라고 한다. 손글씨 분류에서는 Pixel을 그대로 비교하는 대신 Hidden Layer에서 유용한 조합을 만들 수 있다.

특징 하나가 Neuron 하나에 명확히 대응할 필요는 없다. 여러 Neuron의 Activation이 함께 하나의 특징을 표현할 수 있다. 따라서 모든 Hidden Neuron에 “가로획 감지기”처럼 이름을 붙일 수 있는 것은 아니다.

### 5.4 Universal Approximation과 Network Depth

Universal Approximation은 MLP의 표현 가능성에 대한 결과이다. 대표적으로, Sigmoid Hidden Layer의 폭을 충분히 늘리고 선형 Output을 사용하면 유계의 닫힌 입력 영역에서 연속 함수를 원하는 정확도로 근사할 수 있다.

이 결과는 작은 Network가 모든 함수를 정확히 표현한다는 뜻이 아니다. 학습 알고리즘이 좋은 Weight를 찾는다는 보장도 아니다. 필요한 Neuron 수, 학습 데이터, Generalization은 별개의 문제이다.

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

학습 초기의 예측은 정답과 다를 수 있다. 이 차이를 숫자로 나타낸 것이 Cost(Loss)이다.

각 Output과 정답의 차이를 제곱한다. 그 값을 모두 더한다. 아래는 숫자 3을 분류할 때의 계산 일부이다.

$$
\begin{aligned}
&(0.43-0.00)^2+\cdots\\
&\quad +(0.88-1.00)^2+\cdots
\end{aligned}
$$

정답은 숫자 3이다. 숫자 3에 대응하는 목표값은 1이다. 나머지 숫자의 목표값은 0이다.

따라서 정답에 해당하는 Output은 1에 가까워야 한다. 나머지 Output은 0에 가까워야 한다. 예측과 목표의 차이가 커질수록 Cost가 커진다.

전체 Cost를 계산할 때는 생략된 항을 포함해 10개 Output의 제곱오차를 모두 더한다.

### 7.1 Classification에 맞는 Output과 Loss

제곱오차는 학습 원리를 설명하기에 간단하다. 실제 분류에서는 Output의 의미에 맞춰 Loss Function을 선택한다.

| 문제 | Output 구성 | 대표 Loss |
| --- | --- | --- |
| Binary Classification | Sigmoid 하나로 Class 1의 확률을 모델링 | Binary Cross-Entropy |
| Multi-class Classification | Class별 Logit을 Softmax로 변환 | Categorical Cross-Entropy |

Logit은 확률로 변환하기 전의 점수이다. 여기서는 마지막 Layer의 Weighted Sum과 Bias를 더한 값이다.

### 7.2 Binary Classification: Sigmoid

예를 들어 “고양이인가?”처럼 두 Class를 구분한다. Sigmoid Output이 0.8이면 모델이 Class 1에 부여한 확률은 0.8이다. Class 0에는 0.2가 대응한다. 실제 발생 빈도와 완벽히 일치하는 확률이라는 보장은 없다.

확률을 최종 Class로 바꾸려면 Threshold가 필요하다. 기본 예시로 0.5 이상이면 1로 분류할 수 있다. 놓치는 오류와 잘못 경고하는 오류의 비용에 따라 Threshold는 달라질 수 있다.

여기서 분류용 Threshold는 예측을 해석하는 단계이다. 학습 중에는 부드러운 Sigmoid Output으로 Loss와 Gradient를 계산한다.

### 7.3 Multi-class Classification: Softmax

숫자 0–9 중 하나를 고르는 문제에는 서로 배타적인 Class가 10개 있다. Softmax는 10개의 Logit을 비교해, 모든 성분이 양수이고 합이 1인 확률 분포로 바꾼다.

계산은 **각 Logit에 지수함수 적용 → 그 결과의 전체 합으로 나누기**이다. 예를 들어 Logit이 (0, 0, 0)이면 세 확률은 모두 1/3이다. (2, 1, 0)이면 약 (0.665, 0.245, 0.090)이 된다.

Sigmoid를 각 성분에 따로 적용하면 합이 1이라는 보장이 없다. Softmax는 Class 사이의 상대적인 점수를 함께 반영한다. 최종 예측은 보통 가장 큰 확률의 Class이다.

### 7.4 Cross-Entropy Loss

One-hot 정답을 사용하는 Multi-class Classification에서는 정답 Class에 부여한 확률 p만 보면 된다. 개별 example의 Cross-Entropy Loss는 **−log(p)**이다.

정답에 0.9를 부여하면 Loss는 약 0.105이다. 0.1만 부여하면 약 2.303이다. 자연로그 기준이다. 정답을 확신할수록 Loss가 작고, 틀린 Class를 강하게 확신할수록 Loss가 커진다.

Binary Cross-Entropy도 같은 원리이다. 정답이 1이면 Sigmoid의 p를, 정답이 0이면 1−p를 정답 확률로 사용한다. [Softmax와 Cross-Entropy 설명](https://d2l.ai/chapter_linear-classification/softmax-regression.html).

### 7.5 Training과 Inference

Training에서는 정답이 있는 데이터로 Forward Propagation, Loss 계산, Backpropagation, Parameter Update를 수행한다.

Inference에서는 학습한 Parameter로 새로운 Input의 Output을 계산한다. 정답을 입력하지 않아도 예측할 수 있다. 일반적인 Inference에서는 Gradient 계산과 Parameter Update를 하지 않는다. Input 정규화 등 전처리는 학습 때의 기준을 유지한다.

## 8. Neural Network와 Cost Function

같은 Network를 두 관점에서 살펴보자.

| 구분 | Input | Output |
| --- | --- | --- |
| Neural Network | 784개의 Pixel 값 | 10개의 숫자 |
| Cost Function | 13,002개의 Weight(가중치)와 Bias | Cost 하나 |

Neural Network는 현재 Weight(가중치)와 Bias로 예측을 계산한다. Cost Function은 그 설정이 얼마나 좋은지 평가한다. 이때 Training Dataset을 기준으로 예측과 정답을 비교한다.

13,002개는 이 예시 Network의 Parameter 수이다. 모든 Neural Network가 같은 수의 Parameter를 갖는 것은 아니다.

Training Dataset은 Parameter를 조정하는 데 사용한다. Test Dataset은 학습에 사용하지 않은 데이터에서 성능을 확인하는 데 사용한다.

### 8.1 Model Capacity, Overfitting, Generalization

Model Capacity는 모델이 얼마나 다양한 관계를 표현할 수 있는지를 뜻한다. Layer의 폭과 깊이, Activation Function 등이 영향을 준다. Parameter 수는 참고 지표이지만 Capacity를 완전히 설명하지는 않는다.

Capacity가 부족하면 Training Data의 규칙도 충분히 표현하지 못할 수 있다. 반대로 모델이 Training Data의 잡음이나 우연한 패턴까지 따라가면 Overfitting이 생길 수 있다.

Generalization은 학습하지 않은 데이터에서도 유용한 예측을 하는 능력이다. Training Loss가 줄어드는 동안 Validation Loss가 지속적으로 커진다면 Overfitting을 의심할 수 있다. 다만 데이터 분포 차이도 확인해야 한다.

| 데이터 구분 | 용도 |
| --- | --- |
| Training | Weight와 Bias 학습 |
| Validation | 모델 크기, Learning Rate, 학습 종료 시점 선택 |
| Test | 선택을 마친 모델의 최종 성능 평가 |

Test 결과를 반복해서 보고 설정을 고르면 Test도 간접적으로 학습에 사용한 셈이 된다. 더 많은 적절한 데이터, Regularization, Early Stopping은 Overfitting을 줄이는 데 도움이 될 수 있다. Early Stopping은 보통 Validation 성능을 기준으로 학습을 멈춘다.

## 9. Gradient와 경사하강법(Gradient Descent)

### 9.1 Cost를 줄이는 방향

Weight(가중치)와 Bias를 바꾸면 Cost도 변한다. 학습에서는 Cost가 작아지는 방향을 찾는다.

먼저 하나의 Weight에 대한 Cost 그래프를 생각해 보자. 현재 위치의 기울기를 보면 어느 방향으로 움직일지 판단할 수 있다.

여러 Parameter를 함께 다룰 때 사용하는 것이 Gradient이다. Gradient는 각 Parameter에 대한 편미분을 모은 값이다.

### 9.2 음의 Gradient

모든 Weight(가중치)와 Bias를 W vector로 묶어 표현하자. 이때 Cost를 줄이는 국소 방향은 다음과 같다.

$$
-\nabla C(\vec{W})
$$

여기서 W vector는 Bias까지 포함한다. 앞의 Layer 계산식에서 사용한 Weight 행렬 W와는 역할이 다르다.

Gradient는 Cost가 가장 빠르게 증가하는 국소 방향을 나타낸다. 음의 Gradient는 반대 방향이다. 이는 유클리드 거리 기준의 설명이다.

각 성분의 부호는 해당 Parameter를 늘릴지 줄일지 알려 준다. 크기는 현재 위치에서의 변화 민감도를 나타낸다.

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

Local Minimum은 주변보다 낮은 지점이다. Global Minimum은 전체에서 가장 낮은 지점이다.

경사하강법(Gradient Descent)이 항상 Global Minimum을 찾는 것은 아니다. 또한 Gradient가 0인 지점이 반드시 최솟값인 것도 아니다. 학습 과정에서는 Cost와 평가 성능을 함께 확인해야 한다.

![예측 오차와 Cost를 줄이는 과정](/assets/img/deep-learning-fundamentals/loss-and-gradient.png){: style="max-width: 100%; height: auto;"}

**그림 설명.** 왼쪽은 숫자 0–4의 Output만 표시한 예시이다. 청록색은 Prediction, 주황색은 Target이다. Target이 0인 막대는 높이가 없어 보이지 않는다. 두 막대의 차이가 작아지는 것이 목표이다. 오른쪽은 하나의 Weight만 바꾼 단순한 Cost 그래프이다. 점과 화살표는 Cost가 낮아지는 쪽으로 조금씩 이동하는 과정을 보여 준다.

## 10. 이번 학습의 연결

Neuron은 Weighted Sum과 Bias를 계산한다. Activation Function은 그 결과를 변환한다. 이 계산이 Layer를 따라 반복되면 Output이 나온다.

Cost(Loss)는 Output과 정답의 차이를 나타낸다. Gradient는 Parameter를 어느 방향으로 조정할지 알려 준다. 경사하강법(Gradient Descent)은 이 정보를 이용해 Cost를 줄여 나간다.

여러 Layer의 Gradient 계산, Vanishing Gradient, Mini-batch 학습 단위는 [Backpropagation과 Chain Rule](/blog/2026/backpropagation/)에서 이어서 다룬다.
