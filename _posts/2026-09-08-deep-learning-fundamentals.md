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

Neuron은 숫자 하나를 담는 계산 단위로 이해할 수 있다. 이 숫자가 Activation이다. Activation은 해당 Neuron의 반응 정도를 나타낸다.

여기서 다룰 예시는 손글씨 숫자 분류이다. Input 이미지는 28 × 28 Pixel로 구성된다. 따라서 Input에는 784개의 Neuron을 둔다. 각 Neuron은 Pixel 하나의 값을 전달한다.

Output에는 10개의 Neuron을 둔다. 각각 숫자 0부터 9에 대응한다. Output이 큰 숫자를 모델의 예측으로 선택한다.

단, Activation이 0과 1 사이에 있다는 것만으로 확률이 되는 것은 아니다. 이 글에서는 우선 각 숫자에 대한 모델의 반응값으로 이해한다.

![Pixel에서 숫자 예측까지](/assets/img/deep-learning-fundamentals/pixels-to-prediction.png)

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

![하나의 Neuron이 계산하는 순서](/assets/img/deep-learning-fundamentals/neuron-computation.png)

**그림 설명.** 각 Input에 Weight를 곱한 뒤 더한다. Bias를 더하고 Activation Function을 통과시키면 Output이 나온다. 화살표를 왼쪽부터 따라가 보자.

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

## 6. ReLU

Activation Function에는 ReLU도 있다. ReLU는 Rectified Linear Unit의 약자이다.

$$
\operatorname{ReLU}(a)=\max(0,a)
$$

a가 음수이면 0을 출력한다. 양수이면 a를 그대로 출력한다.

따라서 ReLU의 Activation은 1보다 클 수 있다. 모든 Activation이 0과 1 사이에 있는 것은 아니다. 범위는 사용하는 Activation Function에 따라 달라진다.

![Layer 사이의 흐름과 Activation Function](/assets/img/deep-learning-fundamentals/layers-and-activation.png)

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

## 8. Neural Network와 Cost Function

같은 Network를 두 관점에서 살펴보자.

| 구분 | Input | Output |
| --- | --- | --- |
| Neural Network | 784개의 Pixel 값 | 10개의 숫자 |
| Cost Function | 13,002개의 Weight(가중치)와 Bias | Cost 하나 |

Neural Network는 현재 Weight(가중치)와 Bias로 예측을 계산한다. Cost Function은 그 설정이 얼마나 좋은지 평가한다. 이때 Training Dataset을 기준으로 예측과 정답을 비교한다.

13,002개는 이 예시 Network의 Parameter 수이다. 모든 Neural Network가 같은 수의 Parameter를 갖는 것은 아니다.

Training Dataset은 Parameter를 조정하는 데 사용한다. Test Dataset은 학습에 사용하지 않은 데이터에서 성능을 확인하는 데 사용한다.

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

![예측 오차와 Cost를 줄이는 과정](/assets/img/deep-learning-fundamentals/loss-and-gradient.png)

**그림 설명.** 왼쪽은 숫자 0–4의 Output만 표시한 예시이다. 청록색은 Prediction, 주황색은 Target이다. Target이 0인 막대는 높이가 없어 보이지 않는다. 두 막대의 차이가 작아지는 것이 목표이다. 오른쪽은 하나의 Weight만 바꾼 단순한 Cost 그래프이다. 점과 화살표는 Cost가 낮아지는 쪽으로 조금씩 이동하는 과정을 보여 준다.

## 10. 이번 학습의 연결

Neuron은 Weighted Sum과 Bias를 계산한다. Activation Function은 그 결과를 변환한다. 이 계산이 Layer를 따라 반복되면 Output이 나온다.

Cost(Loss)는 Output과 정답의 차이를 나타낸다. Gradient는 Parameter를 어느 방향으로 조정할지 알려 준다. 경사하강법(Gradient Descent)은 이 정보를 이용해 Cost를 줄여 나간다.

다음 학습 주제는 역전파이다. 여러 Layer에 걸친 Gradient를 어떻게 계산하는지 살펴볼 예정이다.
