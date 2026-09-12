---
layout: post
title: "Deep Learning 기초: Backpropagation과 Chain Rule"
date: 2026-09-12 09:00:00 +0900
description: "Output의 오차가 앞쪽 Layer에 전달되는 과정을 살펴보고, Chain Rule로 Weight와 Bias의 Gradient를 계산한다."
tags: [ai, deep-learning, backpropagation, study]
categories: ["AI/DeepLearning"]
math: true
---

<style>
article h2 { margin-top: 3.5rem; }
article h3 { margin-top: 2.5rem; }
article img { max-width: 100%; height: auto; }
</style>

## 초록

Neural Network의 학습에는 두 과정이 필요하다. 먼저 각 Weight와 Bias가 Cost에 미치는 영향을 계산한다. 다음으로 Cost가 줄어드는 방향으로 값을 조정한다.

Backpropagation(역전파)은 첫 번째 과정을 담당한다. Output에서 시작해 앞쪽 Layer로 이동하며 Gradient를 계산한다. Gradient Descent(경사하강법)는 이 Gradient를 사용해 Weight와 Bias를 업데이트한다.

이 글에서는 먼저 직관을 살펴본다. 이어서 한 Neuron의 미분을 직접 전개한다. 마지막으로 여러 Neuron과 Mini-batch로 설명을 확장한다.

![Forward와 Backward의 방향](/assets/img/backpropagation/01-forward-backward.png){: style="max-width: 100%; height: auto;"}

**그림 읽기.** 위쪽에서는 Activation과 Cost를 계산한다. 아래쪽에서는 Cost에서 출발해 Gradient를 계산한다. Input 자체는 학습 Parameter로 업데이트하지 않는다.

## 1. Output을 바꾸려면 무엇을 조정해야 할까?

손글씨 숫자 2를 입력했다고 하자. 숫자 2에 해당하는 Output Activation이 0.2라면, 목표값 1보다 작다. 이 Output을 높이면 해당 항의 제곱오차가 줄어든다.

Sigmoid를 사용하는 Neuron에서는 세 가지 영향을 생각할 수 있다.

- **Bias:** Bias를 높이면 Sigmoid에 들어가는 값이 커진다.
- **Weight:** 이전 Activation이 양수라면 연결된 Weight를 높여 Output을 높일 수 있다.
- **이전 Layer의 Activation:** 연결된 Weight의 부호에 따라 필요한 변화 방향이 달라진다.

예를 들어 양의 Weight로 연결되었다면 이전 Activation이 커질수록 현재 Output도 커진다. 음의 Weight로 연결되었다면 반대이다.

다만 Hidden Layer의 Activation은 직접 저장해 업데이트하는 Parameter가 아니다. 앞쪽 Layer의 Weight와 Bias가 바뀌면서 결과적으로 달라진다. 따라서 “이 Activation이 어떻게 바뀌면 좋을까?”라는 정보를 앞쪽으로 전달해야 한다.

이것이 Backpropagation의 출발점이다.

![연결 부호에 따른 Activation 변화](/assets/img/backpropagation/02-signs.svg){: style="max-width: 100%; height: auto;"}

**그림 읽기.** Output을 높이고 싶어도 앞쪽 Activation을 항상 높이는 것은 아니다. 양의 연결에서는 높이고, 음의 연결에서는 낮추는 방향이 유리하다.

### 큰 Activation과 연결된 Weight의 영향

이전 Activation이 클수록 같은 Weight 변화가 Weighted Sum에 더 큰 변화를 만든다. 따라서 그 연결의 Gradient에도 이전 Activation이 곱해진다.

하지만 이전 Activation만으로 전체 영향이 결정되지는 않는다. Activation Function의 기울기와 현재 예측 오차도 함께 작용한다. 이 세 요소는 뒤에서 하나의 식으로 연결된다.

## 2. 여러 Output의 요구를 함께 모으기

숫자 2를 맞히려면 숫자 2의 Output은 높이고 싶다. 나머지 숫자의 Output은 낮추고 싶다. 그런데 하나의 Hidden Neuron은 여러 Output Neuron과 연결되어 있다.

같은 Hidden Neuron을 두고 서로 다른 Output이 다른 변화를 요구할 수 있다. 어떤 연결에서는 Activation을 높이는 편이 유리하다. 다른 연결에서는 낮추는 편이 유리하다.

Backpropagation은 이 영향을 부호와 크기를 포함해 더한다. 단순히 다수결을 하는 것은 아니다. Cost에 더 민감한 경로는 더 크게 기여한다.

이렇게 구한 정보를 한 Layer 앞쪽으로 전달한다. 앞쪽 Layer에서도 같은 계산을 반복한다. 계산 방향은 Output에서 Input 쪽이지만, Forward Propagation에서 구한 값들이 필요하다.

![여러 Output 기여의 합](/assets/img/backpropagation/10-contributions.svg){: style="max-width: 100%; height: auto;"}

**그림 읽기.** 한 Hidden Activation에 대한 두 경로의 기여가 +0.6과 −0.2라면 합은 +0.4이다. 이 숫자는 합산 원리를 위한 예시이다. 실제 Activation 변화는 앞쪽 Parameter의 업데이트로 간접적으로 일어난다.

## 3. 한 Neuron의 계산부터 시작하기

먼저 마지막 Layer에 Neuron 하나만 있는 간단한 경우를 생각하자.

| 기호 | 의미 |
| --- | --- |
| L | 마지막 Layer의 번호 |
| a⁽ᴸ⁻¹⁾ | 바로 이전 Layer의 Activation |
| w⁽ᴸ⁾ | 마지막 연결의 Weight |
| b⁽ᴸ⁾ | 마지막 Neuron의 Bias |
| z⁽ᴸ⁾ | Activation Function을 적용하기 전의 값 |
| a⁽ᴸ⁾ | 마지막 Neuron의 Output Activation |
| y | 목표 Output |
| C₀ | Training example 하나에 대한 Cost |

괄호가 있는 위첨자 (L)은 거듭제곱이 아니다. 어느 Layer의 값인지 나타낸다. C₀의 아래첨자 0도 Layer 번호가 아니다. 여기서는 첫 번째 Training example을 구분한다.

![한 Neuron의 Forward 계산](/assets/img/backpropagation/03-forward-path.svg){: style="max-width: 100%; height: auto;"}

**그림 읽기.** a⁽ᴸ⁻¹⁾에서 출발해 z, a⁽ᴸ⁾, C₀ 순서로 읽는다. 목표값 y는 Cost를 계산할 때 비교 대상으로 들어간다. 이 그림의 수치를 아래 수식과 대응해 보자.

### 3.1 Forward 계산

먼저 Weight를 곱하고 Bias를 더한다.

$$
z^{(L)}=w^{(L)}a^{(L-1)}+b^{(L)}
$$

그다음 Sigmoid를 적용한다.

$$
a^{(L)}=\sigma\left(z^{(L)}\right)
$$

Weighted Sum과 Bias의 합은 z이다. Sigmoid까지 적용한 결과가 a이다. 두 값을 구분해야 미분의 경로가 보인다.

### 3.2 Cost 계산

예측과 목표의 차이를 제곱한다.

$$
C_0=\left(a^{(L)}-y\right)^2
$$

예를 들어 Output이 0.66이고 목표가 1이면 다음과 같다.

$$
(0.66-1.00)^2
$$

Cost는 Output이 목표에서 얼마나 벗어났는지 나타낸다. 이 값만으로는 어떤 Weight를 얼마나 바꿀지 알 수 없다. 그 정보를 얻으려면 미분이 필요하다.

### 3.3 Computational Graph

앞의 “Activation → z → Output → Cost” 도식은 Computational Graph이다. 계산 결과를 Node로, 값의 의존 관계를 화살표로 표현한다. Weight와 Bias도 계산에 들어오는 Node이다.

Forward에서는 화살표 방향으로 값을 구한다. Backward에서는 각 연산의 국소 미분을 이용해 Cost의 영향을 거꾸로 전달한다. 한 Node에서 여러 경로로 갈라지면 돌아오는 Gradient를 더한다. 중간 결과를 저장하고 재사용하므로 같은 미분을 반복 계산할 필요가 줄어든다.

## 4. Chain Rule(연쇄법칙): 작은 변화의 경로를 따라가기

우리가 구하려는 값은 다음과 같다.

$$
\frac{\partial C_0}{\partial w^{(L)}}
$$

이 값은 Partial Derivative(편미분)이다. 여러 변수 중 하나를 선택해 변화율을 구한다. Weight가 조금 변할 때 Cost가 얼마나 변하는지 나타낸다. 다른 독립적인 Parameter와 Input은 고정한다.

Weight는 Cost를 곧바로 바꾸지 않는다. 중간 계산을 거친다.

**Weight 변화 → z 변화 → Activation 변화 → Cost 변화**

Chain Rule(연쇄법칙)은 이 경로의 변화율을 곱한다.

$$
\frac{\partial C_0}{\partial w^{(L)}}
=
\frac{\partial z^{(L)}}{\partial w^{(L)}}
\frac{\partial a^{(L)}}{\partial z^{(L)}}
\frac{\partial C_0}{\partial a^{(L)}}
$$

왼쪽은 최종적으로 알고 싶은 변화율이다. 오른쪽은 이를 세 구간으로 나눈 것이다. 분수를 단순히 약분하는 규칙이 아니라, 합성된 함수의 미분 규칙이다.

![Chain Rule의 세 구간](/assets/img/backpropagation/04-chain-rule.svg){: style="max-width: 100%; height: auto;"}

**그림 읽기.** 화살표 하나마다 변화율이 하나씩 있다. 세 변화율을 곱하면 Weight가 Cost에 미치는 영향을 얻는다.

## 5. 세 구간을 직접 미분하기

### 5.1 Activation이 Cost에 미치는 영향

제곱오차를 Activation으로 미분한다.

$$
\frac{\partial C_0}{\partial a^{(L)}}
=2\left(a^{(L)}-y\right)
$$

Output이 목표보다 작으면 이 값은 음수이다. Output을 조금 높이는 방향이 해당 Cost를 줄인다. Output이 목표보다 크면 반대이다.

### 5.2 z가 Activation에 미치는 영향

Activation은 z에 Sigmoid를 적용한 결과이다.

$$
\frac{\partial a^{(L)}}{\partial z^{(L)}}
=\sigma'\left(z^{(L)}\right)
$$

프라임 기호는 미분을 뜻한다. 이 값은 Sigmoid가 현재 위치에서 얼마나 가파른지 나타낸다.

Sigmoid가 0이나 1 근처에서 평평해지면 이 기울기는 작아진다. 따라서 Output의 오차가 있어도 앞쪽으로 전달되는 Gradient가 작을 수 있다. “오차가 크면 Weight도 항상 크게 바뀐다”는 해석은 맞지 않는다.

### 5.2.1 Vanishing Gradient와 Depth

Sigmoid의 평평한 구간에서 기울기가 작아지는 현상은 앞쪽 Layer의 학습에도 영향을 준다. Backpropagation은 Layer를 거치며 국소 미분과 Weight를 곱한다. 이 곱이 반복해서 작아지면 앞쪽 Parameter에 도달하는 Gradient도 매우 작아진다. 이것이 Vanishing Gradient이다.

작은 Gradient는 작은 업데이트로 이어진다. 그 결과 앞쪽 Layer가 학습되는 속도가 느려질 수 있다. 반대로 반복되는 곱이 너무 커지는 경우는 Exploding Gradient라고 한다. 깊이만으로 어느 현상이 발생하는지 결정되지는 않는다. Weight와 Activation Function의 조합도 중요하다.

ReLU는 양수 구간의 미분이 1이므로 Sigmoid의 포화에 따른 축소를 완화할 수 있다. 하지만 음수 구간에서는 미분이 0이다. ReLU를 사용한다고 모든 Gradient 문제가 사라지는 것은 아니다. 적절한 초기화, Normalization, Residual Connection 등도 깊은 모델의 학습에 사용된다. [Gradient의 소실과 폭주](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html).

### 5.3 Weight가 z에 미치는 영향

z의 식에서 Weight에 곱해진 값은 이전 Activation이다.

$$
\frac{\partial z^{(L)}}{\partial w^{(L)}}
=a^{(L-1)}
$$

따라서 이전 Activation이 클수록 해당 연결의 기여가 커진다. 다만 나머지 두 변화율도 함께 곱해야 최종 Gradient가 된다.

### 5.4 세 결과를 연결하기

세 구간을 곱하면 다음과 같다.

$$
\frac{\partial C_0}{\partial w^{(L)}}
=a^{(L-1)}\sigma'\left(z^{(L)}\right)
\,2\left(a^{(L)}-y\right)
$$

각 항은 서로 다른 질문에 답한다.

| 항 | 질문 |
| --- | --- |
| 이전 Activation | 이 Weight가 받는 Input은 얼마나 큰가? |
| Sigmoid의 미분 | 현재 Neuron은 Input 변화에 얼마나 민감한가? |
| 제곱오차의 미분 | Output은 어느 방향으로 바뀌어야 하는가? |

Gradient가 양수이면 Gradient Descent는 해당 Weight를 줄이는 방향으로 움직인다. 음수이면 늘리는 방향으로 움직인다. 실제 변화량에는 Learning Rate도 반영된다.

![Gradient의 세 요소](/assets/img/backpropagation/05-three-factors.svg){: style="max-width: 100%; height: auto;"}

**그림 읽기.** 앞의 미분 결과를 세 카드에 대응해 보자. Input이 커도 Sigmoid의 기울기가 작으면 Gradient가 작아질 수 있다.

## 6. Bias와 이전 Activation에도 같은 원리가 적용된다

### 6.1 Bias에 대한 미분

Bias는 z에 그대로 더해진다. 따라서 Bias에 대한 z의 미분은 1이다. 이후의 경로는 Weight를 미분할 때와 같다.

즉, Weight Gradient에 있던 “이전 Activation” 대신 1이 곱해진다. 별도의 원리를 새로 배울 필요는 없다. Chain Rule에서 첫 구간만 달라진다.

### 6.2 이전 Activation에 대한 미분

이전 Activation은 z에서 Weight와 곱해진다. 따라서 이전 Activation에 대한 z의 미분은 Weight이다.

이 Weight의 부호가 앞쪽으로 전달할 방향에 영향을 준다. 양의 연결과 음의 연결에서 필요한 Activation 변화가 다른 이유이다.

이전 Activation의 Gradient를 구했다면, 그 Activation을 만든 앞쪽 Weight와 Bias로 계산을 이어 갈 수 있다. Backpropagation은 이렇게 이미 구한 결과를 재사용한다.

![미분 대상에 따른 첫 구간의 차이](/assets/img/backpropagation/06-bias-input.svg){: style="max-width: 100%; height: auto;"}

**그림 읽기.** 세 줄 모두 오른쪽 두 항은 같다. 첫 항만 Weight, Bias, 이전 Activation 중 무엇을 미분하는지에 따라 달라진다.

### 6.3 Learning Rate와 Weight and Bias Update

Gradient는 변화의 방향과 민감도를 알려 준다. Learning Rate는 한 번에 움직이는 보폭이다. 기본 Gradient Descent 규칙은 **새 Parameter = 현재 Parameter − Learning Rate × 해당 Gradient**이다. Weight와 Bias에 각각 적용한다.

예를 들어 현재 Weight가 0.5, Gradient가 +0.2, Learning Rate가 0.1이면 새 Weight는 0.48이다. Bias가 0.1이고 Gradient가 −0.3이면 새 Bias는 0.13이다. Gradient의 부호가 다르므로 움직이는 방향도 다르다.

기본 방식에서는 같은 현재 Parameter에서 Gradient를 모두 계산한 뒤 업데이트한다. Backward 도중 Weight를 먼저 바꾸면 뒤의 계산이 다른 Network를 기준으로 수행될 수 있다.

Learning Rate가 너무 크면 좋은 영역을 지나치거나 학습이 불안정해질 수 있다. 너무 작으면 학습이 느리다. Learning Rate는 학습되는 Weight가 아니라 학습 절차를 정하는 Hyperparameter이다.

## 7. Training example 여러 개의 Gradient

하나의 Training example에 맞는 변화가 모든 데이터에 좋은 것은 아니다. 숫자 2를 잘 맞히는 방향과 숫자 5를 잘 맞히는 방향이 일부 충돌할 수 있다.

전체 Cost를 각 example의 Cost 평균으로 정의하면, Gradient도 각 example의 Gradient 평균이 된다.

$$
\frac{\partial C}{\partial w^{(L)}}
=\frac{1}{n}\sum_{k=0}^{n-1}
\frac{\partial C_k}{\partial w^{(L)}}
$$

n은 Training example의 수이다. 여기서 k는 example을 구분하는 번호이다. 모든 Gradient는 같은 현재 Parameter에서 계산한다.

**평균을 내는 대상은 최종 Weight가 아니다. 각 example이 만드는 Gradient이다.** 서로 다른 모델을 학습한 뒤 Weight를 평균 내는 과정과는 다르다.

모든 Weight와 Bias의 편미분을 모으면 전체 Gradient가 된다.

$$
\nabla C=
\begin{bmatrix}
\dfrac{\partial C}{\partial w^{(1)}}\\
\dfrac{\partial C}{\partial b^{(1)}}\\
\vdots\\
\dfrac{\partial C}{\partial w^{(L)}}\\
\dfrac{\partial C}{\partial b^{(L)}}
\end{bmatrix}
$$

이 표기는 Layer마다 Neuron 하나를 둔 간단한 경우이다. Neuron이 많아지면 각 Layer의 여러 Weight와 Bias에 대한 성분이 추가된다.

Backpropagation은 이 Gradient를 계산한다. Gradient Descent는 음의 Gradient 방향으로 Parameter를 업데이트한다. 두 알고리즘이 각각 다른 “최종 Weight”를 구하는 것은 아니다.

![여러 example의 Gradient 평균](/assets/img/backpropagation/07-mean.svg){: style="max-width: 100%; height: auto;"}

**그림 읽기.** 같은 Parameter에서 구한 Gradient가 +0.6, −0.2, −0.1이면 평균은 +0.1이다. 양의 Gradient이므로 해당 Weight는 줄이는 방향으로 업데이트한다.

## 8. Mini-batch와 확률적 경사하강법

매번 전체 Training Dataset의 Gradient를 계산하면 업데이트 한 번의 비용이 커진다. 이를 줄이기 위해 데이터를 작은 Mini-batch로 나눈다.

진행 순서는 다음과 같다.

1. Training Dataset의 순서를 무작위로 섞는다.
2. 하나의 Mini-batch를 선택한다.
3. 현재 Parameter로 각 example의 Forward 계산을 수행한다.
4. Backpropagation으로 Gradient를 계산하고 평균을 구한다.
5. 그 Gradient로 Parameter를 한 번 업데이트한다.
6. 다음 Mini-batch에서 새 Parameter로 계산한다.

| 방법 | 한 번의 업데이트에 사용하는 데이터 |
| --- | --- |
| Batch Gradient Descent | 전체 Training Dataset |
| Stochastic Gradient Descent | 엄밀하게는 example 하나 |
| Mini-batch Gradient Descent | 여러 example으로 구성된 작은 묶음 |

실제로는 Mini-batch 방식도 SGD라고 부르는 경우가 많다. 그래서 SGD라는 이름을 볼 때는 Batch Size도 함께 확인하는 것이 좋다.

Mini-batch Gradient는 전체 Gradient의 추정값이다. 따라서 매번 정확히 같은 방향을 가리키지는 않는다. 한 번의 업데이트 후 전체 Training Cost가 반드시 감소하는 것도 아니다. 대신 전체 데이터를 모두 계산하기 전에 여러 번 업데이트할 수 있다.

Mini-batch는 데이터를 나눠 업데이트하는 방식이다. Backpropagation은 Gradient를 계산하는 방식이다. 두 개념을 구분하면 학습 흐름이 명확해진다.

![Mini-batch 업데이트 순서](/assets/img/backpropagation/08-minibatch.svg){: style="max-width: 100%; height: auto;"}

**그림 읽기.** 청록색 네 장으로 Gradient를 구하고 평균 낸다. Parameter를 한 번 바꾼 뒤 다음 묶음을 처리한다. 모든 묶음을 처음부터 같은 Parameter로 계산해 두는 방식은 아니다.

### 8.1 Batch, Stochastic, Mini-batch의 장단점

| 방법 | 장점 | 부담과 주의점 |
| --- | --- | --- |
| Batch Gradient Descent | 전체 Training Cost의 정확한 Gradient 사용 | 한 번 업데이트하기까지 전체 데이터 계산 필요 |
| SGD: example 하나 | 빠르게 업데이트 가능, 한 번에 필요한 메모리가 작음 | Gradient의 변동이 크고 GPU 병렬 처리 효율이 낮을 수 있음 |
| Mini-batch SGD | 병렬 처리와 업데이트 빈도의 절충 | Batch Size 선택에 따라 메모리와 Gradient 변동이 달라짐 |

무작위 추출을 적절히 하면 Mini-batch 평균은 전체 Gradient의 추정값이 된다. Batch Size를 키우면 보통 추정의 변동은 줄지만, 업데이트 하나의 연산량과 메모리가 늘어난다. 같은 Epoch 수에서는 업데이트 횟수도 줄어든다. 큰 Batch가 항상 더 빠르거나 Generalization에 더 좋은 것은 아니다. [Mini-batch SGD의 계산 특성](https://d2l.ai/chapter_optimization/minibatch-sgd.html).

### 8.2 Epoch, Batch Size, Iteration

| 용어 | 이 글에서의 의미 |
| --- | --- |
| Batch Size | 한 Mini-batch에 들어 있는 example 수 |
| Iteration | 한 Mini-batch로 Forward, Backward, 업데이트를 한 번 수행 |
| Epoch | Training Dataset 전체를 한 번 순회 |

Training example이 1,000개이고 Batch Size가 100이면, 10 Iterations가 1 Epoch이다. 5 Epochs 학습하면 총 50회 업데이트한다. Batch Size를 200으로 늘리면 1 Epoch의 업데이트는 5회가 된다.

1,030개를 Batch Size 100으로 처리하면 마지막 묶음은 30개이다. 마지막 묶음까지 사용하면 11 Iterations이다. 마지막 묶음을 버리는 설정에서는 10회이지만 해당 Epoch에서 30개는 사용하지 않는다. 이 설명은 Gradient Accumulation 없이 Mini-batch마다 업데이트하는 기본 방식이다.

![Epoch와 Mini-batch 업데이트 단위](/assets/img/dl-concepts/epoch.svg){: style="max-width: 100%; height: auto;"}

**그림 읽기.** 한 줄 전체가 1 Epoch이다. 작은 묶음마다 Gradient를 구하고 Parameter를 한 번 바꾼다. 다음 Epoch에서는 다시 데이터 전체를 순회한다.

## 9. 한 Layer에 Neuron이 여러 개라면?

이제 마지막 Layer에 여러 Output Neuron이 있다고 하자. 하나의 example에 대한 Cost는 각 Output의 제곱오차를 더한 값이다.

$$
C_0=\sum_{j=0}^{n_L-1}\left(a_j^{(L)}-y_j\right)^2
$$

n_L은 마지막 Layer의 Neuron 수이다. j는 마지막 Layer의 Neuron 번호이다. 각 Output은 자신에게 대응하는 목표값과 비교한다.

각 Neuron의 계산 원리는 앞과 같다. 달라지는 것은 연결을 구분하는 아래첨자이다.

- j: 현재 Layer의 Neuron 번호이다.
- k: 이전 Layer의 Neuron 번호이다.
- w의 아래첨자 jk: 이전 Neuron k에서 현재 Neuron j로 향하는 연결이다.

이 절의 k는 Neuron 번호이다. 앞 절의 데이터 평균식에서 사용한 example 번호와는 문맥이 다르다.

### 9.1 하나의 연결에 대한 Gradient

하나의 Weight가 Cost에 미치는 경로는 여전히 세 구간이다.

$$
\frac{\partial C_0}{\partial w_{jk}^{(L)}}
=
\frac{\partial z_j^{(L)}}{\partial w_{jk}^{(L)}}
\frac{\partial a_j^{(L)}}{\partial z_j^{(L)}}
\frac{\partial C_0}{\partial a_j^{(L)}}
$$

연결 번호가 추가되었지만 원리는 같다. Weight가 z를 바꾸고, z가 Activation을 바꾸고, Activation이 Cost를 바꾼다.

![한 Hidden Activation의 여러 경로](/assets/img/backpropagation/09-branches.svg){: style="max-width: 100%; height: auto;"}

**그림 읽기.** Hidden Activation에서 Cost까지 두 경로가 있다. 각 경로의 미분을 곱한 뒤 두 기여를 더한다. 아래 합 기호가 바로 이 과정을 나타낸다.

### 9.2 이전 Activation에는 여러 경로가 연결된다

이전 Layer의 Neuron 하나는 여러 Output Neuron에 연결된다. 따라서 그 Activation이 Cost에 미치는 영향도 여러 경로를 거친다.

각 경로에서는 변화율을 곱한다. 여러 경로의 영향은 더한다.

$$
\frac{\partial C_0}{\partial a_k^{(L-1)}}
=
\sum_{j=0}^{n_L-1}
\frac{\partial z_j^{(L)}}{\partial a_k^{(L-1)}}
\frac{\partial a_j^{(L)}}{\partial z_j^{(L)}}
\frac{\partial C_0}{\partial a_j^{(L)}}
$$

합을 구하는 대상은 바로 다음 Layer의 Neuron들이다. Network의 모든 Layer를 한 번에 합하는 식은 아니다.

서로 반대 부호를 가진 경로는 일부 상쇄될 수 있다. 같은 방향의 경로는 더해진다. 이렇게 모인 결과가 이전 Layer로 전달된다.

![Forward, Backpropagation, Gradient Descent의 역할](/assets/img/backpropagation/11-summary.svg){: style="max-width: 100%; height: auto;"}

**그림 읽기.** 왼쪽은 값을 계산하고, 가운데는 변화율을 계산한다. 실제로 Weight와 Bias를 바꾸는 단계는 오른쪽이다.

## 10. Backpropagation을 이해하는 세 가지 기준

**첫째, Forward 값을 먼저 계산한다.** Backpropagation에는 각 Layer의 Activation과 z가 필요하다. Output을 계산한 뒤 그 계산 경로를 거꾸로 따라간다.

**둘째, 한 경로의 변화율은 곱한다.** 이것이 Chain Rule이다. Weight에서 Cost까지 이어지는 중간 계산을 빠뜨리지 않는다.

**셋째, 여러 경로의 영향은 더한다.** 하나의 Activation이 여러 Neuron에 영향을 준다면 각 경로의 기여를 합한다. 여러 example을 평균 내는 경우에도 이 선형성이 사용된다.

학습의 흐름은 다음과 같다.

**Forward Propagation → Loss Calculation → Backpropagation으로 Gradient Calculation → Parameter Update**

Gradient Calculation과 Backpropagation을 별도의 연속 단계로 생각하지 않는다. Backpropagation이 Gradient를 계산하는 알고리즘이다. Classification에서 Sigmoid·Softmax·Cross-Entropy를 선택하는 이유와 MLP의 표현력은 [Neuron에서 경사하강법까지](/blog/2026/deep-learning-fundamentals/)에서 확인할 수 있다.

Backpropagation의 결과는 “어떤 Parameter를 어느 방향으로 움직이면 좋은가”에 대한 국소적인 정보이다. 최종 정답 Weight를 한 번에 찾아 주는 과정은 아니다. 이 계산과 업데이트를 반복하며 모델을 학습한다.
