---
layout: post
title: "Chapter 01. Multilayer Perceptron (MLP)"
date: 2026-09-08 00:00:00
description: "이번 학습에서는 Multilayer Perceptron의 구조와 학습 원리를 이론적·수학적 관점에서 살펴보고, 주요 개념과 동작 과정을 자신의 언어로 설명할 수 있는 수준까지 이해하는 것을 목표로 합니다. 특히 Neural Network가 입력으로부터 출력을 계산하는 과정과, 예측 오차를 이용하여 Weight와 Bias를 업데이트하는 과정을 수식 수준에서 이해하는 것을 목표로 합니다."

tags: [ai, deep-learning, study]
categories: ["AI/DeepLearning"]
---


## Topics

### 1.1 Fundamentals of Neural Networks

- Artificial Neuron
- Input, Weight, Bias
- Weighted Sum
- Activation Function
- Layer
- Forward Propagation

### 1.2 Perceptron

- Perceptron Model
- Linear Combination
- Threshold Function
- Decision Boundary
- Perceptron Learning Rule
- Limitations of a Single-Layer Perceptron
- XOR Problem

### 1.3 Multilayer Perceptron (MLP)

- Input Layer
- Hidden Layer
- Output Layer
- Weight and Bias
- Activation Functions
- Non-linearity
- Forward Propagation

### 1.4 Backpropagation Algorithm

- Loss Function
- Gradient
- Partial Derivative
- Chain Rule
- Computational Graph
- Gradient Calculation
- Backpropagation
- Weight and Bias Update

Backpropagation은 단순히 공식을 암기하는 것이 아니라 다음 과정이 왜 필요한지 설명할 수 있도록 학습합니다.

**Forward Propagation → Loss Calculation → Gradient Calculation → Backpropagation → Parameter Update**

특히 Chain Rule을 이용하여 간단한 Neural Network에서 다음과 같은 편미분값이 어떻게 계산되는지 직접 전개해 봅니다.

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial z}
\frac{\partial z}{\partial w}
$$

### 1.5 Mini-batch Stochastic Gradient Descent

- Gradient Descent
- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-batch SGD
- Learning Rate
- Epoch
- Batch Size
- Iteration
- Parameter Update

Batch, Stochastic, Mini-batch Gradient Descent의 차이와 각각의 장단점을 설명할 수 있도록 합니다.

### 1.6 Classification Using Multilayer Perceptrons

- Input Data
- Forward Propagation
- Output Prediction
- Binary Classification
- Multi-class Classification
- Sigmoid
- Softmax
- Cross-Entropy Loss
- Training and Inference

### 1.7 Characteristics of Multilayer Perceptrons

- Non-linear Representation
- Representation Learning
- Universal Approximation
- Model Capacity
- Overfitting
- Generalization
- Vanishing Gradient
- Effect of Network Depth

## Study Notes

<!-- 각 주제에 대해 공부한 내용을 여기에 계속 추가합니다. -->

## 코드 실습

```python
# 실습 코드를 여기에 추가합니다.
```

## 헷갈린 점

<!-- 공부하면서 생긴 질문과 추가로 확인할 내용을 기록합니다. -->

## 다음에 공부할 내용

- 

## 참고 자료

- 
