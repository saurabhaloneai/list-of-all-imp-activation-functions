# activation functions and Their Derivatives

1. **Sigmoid**
2. **Tanh**
3. **ReLU**
4. **Leaky ReLU**
5. **Parametric ReLU**
6. **GELU**
7. **SiLU**
8. **Softmax**
9. **ELU**
10. **SELU**
11. **Softplus**
12. **Mish**


---

## 1. Sigmoid Function

The Sigmoid function is defined as:

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

Its derivative is:

\[
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
\]

### Graph:
![Sigmoid Function and Derivative](path/to/sigmoid_graph.png)

---

## 2. Tanh Function

The Tanh (hyperbolic tangent) function is defined as:

\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]

Its derivative is:

\[
\tanh'(x) = 1 - \tanh^2(x)
\]

### Graph:
![Tanh Function and Derivative](path/to/tanh_graph.png)

---

## 3. ReLU Function

The ReLU (Rectified Linear Unit) function is defined as:

\[
\text{ReLU}(x) = \max(0, x)
\]

Its derivative is:

\[
\text{ReLU}'(x) = \begin{cases} 
1 & x > 0 \\
0 & x \leq 0 
\end{cases}
\]

### Graph:
![ReLU Function and Derivative](path/to/relu_graph.png)

---

## 4. Leaky ReLU Function

Leaky ReLU introduces a small slope for negative values:

\[
\text{Leaky ReLU}(x) = \begin{cases} 
x & x > 0 \\
\alpha \cdot x & x \leq 0 
\end{cases}
\]

Its derivative is:

\[
\text{Leaky ReLU}'(x) = \begin{cases} 
1 & x > 0 \\
\alpha & x \leq 0 
\end{cases}
\]

### Graph:
![Leaky ReLU Function and Derivative](path/to/leaky_relu_graph.png)

---

## 5. Parametric ReLU (PReLU)

The Parametric ReLU function allows the slope of the negative part to be learned:

\[
\text{PReLU}(x) = \begin{cases} 
x & x > 0 \\
\alpha \cdot x & x \leq 0 
\end{cases}
\]

Its derivative is:

\[
\text{PReLU}'(x) = \begin{cases} 
1 & x > 0 \\
\alpha & x \leq 0 
\end{cases}
\]

### Graph:
![PReLU Function and Derivative](path/to/prelu_graph.png)

---

## 6. GELU (Gaussian Error Linear Unit)

The GELU function is defined as:

\[
\text{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 \cdot x^3\right)\right)\right)
\]

Its derivative is complex but provides smooth gradients across values.

### Graph:
![GELU Function and Derivative](path/to/gelu_graph.png)

---

## 7. SiLU (Sigmoid Linear Unit)

The SiLU (also known as Swish) is defined as:

\[
\text{SiLU}(x) = x \cdot \sigma(x)
\]

Its derivative is:

\[
\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma'(x)
\]

### Graph:
![SiLU Function and Derivative](path/to/silu_graph.png)

---

## 8. Softmax Function

Softmax is defined as:

\[
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
\]

Its derivative is:

\[
\text{Softmax}'(x) = \text{Softmax}(x) \cdot (1 - \text{Softmax}(x))
\]

### Graph:
![Softmax Function and Derivative](path/to/softmax_graph.png)

---

## 9. ELU (Exponential Linear Unit)

The ELU function is defined as:

\[
\text{ELU}(x) = \begin{cases} 
x & x \geq 0 \\
\alpha \cdot (e^x - 1) & x < 0 
\end{cases}
\]

Its derivative is:

\[
\text{ELU}'(x) = \begin{cases} 
1 & x \geq 0 \\
\alpha \cdot e^x & x < 0 
\end{cases}
\]

### Graph:
![ELU Function and Derivative](path/to/elu_graph.png)

---

## 10. SELU (Scaled Exponential Linear Unit)

The SELU function is similar to ELU but scales the output:

\[
\text{SELU}(x) = \lambda \cdot \begin{cases} 
x & x \geq 0 \\
\alpha \cdot (e^x - 1) & x < 0 
\end{cases}
\]

Its derivative is:

\[
\text{SELU}'(x) = \lambda \cdot \begin{cases} 
1 & x \geq 0 \\
\alpha \cdot e^x & x < 0 
\end{cases}
\]

### Graph:
![SELU Function and Derivative](path/to/selu_graph.png)

---

## 11. Softplus Function

The Softplus function is a smooth approximation to ReLU:

\[
\text{Softplus}(x) = \log(1 + e^x)
\]

Its derivative is:

\[
\text{Softplus}'(x) = \sigma(x)
\]

### Graph:
![Softplus Function and Derivative](path/to/softplus_graph.png)

---

## 12. Mish Function

The Mish function is defined as:

\[
\text{Mish}(x) = x \cdot \tanh(\log(1 + e^x))
\]

Its derivative involves a complex combination of terms but retains smooth gradients.

### Graph:
![Mish Function and Derivative](path/to/mish_graph.png)

---

### Usage
To run the code and generate the graphs, run the `main()` function in the provided script. The graphs are saved to the `./graphs` folder.

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install jax matplotlib
