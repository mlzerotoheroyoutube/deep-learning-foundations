# Backpropagation

**Level:** Intermediate  ·  **Topic:** Deep Learning Foundations

▶️ **Watch the video:** https://youtu.be/st2Ufh6kNJQ

> Backpropagation computes the gradient of a scalar loss with respect to every parameter of a network in one backward sweep, by applying the chain rule from the output toward the input and reusing intermediate results. It costs roughly two forward passes, which is why training billion-parameter models is possible at all.

## The intuition
Think of a relay race team that loses by 3 seconds. The coach wants to know how much each runner is to blame. She starts at the **last** runner: "you were 1 second slow." Then she asks the second-to-last runner, "how much did your slow handoff slow down the last runner?" Blame flows **backwards** down the team, one runner at a time, and each runner only needs to know how they affected the next person. That's backpropagation.

A neural network is a chain of simple calculations. Each one takes numbers in and passes numbers out. At the end we get a **loss**, a single number measuring how wrong the network was. We want to know, for every adjustable knob (a **weight**), "if I turn this knob up a tiny bit, does the loss go up or down, and by how much?" That "how much" is called the **gradient**.

Tiny example: suppose $y = 3 \times w$ and the loss is $L = y \times y$. With $w = 2$: $y = 6$, $L = 36$. Going backwards: nudging $y$ up by 1 raises $L$ by about $2 \times 6 = 12$. Nudging $w$ up by 1 raises $y$ by $3$. So nudging $w$ raises $L$ by about $12 \times 3 = 36$. We multiplied the "blame" at each link of the chain. A real network does exactly this, but with millions of links, and it reuses each partial answer instead of recomputing it. Then Gradient Descent uses those numbers to turn every knob slightly in the helpful direction.

## Core notes
- Backprop = **chain rule + dynamic programming**: compute each local derivative once, multiply adjoints backward, reuse shared sub-results.
- Two passes: the **forward pass** computes and caches activations; the **backward pass** propagates $\bar{h} = \partial L/\partial h$ from loss to inputs.
- For a linear layer $z = Wx + b$: $\partial L/\partial W = \bar{z}\,x^\top$, $\partial L/\partial b = \bar{z}$, $\partial L/\partial x = W^\top \bar{z}$. Memorise these three.
- Elementwise nonlinearities multiply the adjoint by their derivative: $\bar{z} = \bar{h} \odot \sigma'(z)$ (see Activation Functions).
- Softmax Function + cross-entropy has the famously clean gradient $\partial L/\partial z = p - y$.
- When a variable feeds several consumers, its adjoint is the **sum** of the adjoints flowing back from each (multivariate Chain Rule).
- Cost: backward ≈ 2× forward FLOPs; memory is dominated by stored activations, which is why Gradient Checkpointing and Mixed Precision Training exist.
- Repeated multiplication by Jacobians causes Vanishing and Exploding Gradients; Residual Connections, Weight Initialization and normalisation fix it.
- Through time, backprop on an unrolled RNN is Backpropagation Through Time.
- Always **gradient-check** a hand-written backward pass with central finite differences (relative error ~1e-7 or smaller in float64).
- Frameworks (PyTorch, JAX) implement this generally as Automatic Differentiation; you rarely write backward passes by hand, but you must understand them to debug.

## The math
**Setup.** An $L$-layer MLP with input $h^{(0)} = x$:
$$
z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}, \qquad h^{(l)} = \sigma\big(z^{(l)}\big), \qquad l = 1,\dots,L,
$$
and a scalar loss $\mathcal{L} = \ell\big(h^{(L)}, y\big)$. Here $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$, $b^{(l)} \in \mathbb{R}^{n_l}$, $\sigma$ is an elementwise nonlinearity, and $\odot$ is the elementwise product.

**Define the error signal** $\delta^{(l)} := \partial \mathcal{L}/\partial z^{(l)} \in \mathbb{R}^{n_l}$. The four backprop equations are:
$$
\begin{aligned}
\delta^{(L)} &= \nabla_{h^{(L)}}\ell \odot \sigma'\big(z^{(L)}\big) &\text{(BP1: output error)}\\
\delta^{(l)} &= \big(W^{(l+1)\top}\delta^{(l+1)}\big) \odot \sigma'\big(z^{(l)}\big) &\text{(BP2: propagate)}\\
\frac{\partial \mathcal{L}}{\partial b^{(l)}} &= \delta^{(l)} &\text{(BP3: bias)}\\
\frac{\partial \mathcal{L}}{\partial W^{(l)}} &= \delta^{(l)}\, h^{(l-1)\top} &\text{(BP4: weights)}
\end{aligned}
$$

**Derivation of BP2.** $z^{(l+1)}_j = \sum_k W^{(l+1)}_{jk}\sigma(z^{(l)}_k) + b^{(l+1)}_j$, so by the multivariate Chain Rule
$$
\delta^{(l)}_k = \sum_j \frac{\partial \mathcal{L}}{\partial z^{(l+1)}_j}\frac{\partial z^{(l+1)}_j}{\partial z^{(l)}_k} = \sum_j \delta^{(l+1)}_j W^{(l+1)}_{jk}\,\sigma'\big(z^{(l)}_k\big).
$$
The sum over $j$ is exactly the accumulation of adjoints from every consumer of $z^{(l)}_k$.

**Softmax + cross-entropy.** With $p = \operatorname{softmax}(z)$, one-hot $y$, and $\mathcal{L} = -\sum_i y_i \log p_i$: using $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$,
$$
\frac{\partial \mathcal{L}}{\partial z_j} = -\sum_i y_i(\delta_{ij} - p_j) = p_j - y_j .
$$
The same form $p - y$ holds for sigmoid + binary cross-entropy and for any canonical-link GLM.

**Mini-batch form.** Stacking a batch as rows $H^{(l)} \in \mathbb{R}^{B\times n_l}$ with $Z^{(l)} = H^{(l-1)}W^{(l)\top} + \mathbf{1}b^{(l)\top}$ and $\mathcal{L} = \frac{1}{B}\sum_i \ell_i$:
$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \Delta^{(l)\top} H^{(l-1)}, \qquad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \Delta^{(l)\top}\mathbf{1}, \qquad \Delta^{(l-1)} = \big(\Delta^{(l)}W^{(l)}\big)\odot\sigma'\big(Z^{(l-1)}\big),
$$
where $\Delta^{(l)} \in \mathbb{R}^{B\times n_l}$ already contains the $1/B$ factor from the top.

**Why reverse mode is cheap.** For $f:\mathbb{R}^P \to \mathbb{R}$ composed of $K$ elementary operations, reverse mode computes $\nabla f$ with $O(K)$ operations (a small constant times the cost of $f$), while forward mode needs $P$ Jacobian–vector products, i.e. $O(PK)$. Memory for reverse mode is $O(K)$ stored intermediates, reducible to $O(\sqrt{K})$ with checkpointing.

**Gradient check.** For a parameter $\theta_i$,
$$
g_i^{\text{num}} = \frac{\mathcal{L}(\theta + \epsilon e_i) - \mathcal{L}(\theta - \epsilon e_i)}{2\epsilon} + O(\epsilon^2), \qquad \text{rel.\ err} = \frac{\|g^{\text{num}} - g\|_\infty}{\|g^{\text{num}}\|_\infty + \|g\|_\infty}.
$$

## Run the code
```bash
pip install numpy torch
python backpropagation.py
```

## Check yourself
<details><summary>Derive the gradient of softmax + cross-entropy with respect to the logits.</summary>

Let $p_i = e^{z_i}/\sum_k e^{z_k}$ and $\mathcal{L} = -\sum_i y_i\log p_i$ with one-hot $y$. The softmax Jacobian is $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$. Then $\partial\mathcal{L}/\partial z_j = -\sum_i y_i \frac{1}{p_i} p_i(\delta_{ij}-p_j) = -y_j + p_j\sum_i y_i = p_j - y_j$. The result is bounded in $[-1, 1]$, which is one reason this pairing trains well; it is also why frameworks fuse the two ops into one numerically stable kernel.
</details>

<details><summary>Why is reverse mode used for neural networks instead of forward mode?</summary>

Forward mode computes a Jacobian–vector product per pass, so getting the gradient of a scalar loss with respect to $P$ parameters needs $P$ passes. Reverse mode computes a vector–Jacobian product per pass, so one backward sweep yields all $P$ partial derivatives at a cost of a small constant times the forward pass. Neural nets have one scalar output and millions to trillions of inputs (parameters), so reverse mode is asymptotically better by a factor of $P$. The price is memory: reverse mode must store intermediate activations.
</details>

<details><summary>What is the time and memory cost of backprop, and how do you reduce memory?</summary>

Backward FLOPs are roughly 2× forward FLOPs for matmul-dominated nets (one matmul for the weight gradient and one for the input gradient), giving total training cost ≈ 3× forward, or the familiar ≈ 6N FLOPs per token for Transformers. Memory is dominated by cached activations, scaling with batch × sequence length × width × depth. Reduce it with Gradient Checkpointing (recompute activations; $O(\sqrt{K})$ memory for one extra forward), Mixed Precision Training (bf16 activations), smaller micro-batches with gradient accumulation, activation offloading, and sharding across devices (Data Parallelism and FSDP).
</details>

<details><summary>Your hand-written backward pass gives a gradient-check relative error of 1e-2. How do you debug it?</summary>

Check each layer in isolation, gradient-checking one module at a time with random upstream gradients. Common culprits: a missing $1/B$ factor, a missing transpose, not accumulating gradients for a tensor used twice, caching the wrong tensor (post- instead of pre-activation), or in-place modification of a cached value. Make sure the check runs in float64, uses central differences with $\epsilon \approx 10^{-5}$, and avoids points right at ReLU kinks. Dropout or other randomness must use the same mask in the perturbed evaluations.
</details>

<details><summary>Is backpropagation the reason networks suffer from vanishing gradients?</summary>

No. Backprop computes the exact gradient; vanishing gradients mean the exact gradient is genuinely tiny. The cause is the product of layer Jacobians $\prod_l W^{(l)\top}\operatorname{diag}(\sigma'(z^{(l)}))$, whose norm shrinks or grows exponentially with depth when singular values sit away from 1 (e.g. saturating sigmoids with $\sigma' \le 0.25$). The fixes are architectural: ReLU-family activations, careful Weight Initialization, Batch Normalization or Layer Normalization and RMSNorm, and Residual Connections, which add an identity path to the Jacobian.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.