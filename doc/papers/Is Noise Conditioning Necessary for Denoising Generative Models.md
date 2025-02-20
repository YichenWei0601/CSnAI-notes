# Is Noise Conditioning Necessary for Denoising Generative Models?

#### Link: [Is Noise Conditioning Necessary for Denoising Generative Models?](https://arxiv.org/abs/2502.13129)

### Abstract

> It is widely believed that noise conditioning is indispensable for denoising diffusion models to work successfully. This work challenges this belief. Motivated by research on blind image denoising, we investigate a variety of denoising-based generative models in the absence of noise conditioning. To our surprise, most models exhibit graceful degradation, and in some cases, they even perform better without noise conditioning. We provide a theoretical analysis of the error caused by removing noise conditioning and demonstrate that our analysis aligns with empirical observations. We further introduce a noise-*unconditional* model that achieves a competitive FID of 2.23 on CIFAR-10, significantly narrowing the gap to leading noise-conditional models. We hope our findings will inspire the community to revisit the foundations and formulations of denoising generative models.

> 广泛认为，噪声条件对于去噪扩散模型成功工作至关重要。这项工作挑战了这一信念。受盲图像去噪研究启发，我们调查了在无噪声条件下的各种基于去噪的生成模型。令人惊讶的是，大多数模型表现出优雅的退化，在某些情况下，它们甚至在没有噪声条件的情况下表现更好。我们对去除噪声条件引起的错误进行了理论分析，并证明我们的分析与经验观察一致。我们进一步介绍了一个无噪声条件模型，在 CIFAR-10 上实现了具有竞争力的 FID 2.23，显著缩小了与领先噪声条件模型的差距。我们希望我们的发现能够激励社区重新审视去噪生成模型的基础和公式。

### Notes

- **Training**: During training, a data point $x$ is sampled from the data distribution $p(x)$, and a noise $\epsilon$ is sampled from a noise distribution $p(\epsilon)$, such as a normal distribution $\mathcal{N}(0, I)$. A noisy image $z$ is given by:

  $z = a(t)x + b(t)\epsilon.$

  Here, $a(t)$ and $b(t)$ are schedule functions that are method-dependent. The time step $t$, which can be a continuous or discrete scalar, is sampled from $p(t)$. Without loss of generality, we refer to $b(t)$, or simply $t$, as the noise level. **Note that $a(t)$ and $b(t)$ is pre-designed, not trained.**

- **Sampling**: Given trained $NN_{\theta}$, the sampler performs iterative denoising. Specifically, with an initial noise $x_0 \sim \mathcal{N}(0, b(t_{\max})^2 I)$, the sampler iteratively computes:

  $$x_{i+1} := \kappa_i x_i + \eta_i NN_{\theta}(x_i \mid t_i) + \zeta_i \epsilon_i. \qquad (4)$$

  Here, a discrete set of time steps $\{t_i\}$ is pre-specified and indexed by $0 \leq i < N$. The schedules, $\kappa_i$, $\eta_i$, and $\zeta_i$, can be computed from the training-time noise schedules in Table 1 (see their specific forms in Appendix D). 

- ![image-20250220215022888](.\pic\image-20250220215022888.png)

- If the conditional distribution $p(t \mid z)$ is close to a **Dirac delta function**, the effective target would be the same with and without conditioning on $t$. If so, assuming the network is capable enough to fit the target, the noise-unconditional variant would produce the same output as the conditional one.

- Given a noisy image $z = (1 - t_*)x + t_*\epsilon$ produced by a given $t_*$, the variance of $t$ under the conditional distribution $p(t \mid z)$, is: $$\text{Var}_{t \sim p(t \mid z)}[t] \approx  \frac{t_*^2}{2d}$$, suggesting that **high-dimensional data will lead to low variance.**

- In addition to investigating existing models, we also design a diffusion model specifically tailored for noise unconditioning. Our motivation is to find schedule functions that are more robust in the absence of noise conditioning, while still maintaining competitive performance. To this end, we build upon the highly effective EDM framework (Karras et al., 2022) and modify its schedules.

  A core component of EDM is a "preconditioned" denoiser:

  $$c_{\text{skip}}(t)\hat{z} + c_{\text{out}}(t)NN_{\theta}\left(c_{\text{in}}(t)\hat{z} \mid t\right)$$

  Here, $\hat{z} := x + t\epsilon$ is the noisy input before the normalization performed by $c_{\text{in}}(t)$, **which we simply set as $c_{\text{in}}(t) = \frac{1}{\sqrt{1 + t^2}}$. The main modification we adopt for the noise unconditioning scenario is to set: $$c_{\text{out}}(t) = 1.$$**

  As a reference, EDM set $c_{\text{out}}(t) = \frac{\sigma_d t}{\sqrt{\sigma_d^2 + t^2}}$ where $\sigma_d$ is the data std. As $c_{\text{out}}(t)$ is the coefficient applied to $NN_{\theta}$, we expect setting it to a constant will free the network from modeling a $t$-dependent scale. In experiments (Section 6.2), this simple design exhibits a lower error bound (Section 4.4) than EDM. We name this model as uEDM, which is short for (noise-)unconditional EDM. For completeness, the resulting schedules of uEDM are provided in Section D.5.
