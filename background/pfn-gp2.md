# PFN and GP: Correcting the Theory and Setting the Research Agenda

*This document continues from `pfn-gp.md`. It corrects two claims in that document that are approximately right but misleading, and frames the open research questions that arise naturally from your existing GP work.*

---

## Part 1: Theoretical Background

### 1.1 Why One Attention Head Is Not a Kernel Smoother

`pfn-gp.md` claimed that a single attention head acts "almost exactly" like a Nadaraya-Watson (NW) kernel estimator. This is wrong in a precise way that matters.

In a PFN transformer, each training point is packed into a **single token** containing both its feature and its label:

$$V_j = (Y_j,\; X_j)^\top$$

The test point is the token $v = (0, x_*)^\top$. For head $h$, the attention score between the test token and training token $j$ is:

$$a_j^{(h)} \propto \exp\!\bigl(v^\top W_Q^{(h)} V_j\bigr)$$

Expanding for 1-D inputs:

$$v^\top W_Q^{(h)} V_j = \underbrace{x_* \cdot [W_Q]_{22}}_{\alpha} \cdot X_j \;+\; \underbrace{x_* \cdot [W_Q]_{21}}_{\beta} \cdot Y_j$$

The weight on training point $j$ is a **linear combination of $X_j$ and $Y_j$**. The GP kernel weight $w_j = k(x_*, X_j)/\sum_k k(x_*, X_k)$ depends only on $X_j$. As long as $\beta \neq 0$ the two are structurally different things.

**Nagler (2023) Theorem 6.3** formalises this: the bias of a one-layer PFN converges to an integral under a *tilted* version of the data distribution, $g_h(s) \propto \exp(v^\top W_Q^{(h)} s)\, p_0(s)$. The iso-density lines of this distribution in the $(X_j, Y_j)$ plane are **straight diagonal lines**, not vertical bands. **Theorem 5.4** strengthens this: the bias cannot vanish for a one-layer model regardless of how many training points are given. One layer is theoretically limited.

> **Reading.** Nagler (2023), *Statistical Foundations of Prior-Data Fitted Networks*, ICML 2023. Read Sections 5 and 6, focusing on Theorem 5.4 and Theorem 6.3.

---

### 1.2 Multi-Layer Theory: The Neumann Series

`pfn-gp.md` stated "Layers 2+ approximate matrix inversion." This is right in spirit but the precision matters.

The GP posterior mean requires $(K + \sigma^2 I)^{-1} y$. This can be computed iteratively via gradient descent:

$$\alpha^{(0)} = 0, \qquad \alpha^{(t+1)} = \alpha^{(t)} + \eta\bigl(y - (K+\sigma^2 I)\alpha^{(t)}\bigr)$$

Unrolling this gives the **Neumann series**:

$$(K+\sigma^2 I)^{-1} = \frac{1}{\eta}\sum_{t=0}^\infty \bigl(I - \eta(K+\sigma^2 I)\bigr)^t$$

which converges for any positive definite matrix with $0 < \eta < 2/\lambda_{\max}$. The $t=0$ term gives the Nadaraya-Watson estimator; each subsequent term corrects for correlations between training points.

**How many layers?** The series reaches error $\varepsilon$ in $O(\kappa \log 1/\varepsilon)$ steps, where:

$$\kappa = \frac{\lambda_{\max}(K) + \sigma^2}{\sigma^2}$$

| Setting | $\ell$ | $\sigma^2$ | $\kappa$ | Layers needed |
|---------|--------|------------|----------|---------------|
| Easy    | 0.1    | 0.5        | ~2       | 1–2           |
| Hard    | 1.0    | 0.01       | ~100     | many          |

A 6-layer model is a compromise over a distribution of hyperparameters. It works well when the kernel is easy to invert and degrades for ill-conditioned systems.

**Caveat.** The Neumann series picture is exact only under *linear* attention. With softmax, the correspondence is approximate — the network may find different strategies.

> **Reading.** von Oswald et al. (2022), *Transformers Learn In-Context by Gradient Descent*, arXiv:2212.07677. Read Sections 1–3 for the precise construction. Akyürek et al. (2023), *What Learning Algorithm is In-Context Learning?*, ICLR 2023 (arXiv:2211.15661) gives empirical evidence.

---

### 1.3 The Key Advantage: Amortised Bayesian Model Averaging

Your `simple_gp.ipynb` demonstrates the standard GP workflow: fit hyperparameters $\theta = (\ell, \sigma_f^2, \sigma_n^2)$ by maximising the marginal likelihood (Type-II ML). This gives a point estimate $\hat\theta$. The fully Bayesian alternative marginalises over $\theta$:

$$p(y_* \mid x_*, X, y) = \int p(y_* \mid x_*, X, y, \theta)\; p(\theta \mid X, y)\; d\theta$$

Your `toy_problem.ipynb` already approximates this with MCMC via Pyro — that is precisely the ground-truth comparison.

**What a PFN does:** it is meta-trained over a prior $p(\theta)$. A single forward pass approximates the above integral without any per-dataset optimisation. Inference costs $O(n^2)$ regardless of how many hyperparameter configurations were covered during training.

**The mixture-of-GPs view.** For fixed $\theta$ the GP predictive is Gaussian $\mathcal{N}(\mu_\theta, \sigma^2_\theta)$. Marginalising gives a **mixture of Gaussians**:

$$p(y_* \mid x_*, X, y) = \int \mathcal{N}(\mu_\theta, \sigma^2_\theta)\; p(\theta \mid X, y)\; d\theta$$

Its total variance decomposes as:

$$\text{Var}[y_*] = \underbrace{\mathbb{E}_\theta[\sigma^2_\theta]}_{\text{noise}} + \underbrace{\text{Var}_\theta[\mu_\theta]}_{\text{uncertainty about }\theta}$$

The second term is what ML-II discards when it plugs in $\hat\theta$. It is large when $n$ is small and $p(\theta \mid X, y)$ is broad. This gives a clear prediction: at small $n$ the PFN predictive variance should be larger than GP-ML variance and closer to the MCMC estimate. As $n$ grows and $p(\theta \mid X, y)$ concentrates, all three should converge.

The mixture is generally non-Gaussian and can be bimodal when two very different lengthscales are equally consistent with the data. A Gaussian output head cannot represent this — it is the theoretical reason why PFN training uses a discretised output distribution (`BarDistribution`) that can represent arbitrary shapes.

> **Reading.** Müller et al. (2022), *Transformers Can Do Bayesian Inference*, ICLR 2022. Sections 1–3 and the GP experiments in Section 5. Garnelo et al. (2018), *Neural Processes* (arXiv:1807.01622) for a related latent-variable perspective.

---

### 1.4 Synthesis: Three Lenses on the Same Architecture

| | **Nagler (2023)** | **von Oswald et al.** | **This project** |
|---|---|---|---|
| Attention type | Softmax | Linear | Softmax |
| Kernel | Fixed | Fixed (linear) | Data-adaptive |
| Labels in attn. score | Yes ($\beta Y_j$) | No | Yes, but structured? |
| Hyperparameter tuning | Not modelled | Not modelled | Central topic |
| Multi-layer theory | Open | Yes ($L$ GD steps) | To derive |
| Bias vanishes? | No (Thm 5.4) | Yes (linear model) | Under conditions? |

Nagler describes what one layer *does* and why the bias cannot vanish. Oswald describes what many layers *could* do — approximate the GP posterior — but only under linear attention and a fixed kernel. The trained PFN adapts its effective kernel to the observed data, which is the source of its practical advantage and the open theoretical question.

---

## Part 2: Research Objectives

**Q1 — Where is the breakpoint between GP-ML and PFN?**

With few observations, Type-II ML overfits hyperparameters and the PFN (which carries a prior over $\theta$ from meta-training) should win. As $n$ grows, ML-II becomes reliable and the advantage disappears. Find the crossover $n^*$ empirically by comparing all three approaches — GP-ML, GP-MCMC, and PFN — over a range of $n$.

**Q2 — How many layers are needed for a given condition number?**

The Neumann series predicts that ill-conditioned kernel systems need more correction steps. Does the empirical NLL gap between a 1-layer PFN and the exact GP posterior close faster with $L$ for easy-$\kappa$ settings than for hard-$\kappa$ settings?

**Q3 — Does the PFN's predictive distribution capture hyperparameter uncertainty?**

When $p(\theta \mid X, y)$ is broad, the true predictive distribution is a non-Gaussian mixture. Does the PFN's output distribution reflect this? Is it wider than GP-ML, closer to the MCMC estimate, and does it narrow as $n$ increases and $\theta$ becomes identifiable?

---

## Part 3: Guiding Experiments

### Step 1: Three-Way Comparison as a Function of $n$  (Q1)

This is the central experiment. For a 1-D RBF GP with known hyperparameters $(\ell, \sigma_f^2, \sigma_n^2)$, compare the three predictive approaches at each $n \in \{2, 5, 10, 20, 50\}$:

1. **GP-ML**: fit $\hat\theta$ with your GPyTorch `ExactGP` and marginal likelihood optimisation (as in `simple_gp.ipynb`)
2. **GP-MCMC**: use Pyro to sample from $p(\theta \mid X, y)$ and marginalise (as in `toy_problem.ipynb`)
3. **PFN**: single forward pass with the pretrained model

For each $n$, generate 200 independent test instances and compute:
- **NLL** — negative log-likelihood of the true $y_*$ under each predictive distribution
- **RMSE** — mean squared error of the predictive mean

Plot both metrics vs $n$ on the same axes for all three methods. Look for two things: (a) the crossover $n^*$ where GP-ML catches up to the PFN, and (b) whether GP-MCMC tracks the PFN closely or GP-ML closely across the full range.

The key question is whether $n^* \approx 10$ (consistent with the prior width you use) and whether MCMC always dominates ML-II, or whether ML-II sometimes wins when the prior width is mismatched.

### Step 2: Layer-Count Sweep (Q2)

Train PFNs with $L \in \{1, 2, 4, 8\}$ layers on the same GP-RBF prior. For each, compute the NLL gap to the exact GP posterior on 200 test instances with $n = 20$ context points. Run under two configurations:

| Config | $\ell$ | $\sigma^2$ | $\kappa$ |
|--------|--------|------------|----------|
| Easy   | 0.1    | 0.5        | ~2       |
| Hard   | 0.8    | 0.05       | ~80      |

Plot $\Delta_\text{NLL}(L)$ for both. If the Neumann series picture holds, the Hard gap should close much more slowly with $L$ than the Easy gap.

### Step 3: Predictive Distribution Shape (Q3)

Construct an ambiguous context: $n = 5$ observations consistent with both a smooth ($\ell = 0.8$) and a wiggly ($\ell = 0.1$) function. For example, draw from the $\ell = 0.8$ GP but place all five $X_j$ within a narrow range so pairwise distances cannot distinguish the two lengthscales.

At a test point $x^*$ outside this range, compute and plot side by side:
1. GP-ML predictive (a single Gaussian at $\hat\theta$)
2. MCMC mixture (histogram of samples from $p(y_* \mid x_*, X, y)$)
3. PFN output (BarDistribution histogram)

The mixture-of-GPs prediction: the MCMC histogram should be wider and possibly bimodal; the PFN should be closer to MCMC than to GP-ML. If the PFN collapses to a narrow Gaussian, it is not representing the hyperparameter uncertainty.

Repeat for an unambiguous context ($n = 30$, clearly smooth observations) and check that all three methods converge to approximately the same distribution.
