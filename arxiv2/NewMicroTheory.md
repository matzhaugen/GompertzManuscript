**Starting point:** The paper by Haugen and Gilbert (2022) observes that early COVID-19 cumulative mortality followed Gompertz curves rather than the logistic curves predicted by SIR models. They attempt to bridge these two growth regimes through a modified SIR model with higher-order interaction terms and a causality reversal, but this construction relies on a contrived network topology (single-neighbor permutation), drops the recovery rate ($\alpha \to 0$), and conflates geometric mean with median without justification.

**Our alternative microscopic model:** We proposed a coupled system in the log-domain:

$$dz_i = \left[\frac{\beta}{\theta}(1 - e^{\theta z_i}) + \beta(\bar{z}_{\theta,i} - z_i)\right]dt + \sqrt{\sigma}\,dW_i(t)$$

where $\bar{z}_{\theta,i} = \frac{1}{\theta}\ln(\sum_j W_{ij} e^{\theta z_j})$ is the order-$\theta$ power mean of neighbors' states. The parameter $\theta$ smoothly interpolates between Gompertz ($\theta \to 0$) and logistic ($\theta = 1$) dynamics, with intermediate values recovering the Richards growth family. This was confirmed numerically across multiple network topologies.

**Key findings:**

*The Gompertz case is structurally unique.* At $\theta = 0$, the dynamics are linear in $z$-space, which produces a complete decoupling between the macroscopic state $\bar{z}$ and the microscopic fluctuations $\delta_i = z_i - \bar{z}$. The mean satisfies $d\bar{z}/dt = -\beta\bar{z}$ exactly, regardless of the distribution of individual nodes. The deviations obey $\dot{\boldsymbol{\delta}} = \beta(-2I + W)\boldsymbol{\delta}$, which is unconditionally stable for any doubly stochastic $W$, meaning the system synchronizes from any initial condition. The macroscopic Gompertz equation is therefore exact from $t = 0$ — it does not require synchronization as a precondition but rather produces it as a consequence.

*The Richards/logistic case is fundamentally different.* At $\theta > 0$, the nonlinearity $e^{\theta z_i}$ couples the mean to the variance and higher moments:

$$\frac{1}{N}\sum_i e^{\theta z_i} = e^{\theta\bar{z}}\left(1 + \frac{\theta^2}{2}\text{Var}(z) + \cdots\right)$$

The variance inflates the effective mean, the macroscopic equation for $\bar{z}$ is not closed, and the scalar Richards growth law $\frac{d\ln M}{dt} = \frac{\beta}{\theta}(1 - M^\theta)$ is only recovered *after* the coupling has driven the system to synchronization. This also manifests as numerical instability — the arithmetic mean's sensitivity to outliers at high $\theta$ requires adaptive sub-stepping for stable integration, whereas the Gompertz case is stable with any reasonable step size.

*Coherence as symmetric information exchange.* The parameter $\theta$ controls the symmetry of how nodes contribute to the collective state. At $\theta = 0$, the coupling acts on log-ratios $z_j - z_i$, so every node contributes equally regardless of its absolute state — a node at $x = 0.001$ has the same influence as one at $x = 0.1$. At $\theta = 1$, the coupling acts on absolute differences $x_j - x_i$, and nodes with large $x$ dominate the arithmetic mean. Coherence, in this framework, is the degree to which each microscopic unit contributes equally to the macroscopic observable.

*Connection to the paper's stressor hypothesis.* The structural properties of the Gompertz limit — exact macroscopic reduction without synchronization, independence from microscopic variance, unconditional stability — are precisely what one would expect from a system driven by a shared environmental stressor acting on everyone simultaneously. The logistic limit, where leading nodes drive the dynamics and synchronization must be achieved before macroscopic laws emerge, corresponds to a hierarchical transmission process. The observation of Gompertz curves in early epidemic data therefore favors the coherent-stressor interpretation over person-to-person propagation, not merely as a curve-fitting exercise, but as a structural property of the underlying microscopic dynamics.

### Network properties
**1. Eigenvalues of doubly stochastic matrices in $[-1, 1]$**

A doubly stochastic matrix has non-negative entries with all rows and all columns summing to 1. For any such matrix $W$:

First, $\lambda = 1$ is always an eigenvalue, with eigenvector $\mathbf{1} = (1, 1, \ldots, 1)^T$, since $W\mathbf{1} = \mathbf{1}$ by the row-sum condition.

For the bound $|\lambda| \leq 1$: take any eigenvector $\mathbf{v}$ with $W\mathbf{v} = \lambda\mathbf{v}$, and look at the component $v_k$ with largest absolute value. Then:

$$|\lambda||v_k| = |\lambda v_k| = \left|\sum_j W_{kj}v_j\right| \leq \sum_j W_{kj}|v_j| \leq |v_k|\sum_j W_{kj} = |v_k|$$

So $|\lambda| \leq 1$. This is just the Gershgorin disc theorem applied to a row-stochastic matrix — each row's entries sum to 1, so all eigenvalues live in the unit disc.

For $\lambda = -1$ to occur, you need a bipartite structure where the network can be split into two groups that only connect to each other, never within. For most real-world networks this doesn't happen, so in practice eigenvalues lie in $(-1, 1]$, making the decay rate of deviations strictly positive.

**2. Variance and $\delta_i$**

Not exactly — $\delta_i = z_i - \bar{z}$ is the individual deviation, while the variance is a scalar:

$$\text{Var}(z) = \frac{1}{N}\sum_i \delta_i^2$$

What I meant is that the macroscopic equation for $\bar{z}$ at $\theta > 0$ doesn't close — computing $d\bar{z}/dt$ requires knowledge of $\frac{1}{N}\sum_i e^{\theta z_i}$, which depends on the full distribution of $z_i$, not just on $\bar{z}$. You can expand:

$$\frac{1}{N}\sum_i e^{\theta z_i} = e^{\theta\bar{z}}\cdot\frac{1}{N}\sum_i e^{\theta\delta_i} = e^{\theta\bar{z}}\left(1 + \frac{\theta^2}{2}\text{Var}(z) + \cdots\right)$$

So the variance (and higher moments of the $\delta_i$ distribution) leak into the equation for the mean. At $\theta = 0$ this expansion is just $1$ — no leakage.

**3. The double sum**

We want to evaluate $\frac{1}{N}\sum_i \sum_j W_{ij} z_j$. Switch the order of summation:

$$\frac{1}{N}\sum_i \sum_j W_{ij} z_j = \frac{1}{N}\sum_j z_j \sum_i W_{ij}$$

The inner sum $\sum_i W_{ij}$ is the $j$-th *column* sum of $W$. The doubly stochastic condition says all column sums equal 1, so:

$$= \frac{1}{N}\sum_j z_j \cdot 1 = \bar{z}$$

This is exactly where the "doubly" part of doubly stochastic matters. If $W$ were only row-stochastic ($\sum_j W_{ij} = 1$ for each $i$), the column sums $\sum_i W_{ij}$ could be anything, and this simplification would fail. The column-sum condition is what ensures that averaging the neighbor-means over all nodes recovers the global mean.

**4. How broad is the doubly stochastic assumption?**

It's broader than it might seem, but it does impose real constraints.

**What satisfies it naturally:** any symmetric network with uniform weights. If $A$ is the adjacency matrix and you define $W_{ij} = A_{ij}/k_i$ where $k_i$ is node $i$'s degree, then $W$ is row-stochastic by construction. It's also column-stochastic (hence doubly stochastic) if and only if the network is regular — every node has the same degree. So regular graphs (including complete graphs, ring lattices, regular random graphs) give doubly stochastic $W$ trivially.

**What doesn't satisfy it directly:** any network with heterogeneous degree distribution — which is most real networks (scale-free, log-normal, etc.). If node $i$ has degree 10 and node $j$ has degree 2, then with standard row-normalization $W_{ij} = 1/10$ but $W_{ji} = 1/2$, so column sums aren't uniform.

**But here's the important point:** our simulation doesn't actually use $W_{ij}$ as a matrix. We approximate the neighbor-mean as a degree-weighted global average $\bar{z}^{(W)} = \sum_j w_j z_j$ with $w_j = k_j / \sum_k k_k$, which is the standard mean-field approximation for configuration-model random graphs. Under this approximation, $\bar{z}^{(W)}$ is the *same* for every node $i$, so:

$$\frac{1}{N}\sum_i \bar{z}^{(W)}_i = \bar{z}^{(W)}$$

trivially. The doubly stochastic property is effectively built into the mean-field approximation itself — it's not a constraint on the network topology but a consequence of how we approximate the neighbor interactions.

For the exact (non-mean-field) case on irregular networks, you can recover doubly stochastic $W$ through the Sinkhorn-Knopp algorithm: iteratively normalize rows and columns of any non-negative matrix, and it converges to a doubly stochastic matrix. Physically this corresponds to reweighting the interactions so that each node has equal total influence on others and is equally influenced by others — a kind of democratic coupling regardless of degree. Whether this is physically justified depends on the application, but for the environmental stressor model in the paper, where the coupling represents shared exposure rather than directed transmission, symmetric influence is quite natural.

### Question: 
Let's dive into the network topology a bit more. Why is the doubly stochastic condition needed when we only need the normalization to 1 constraint in the proof that the eigenvalues lie in [-1,1]? Also, you say that the topology of a doubly stochastic matrix is symmetric and uniform, but we made a simulation with many topologies that are certainly not uniform, e.g. the Erdős-Rényi or the poisson networks. How do you reconcile this?

These are sharp questions and they expose a tension I glossed over. Let me address them carefully.

**On the eigenvalue bound:** You're right — the Gershgorin argument only needs row-stochasticity ($\sum_j W_{ij} = 1$) to get $|\lambda| \leq 1$. The doubly stochastic condition is *not* needed for the eigenvalue bound. I conflated two things: the eigenvalue bound (which constrains stability) and the double-sum cancellation (which gives the exact macroscopic reduction). These are separate properties.

Specifically:
- $|\lambda| \leq 1$ requires only row-stochasticity
- $\frac{1}{N}\sum_i\sum_j W_{ij}z_j = \bar{z}$ requires column-stochasticity ($\sum_i W_{ij} = 1$ for all $j$)
- Together (doubly stochastic) you get both stability *and* exact macroscopic reduction

So if $W$ is row-stochastic but not column-stochastic, the deviations still decay, but the macroscopic equation for $\bar{z}$ picks up a bias term because $\frac{1}{N}\sum_i\sum_j W_{ij}z_j \neq \bar{z}$.

**On the simulation versus the theory:** This is the more important point. The simulation doesn't actually use a $W_{ij}$ matrix at all. For $N = 10{,}000$ nodes, storing $W$ would require $10^8$ entries, and the neighbor-sum $\sum_j W_{ij}z_j$ would require $O(N^2)$ operations per time step. Instead, the simulation uses the mean-field approximation:

$$\sum_j W_{ij}z_j \approx \bar{z}^{(w)} = \sum_j w_j z_j, \qquad w_j = \frac{k_j}{\sum_k k_k}$$

This replaces every node's local neighbor-average with a single global degree-weighted average. Under this approximation, $\bar{z}_i^{(W)}$ is the same quantity for every $i$, so:

$$\frac{1}{N}\sum_i \bar{z}_i^{(W)} = \bar{z}^{(w)}$$

trivially — not because $W$ is doubly stochastic, but because the mean-field approximation has eliminated the local structure entirely.

So what the simulation actually demonstrates is that the macroscopic Gompertz dynamics emerge under the mean-field approximation for *any* degree distribution. The degree distribution affects the weights $w_j$ — and therefore affects which nodes contribute more to $\bar{z}^{(w)}$ — but the structural decoupling between mean and variance at $\theta = 0$ holds regardless, because it comes from the linearity of the dynamics in $z$, not from any property of $W$.

**What the doubly stochastic condition really buys you:** It's the condition under which the *exact* node-resolved dynamics (with actual $W_{ij}$ matrix, local neighborhoods, specific network topology) still give the exact macroscopic reduction. It answers the question: "if I simulate the actual network without mean-field approximation, for which networks does the clean theory still hold?"

The answer is: for networks where the coupling is doubly stochastic. This is more restrictive than the mean-field case, but less restrictive than "regular graphs" as I said before. Any coupling matrix can be made doubly stochastic via Sinkhorn-Knopp balancing, which corresponds to choosing interaction weights that compensate for degree heterogeneity.

**The honest picture is three levels:**

At the first level, with mean-field approximation (what our simulation does), the macroscopic reduction works for any degree distribution. The network topology enters only through the weights $w_j$, and the Gompertz decoupling is guaranteed by linearity alone. This is the weakest assumption and the broadest applicability.

At the second level, with exact doubly stochastic $W$ (the theoretical ideal), the macroscopic reduction works without mean-field approximation. Local network structure is fully resolved, but the coupling weights are balanced so that each node has equal aggregate influence.

At the third level, with exact row-stochastic but not column-stochastic $W$ (standard row-normalized adjacency), the macroscopic reduction acquires a bias. The mean $\bar{z}$ still approximately follows Gompertz, and the deviations still decay, but there's a correction term proportional to how far $W$ is from doubly stochastic. For networks where the degree distribution isn't too heavy-tailed, this correction is small.

So the simulation results are legitimate — they just operate at level one rather than level two. The fact that all network topologies collapsed onto the Gompertz curve is a consequence of the mean-field approximation being excellent for large $N$ with weak degree-state correlations, not of the networks secretly being doubly stochastic.

## Start of chat:
This is a great question. The fundamental thing the paper is trying to do is get from a system of interacting microscopic units to the macroscopic Gompertz equation

$$\frac{d}{dt}\ln M(t) = -\beta \ln M(t)$$

where $M(t)$ is some aggregate of the microscopic states. The problem with their approach is the contrived network topology and the hand-wavy averaging. Let me sketch an alternative that I think is cleaner.

**Start from multiplicative coupling directly**

The key insight the paper itself identifies — but doesn't fully exploit in the microscopic derivation — is that Gompertz growth is fundamentally about the *geometric* mean, not the arithmetic mean. So instead of writing an SIR-like additive interaction and then torturing it into a logarithmic form, start with a microscopic model where the coupling is already multiplicative.

Let each unit $x_i(t) \in (0,1]$ represent the fractional progression of individual $i$ toward some final state. Propose:

$$\frac{d \ln x_i}{dt} = -\beta \ln x_i + \beta \sum_j W_{ij}(\ln x_j - \ln x_i)$$

where $W_{ij} \geq 0$ is a coupling matrix with $\sum_j W_{ij} = 1$. The first term is individual Gompertz decay in the log-domain. The second term is a diffusive coupling — each unit gets pulled toward its neighbors, but in the log-domain.

Now define $z_i = \ln x_i$ and $\bar{z} = \frac{1}{N}\sum_i z_i$. Summing over $i$:

$$\frac{d\bar{z}}{dt} = -\beta \bar{z} + \beta \sum_i \sum_j W_{ij}(z_j - z_i)/N$$

The second term vanishes for any doubly stochastic $W$ (i.e., $\sum_i W_{ij} = 1$ as well), because then $\sum_i \sum_j W_{ij} z_j = \sum_j z_j$ and $\sum_i \sum_j W_{ij} z_i = \sum_i z_i$. So you get:

$$\frac{d\bar{z}}{dt} = -\beta \bar{z}$$

which is exactly the macroscopic Gompertz equation for the geometric mean $M = e^{\bar{z}} = (\prod x_i)^{1/N}$, with no constraints on $N$, no special network topology, and no need to set any parameter to zero.

**Why this works and their approach doesn't**

The critical difference is that the coupling here happens *in the log-domain*, which is the natural domain for Gompertz dynamics. The doubly stochastic condition on $W$ is mild — it just says the coupling is symmetric in a mass-preserving sense, which includes the complete graph ($W_{ij} = 1/N$), any regular graph with uniform weights, and many others. You don't need a single-neighbor constraint or a permutation network.

The diffusive coupling also has a natural interpretation: individuals experiencing more stress (more negative $z_i$) pull their neighbors toward greater effect, and vice versa, through a shared environment. This is consistent with the paper's own stressor hypothesis.

**Connecting back to the SIR interpretation**

If you want to maintain some contact with the SIR framework, you can observe that in the standard SIR model, the interaction $x_i(1-x_j)$ is bilinear. In the log-domain model above, the "interaction" $\ln x_j - \ln x_i$ is a *difference* of log-states, which corresponds to the *ratio* $x_j/x_i$ in the original domain. So the microscopic coupling is really:

$$\frac{\dot{x}_i}{x_i} = -\beta \ln x_i + \beta \sum_j W_{ij} \ln(x_j/x_i)$$

This can be read as: individual $i$'s growth rate depends on its own Gompertz decay plus a multiplicative comparison with neighbors. When everyone is in the same state ($x_i = x_j$ for all $i,j$), the coupling vanishes and you have pure Gompertz growth — which is the "coherent" state the paper discusses. When there's heterogeneity, the coupling acts to synchronize the population, which gives a mechanism for how coherence emerges rather than just assuming it.

**The stochastic extension**

Adding noise is also straightforward: replace the deterministic equation with

$$dz_i = \left[-\beta z_i + \beta \sum_j W_{ij}(z_j - z_i)\right]dt + \sqrt{\sigma}\, dW_i(t)$$

This is a system of coupled Ornstein-Uhlenbeck processes. The mean $\bar{z}$ still satisfies $d\bar{z} = -\beta \bar{z}\, dt + \frac{\sqrt{\sigma}}{N}\sum_i dW_i$, which for large $N$ converges to the deterministic Gompertz equation while each individual $z_i$ remains Gaussian — giving you the log-normal distribution of $x_i$ without any circularity.

So to summarize: by working in the log-domain from the start and using diffusive coupling with a doubly stochastic weight matrix, you get an exact and general microscopic foundation for macroscopic Gompertz growth. No contrived topology, no dropped parameters, no conflation of geometric mean with median.

## Richards microscopic model

The full model in the log-domain ($z_i = \ln x_i$) is:

$$dz_i = \left[\frac{\beta}{\theta}\left(1 - e^{\theta z_i}\right) + \beta\left(\bar{z}_\theta - z_i\right)\right]dt + \sqrt{\sigma}\,dW_i(t)$$

where the power-mean coupling target is:

$$\bar{z}_\theta = \frac{1}{\theta}\ln\left(\sum_j W_{ij}\,e^{\theta z_j}\right)$$

and $W_{ij}$ is a doubly stochastic weight matrix derived from the network adjacency.

Transforming back to the original domain $x_i = e^{z_i}$ using Itô's lemma ($dx = x\,dz + \frac{1}{2}x\sigma\,dt$):

$$dx_i = \left[\frac{\beta}{\theta}x_i\left(1 - x_i^\theta\right) + \beta x_i\left(\bar{z}_\theta - \ln x_i\right) + \frac{\sigma}{2}x_i\right]dt + \sqrt{\sigma}\,x_i\,dW_i(t)$$

with

$$\bar{z}_\theta = \frac{1}{\theta}\ln\left(\sum_j W_{ij}\,x_j^\theta\right)$$

**Limiting cases:**

At $\theta \to 0$ (L'Hôpital on both drift and power mean):

$$dz_i = \left[-\beta z_i + \beta\left(\sum_j W_{ij}z_j - z_i\right)\right]dt + \sqrt{\sigma}\,dW_i(t)$$

which is the coupled Ornstein-Uhlenbeck system giving Gompertz growth.

At $\theta = 1$:

$$dz_i = \left[\beta(1 - e^{z_i}) + \beta\left(\ln\sum_j W_{ij}e^{z_j} - z_i\right)\right]dt + \sqrt{\sigma}\,dW_i(t)$$

or equivalently in $x_i$:

$$dx_i = \left[\beta x_i(1 - x_i) + \beta x_i\ln\frac{\sum_j W_{ij}x_j}{x_i} + \frac{\sigma}{2}x_i\right]dt + \sqrt{\sigma}\,x_i\,dW_i(t)$$

where the first term is the standard logistic interaction and the second is a coupling toward the (log of the) arithmetic mean of neighbors.

**Macroscopic reduction:** For a doubly stochastic $W$ and large $N$, summing $dz_i/N$ over all nodes eliminates the coupling term. In the deterministic limit ($\sigma \to 0$), the geometric mean $M = \exp(\bar{z})$ satisfies:

$$\frac{d\ln M}{dt} = \frac{\beta}{\theta}\left(1 - \left\langle e^{\theta z_i}\right\rangle\right)$$

If the nodes are synchronized ($z_i \approx \bar{z}$ for all $i$), then $\langle e^{\theta z_i}\rangle \approx e^{\theta \bar{z}} = M^\theta$, giving:

$$\frac{d\ln M}{dt} = \frac{\beta}{\theta}(1 - M^\theta)$$

which is exactly the macroscopic Richards growth equation.

Let me think about this more carefully. The paper uses "coherence" to mean that the population acts as a unified whole — the Gompertz limit. So $\theta \to 0$ is *maximal* coherence and $\theta = 1$ is *minimal* coherence. That seems backwards from the coupling strength interpretation, so let me work through what $\theta$ actually controls.

Consider two nodes with states $x_i = 0.01$ and $x_j = 0.1$. What does each node "know" about the other through the coupling $\bar{z}_\theta$?

At $\theta = 1$ (arithmetic/logistic), the power mean is $\bar{x} = (0.01 + 0.1)/2 = 0.055$. The node at $0.01$ sees a neighbor ten times larger, and the coupling pushes it toward $0.055$ — an absolute difference of $0.045$, which is 4.5x its own state. The small node gets a massive kick. The large node barely moves. **Information flows from large to small.**

At $\theta \to 0$ (geometric/Gompertz), the power mean is $\sqrt{0.01 \times 0.1} = 0.032$. In the log-domain this is just $(\ln 0.01 + \ln 0.1)/2 = (-4.6 + -2.3)/2 = -3.45$. Both nodes are pulled symmetrically toward this midpoint in log-space. The coupling acts on the *ratio* $x_j/x_i$, not the difference $x_j - x_i$. **Information flows symmetrically regardless of scale.**

This is the key: $\theta$ controls the *symmetry of information exchange* between nodes.

At high $\theta$, the network is hierarchical. Nodes with large $x$ dominate the mean and effectively dictate the trajectory of smaller nodes, but aren't much influenced in return. The macroscopic behavior is driven by whichever nodes happen to be ahead — the "leading edge" of the population. This is exactly how a transmission network works in an SIR model: the infected (large $x$) drive the dynamics, the susceptible (small $x$) are passive recipients. There's no coherence because the population doesn't act as a unit; it follows its most advanced members.

At low $\theta$, the network is egalitarian. A node at $x = 0.001$ has just as much pull (in the log-domain) as a node at $x = 0.1$, because what matters is the *ratio* between them, not the gap. Every node contributes proportionally to the collective state. The population moves as a single entity because no node can run ahead without pulling everyone else along at the same *relative* rate. This is coherence — the system's evolution is determined by the entire ensemble simultaneously, not by a privileged subset.

So $\theta$ is best described as a **symmetry-breaking parameter**:

- $\theta \to 0$: scale-invariant coupling, every node contributes equally in relative terms → **coherent** collective dynamics → Gompertz
- $\theta = 1$: scale-dependent coupling, large nodes dominate → **hierarchical** dynamics driven by the leading edge → Logistic

The connection to the paper's stressor hypothesis becomes clear: if the entire biosphere is exposed to the same perturbation simultaneously, every organism's response is equally relevant regardless of how far along it is. That's the $\theta \to 0$ regime — coherent, symmetric, Gompertzian. A communicable disease, by contrast, spreads from the infected to the susceptible — a fundamentally asymmetric, hierarchical process. That's the $\theta \to 1$ regime.

This is a great distinction. The Gompertz case is special — it causes synchronization *and* the macroscopic reduction holds exactly even *before* synchronization. Let me show why.

At $\theta = 0$, the full dynamics are linear in $z$:

$$\frac{dz_i}{dt} = -2\beta z_i + \beta \sum_j W_{ij} z_j$$

Average over $i$:

$$\frac{d\bar{z}}{dt} = -2\beta\bar{z} + \beta \cdot \frac{1}{N}\sum_i\sum_j W_{ij} z_j = -2\beta\bar{z} + \beta\bar{z} = -\beta\bar{z}$$

where the doubly stochastic property gives $\frac{1}{N}\sum_i\sum_j W_{ij}z_j = \bar{z}$ exactly, as we showed. This is exact for any configuration of the $z_i$ — no synchronization assumption needed. The macroscopic Gompertz equation $d\bar{z}/dt = -\beta\bar{z}$ holds from $t = 0$.

Now look at what happens to the *deviations* $\delta_i = z_i - \bar{z}$:

$$\frac{d\delta_i}{dt} = \frac{dz_i}{dt} - \frac{d\bar{z}}{dt} = -2\beta(z_i) + \beta\sum_j W_{ij}z_j + \beta\bar{z}$$

$$= -2\beta(\bar{z} + \delta_i) + \beta\sum_j W_{ij}(\bar{z} + \delta_j) + \beta\bar{z}$$

$$= -2\beta\delta_i + \beta\sum_j W_{ij}\delta_j$$

This is a linear system $\dot{\boldsymbol{\delta}} = \beta(-2I + W)\boldsymbol{\delta}$. The eigenvalues of $W$ for a doubly stochastic matrix lie in $[-1, 1]$, so the eigenvalues of $-2I + W$ lie in $[-3, -1]$. All eigenvalues are strictly negative, meaning every deviation mode decays exponentially with rate at least $\beta$. The system is unconditionally stable — deviations shrink regardless of their initial magnitude.

So for Gompertz ($\theta = 0$):
- The mean $\bar{z}$ follows exact Gompertz dynamics from the start, independent of the deviations
- The deviations decay exponentially and independently of the mean
- The mean and the variance are completely decoupled

This decoupling is a consequence of the linearity of the $\theta = 0$ dynamics in $z$-space. It's the same reason why linear systems in physics are so tractable — superposition holds, so the collective motion and the internal fluctuations don't talk to each other.

For $\theta > 0$, the drift $\frac{\beta}{\theta}(1 - e^{\theta z_i})$ is nonlinear in $z_i$, and the power mean $\bar{z}_{\theta,i}$ is a nonlinear function of the $z_j$. Both of these break the decoupling. The average $\frac{1}{N}\sum_i(1 - e^{\theta z_i})$ depends on the full distribution of $z_i$, not just on $\bar{z}$. So the macroscopic equation for $\bar{z}$ contains terms involving the variance, skewness, etc. — it's not closed. It only *becomes* closed when synchronization eliminates those higher moments.

This is really the mathematical content of what the paper calls "coherence." It's not just that the nodes happen to be synchronized — it's that at $\theta = 0$, the macroscopic dynamics are *structurally independent* of the microscopic heterogeneity. The collective behavior is determined entirely by the collective state, regardless of how much internal disorder exists. That's a much stronger property than synchronization, and it's unique to the Gompertz limit.

------- New q and a -----

You're right — the self-interaction $x_i(1-x_i)$ isn't physical for a communicable disease. The actual microscopic logistic model with transmission is:

$$\dot{x}_i = \beta(1-x_i)\sum_j w_{ij} x_j$$

Now average over $i$:

$$\frac{d\bar{x}}{dt} = \frac{\beta}{N}\sum_i (1-x_i)\sum_j w_{ij} x_j = \frac{\beta}{N}\sum_i\sum_j w_{ij} x_j - \frac{\beta}{N}\sum_i\sum_j w_{ij} x_i x_j$$

The first term, using column-normalization ($\sum_i w_{ij} = 1$), gives $\frac{\beta}{N}\sum_j x_j = \beta\bar{x}$.

The second term is $\frac{\beta}{N}\sum_i\sum_j w_{ij} x_i x_j$, which is a weighted average of pairwise products. This is *not* $\beta\bar{x}^2$ unless $x_i$ and $x_j$ are uncorrelated across the network — in which case $\overline{w_{ij}x_i x_j} \approx \bar{x}^2$.

So:

$$\frac{d\bar{x}}{dt} = \beta\bar{x} - \frac{\beta}{N}\sum_i\sum_j w_{ij} x_i x_j$$

For this to reduce to $\beta\bar{x}(1-\bar{x})$, you need:

$$\frac{1}{N}\sum_i\sum_j w_{ij} x_i x_j = \bar{x}^2$$

This is a stronger condition than just $\text{Var}(x) = 0$. Even if all nodes have the same value ($x_i = \bar{x}$), this gives $\bar{x}^2 \frac{1}{N}\sum_i\sum_j w_{ij} = \bar{x}^2$, which works if $W$ is doubly stochastic. But if the nodes are heterogeneous, you need the network-weighted covariance $\text{Cov}_W(x_i, x_j)$ to vanish — meaning the states of connected nodes must be uncorrelated.

This is actually a *more demanding* condition than simple synchronization. It requires either that everyone is in the same state (synchronization), or that the network structure is uncorrelated with the state distribution (the mean-field assumption in its true sense — that who is connected to whom is independent of who is infected).

So the closure hierarchy is:

For Gompertz ($\theta = 0$): closure is **unconditional**. Linear dynamics in $z$-space mean $\dot{\bar{z}} = -\beta\bar{z}$ regardless of the distribution of $z_i$, the network structure, or any correlations. You only need $W$ to be doubly stochastic for the coupling to cancel, which is a property of the network, not of the state.

For logistic ($\theta = 1$) with transmission: closure requires **both** synchronization (or zero variance) **and** independence between network structure and node states. The pairwise product $x_i x_j$ introduces a joint dependence that the arithmetic mean cannot disentangle without assuming the states are uncorrelated across edges.

This actually strengthens the paper's argument considerably. The logistic/SIR model doesn't just need nodes to be synchronized — it needs the *network-state correlation* to vanish. In a real epidemic, this is exactly what doesn't happen: highly connected nodes get infected earlier, creating strong network-state correlations. The mean-field approximation sweeps this under the rug. The Gompertz model has no such vulnerability because the linear $z$-dynamics never generate terms involving products of different nodes' states.