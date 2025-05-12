
+++
date = '2025-05-12T07:50:00-04:00'
draft = false
title = '[DRAFT] Tutorial & Reflections: Building a Variational Autoencoder (VAE) for Airfoil Generation'
tags = ["VAE", "PyTorch", "Deep Learning", "Generative Models", "Tutorial", "Learning Reflections"]
+++

> _This article is a preliminary draft and subject to future revisions._

## Introduction

This post walks through the process of building and training a Variational Autoencoder (VAE) using PyTorch, based on an assignment focused on generating airfoil shapes. We'll cover the fundamental concepts of VAEs, the implementation details, and share some key learnings from the process.

**Setup Notes:**
* This project uses Python 3.10 and PyTorch >= 2.
* Specific utility functions (`AirfoilDataset`, `VAE_Tracker`, etc.) are assumed to be available.

## Tutorial: Understanding and Implementing VAEs

VAEs are powerful generative models. Unlike standard autoencoders, they learn a probability distribution for the latent space.

### 1. VAE Fundamentals

#### a) The Encoder
The VAE encoder doesn't just output a latent vector $z$. Instead, given an input $x$, it outputs parameters for a probability distribution $q_{\phi}(z|x)$, typically a Normal distribution with mean $\mu_{\phi}(x)$ and variance $\sigma_{\phi}^{2}(x)$. $\phi$ represents the learnable parameters of the encoder network. The input $x$ can be various data types (vectors, images, etc.), and the latent code $z$ is often a vector. For simplicity, elements of $z$ are usually assumed to be independent, making the covariance matrix diagonal.

#### b) The Decoder
The decoder aims to model $p_{\theta}(x|z)$, mapping a latent vector $z$ back to the data space. In practice, modeling the full probability distribution for high-dimensional data is often infeasible. Therefore, the decoder is typically implemented as a deterministic function that outputs a generated sample $\hat{x}$ given $z$, essentially representing the mean of $p_{\theta}(x|z)$.

#### c) Training and Generation
During training, the encoder and decoder are trained end-to-end. An input $x$ is encoded to the distribution $q_{\phi}(z|x)$, a latent vector $z$ is *sampled* from this distribution, and the decoder tries to reconstruct the original $x$ from $z$.

For generation, we sample $z$ from a chosen *prior* distribution $p(z)$, usually a standard Normal distribution $N(0, I)$, and feed it to the trained decoder.

#### d) The Loss Function
To ensure the learned latent space $q_{\phi}(z|x)$ resembles the desired prior $p(z)$, a second term is added to the standard reconstruction loss. The total VAE loss is:

$$
L_{VAE} = L_{rec} + \beta L_{prior}
$$

* $L_{rec}$: The reconstruction loss (e.g., Mean Squared Error between $x$ and $\hat{x}$).
* $L_{prior}$: Penalizes the divergence between the learned distribution $q_{\phi}(z|x)$ and the prior $p(z)$.
* $\beta$: A hyperparameter balancing the two loss terms.

The distance between distributions is typically measured using the Kullback-Leibler (KL) divergence:

$$
L_{prior} = D_{KL}(q_{\phi}(z|x) || p(z))
$$

The general definition is $D_{KL}(P||Q) = \mathbb{E}_{x \sim P}[-\log(\frac{Q(x)}{P(x)})]$. For our VAE case:

$$
D_{KL}(q_{\phi}(z|x)||p(z))=\mathbb{E}_{z\sim q(z|x)}[-\log(\frac{p(z)}{q_{\phi}(z|x)})]
$$

### 2. KL Divergence Derivation (for $p(z) = N(0,1)$ and $q_{\phi}(z|x) = N(\mu, \sigma^2)$)

Let's derive the KL divergence term for a single latent dimension.
Given:
$q(z|x)=N(\mu,\sigma^{2})=\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(z-\mu)^{2}}{2\sigma^{2}}}$ 
$p(z)=N(0,1)=\frac{1}{\sqrt{2\pi}}e^{-\frac{z^{2}}{2}}$ 

And properties of expectation for $z \sim N(\mu, \sigma^2)$:
$\mathbb{E}[z] = \mu$ 
$\mathbb{E}[z^2] = \mu^2 + \sigma^2$ 
$\mathbb{E}[ax+b]=a\mathbb{E}[x]+b$ 

Derivation:
$D_{KL}=\mathbb{E}[-log(\frac{p(z)}{q(z|x)})]$ 
$D_{KL}=\mathbb{E}[\log q(z|x) - \log p(z)]$ 
$D_{KL}=\mathbb{E}[\log(\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(z-\mu)^{2}}{2\sigma^{2}}}) - \log(\frac{1}{\sqrt{2\pi}}e^{-\frac{z^{2}}{2}})]$
$D_{KL}=\mathbb{E}[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2} - (-\frac{1}{2}\log(2\pi) - \frac{z^2}{2})]$
$D_{KL}=\mathbb{E}[-\frac{1}{2}\log(\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2} + \frac{z^2}{2}]$
$D_{KL}=-\frac{1}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\mathbb{E}[(z-\mu)^2] + \frac{1}{2}\mathbb{E}[z^2]$
$D_{KL}=-\frac{1}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}(\sigma^2) + \frac{1}{2}(\mu^2 + \sigma^2)$ 
$D_{KL}=-\frac{1}{2}\log(\sigma^2) - \frac{1}{2} + \frac{1}{2}\mu^2 + \frac{1}{2}\sigma^2$
$D_{KL}=\frac{1}{2}(\mu^2 + \sigma^2 - 1 - \log(\sigma^2))$ 

Note: In implementation, we often work with `logvar` = $\log(\sigma^2)$. Substituting $\sigma^2 = \exp(\text{logvar})$:
$D_{KL}=\frac{1}{2}(\mu^2 + \exp(\text{logvar}) - 1 - \text{logvar})$

### 3. Implementation Details (Airfoil Example)

The goal was to generate 2D airfoil shapes represented by 200 y-coordinates for fixed x-coordinates.

#### a) Encoder
The encoder is an MLP that outputs parameters for the latent distribution. The final layer outputs $2 \times \text{latent\_size}$ values, which are then split into `mu` and `logvar`.

```python
class Encoder(nn.Module):
    # Probabilistic encoder. Output: mu and logvar of q(z|x).
    def __init__(self, input_size: int, latent_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, 2 * latent_size) 
        )

    def forward(self, y: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # y: shape (batch_size, input_size)
        # returns mu, logvar: both shape (batch_size, latent_size)
        out = self.net(y) 
        mu, logvar = torch.chunk(out, chunks=2, dim=1) # Split output into mu and logvar
        return mu, logvar 
```

#### b) Decoder
The decoder is another MLP mapping from the latent space $z$ back to the data space (200 y-coordinates).

```python
class Decoder(nn.Module):
    # Treat this as a normal decoder.
    def __init__(self, latent_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, output_size) 
        )

    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        # z: shape (batch_size, latent_size)
        # returns y_hat: shape (batch_size, output_size)
        return self.net(z) 
```

#### c) VAE Model
Combines the encoder and decoder. The forward pass includes the *reparameterization trick* to sample $z$ in a differentiable way.

```python
class VAE(nn.Module):
    def __init__(self, input_size: int, latent_size: int, hidden_size: int):
        super().__init__()
        self.latent_size = latent_size 
        self.encoder = Encoder(input_size, latent_size, hidden_size) 
        self.decoder = Decoder(latent_size, input_size, hidden_size) 

    def forward(self, y: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # y: shape (batch_size, input_size)
        # returns y_hat, mu, logvar
        mu, logvar = self.encoder(y)  # Get distribution parameters

        # Reparameterization trick: z = mu + std * epsilon
        std = torch.exp(0.5 * logvar) # Calculate std from logvar
        eps = torch.randn_like(std) # Sample epsilon from N(0, I)
        z = mu + eps * std # Sample z

        y_hat = self.decoder(z) # Decode z to reconstruct y
        return y_hat, mu, logvar 

    @torch.inference_mode()
    def generate(self, n_samples: int, seed: int = 0, device: str = 'cpu') -> torch.FloatTensor:
        # shape (n_samples, input_size)
        torch.manual_seed(seed) 
        self.decoder.eval().to(device) 
        # Sample z directly from the prior N(0, I)
        z = torch.randn(n_samples, self.latent_size, device=device, dtype=torch.float) 
        y_hat = self.decoder(z).cpu()  # Decode sampled z
        return y_hat 

```

#### d) KL Divergence Implementation
The derived formula is implemented to calculate $L_{prior}$. It's averaged over batch and latent dimensions.

```python
def D_KL(mu: torch.FloatTensor, logvar: torch.FloatTensor) -> torch.FloatTensor:
    # mu: shape (batch_size, latent_size) 
    # logvar: shape (batch_size, latent_size) 
    # returns: scalar tensor
    # Elementwise KL divergence: 0.5 * (exp(logvar) + mu^2 - 1.0 - logvar)
    kl_elementwise = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar) 
    # Average over batch and latent dimensions
    return kl_elementwise.mean() 
```

#### e) Training Loop
The training loop calculates reconstruction loss ($L_{rec}$, e.g., MSE) and prior loss ($L_{prior}$ using `D_KL`), combines them using $\beta$, and performs backpropagation.

```python
# Inside the train_VAE function loop:
# y: input batch
y_hat, mu, logvar = model(y) # Forward pass 
rec_loss = rec_loss_fn(y_hat, y) # Reconstruction Loss 
prior_loss = D_KL(mu, logvar) # Prior Loss (KL Divergence) 
loss = rec_loss + beta * prior_loss # Total VAE Loss 

loss.backward() # Backpropagation 
optimizer.step() # Update weights
# ... (logging, scheduler steps, etc.) 
```

### 4. Hyperparameter Tuning (Especially $\beta$)

Finding a good set of hyperparameters is crucial, particularly the value of $\beta$. The assignment involved tuning parameters like `hidden_size`, `beta`, optimizer settings (`lr`, `weight_decay`), learning rate scheduler, and training iterations (`n_iters`, `batch_size`). Qualitative evaluation of generated samples (smoothness, diversity) was used to assess model performance. The provided example used `beta=0.0025`, `hidden_size=256`, Adam optimizer, StepLR scheduler, and 5000 iterations.

## Learning Reflections

This VAE implementation project highlighted several key concepts:

1.  **The Role of $\beta$:** This hyperparameter is critical for balancing reconstruction quality and latent space regularization.
    * **If $\beta$ is too large:** The $L_{prior}$ term dominates. The model focuses heavily on matching the prior $p(z)$, potentially forcing the latent space to be very close to $N(0, I)$. This can lead to poor reconstruction quality as the model might ignore input details ("posterior collapse" in extreme cases). Generated samples might seem diverse initially but could become overly generic.
    * **If $\beta$ is too small:** The $L_{rec}$ term dominates, making the VAE behave more like a standard autoencoder. Reconstruction quality for training data might be high. However, the latent space might not follow the prior $p(z)$ well, becoming discontinuous or overfitting to the training data. Sampling from the prior $p(z)$ during generation could then produce unrealistic or low-diversity results, as the decoder hasn't learned to generate from those areas of the latent space effectively.

2.  **Reparameterization Trick:** Understanding how sampling from the encoder's output distribution ($N(\mu, \sigma^2)$) can be done differentiably ($z = \mu + \sigma \times \epsilon$, where $\epsilon \sim N(0,1)$) is fundamental. This allows gradients to flow back through the sampling process during training.

3.  **Qualitative Evaluation:** Assessing generative models often requires looking at the generated samples. Metrics like loss curves are important, but visual inspection helps determine if the model produces diverse, realistic, and high-quality outputs (e.g., smooth vs. noisy airfoils, varied shapes vs. mode collapse). The example plots show the learning curves and generated airfoil samples.

4.  **Probabilistic Encoding:** Grasping the shift from a deterministic encoder output (in standard AE) to a probabilistic one (in VAE) is key. The encoder learns a distribution for each input, captured by $\mu$ and $\sigma^2$ (or `logvar`).

Overall, training VAEs involves a careful balancing act, particularly influenced by $\beta$, to achieve both good reconstruction and a well-structured latent space suitable for generating novel, high-quality data.

```