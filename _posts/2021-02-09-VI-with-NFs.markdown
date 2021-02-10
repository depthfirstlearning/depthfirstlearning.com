---
layout: post
title:  "Variational Inference with Normalizing Flows"
date:   2021-02-09 6:00:00 -0400
categories: neural-nets,variational-inference,normalizing-flows
author: 'kroon' 
blurb: "Large-scale neural models using amortized variational inference, such as the variational auto-encoder, typically rely on simple variational families.
On the other hand, normalizing flows provide a bridge between simple base distributions and more complex distributions of interest.
The paper this guide targets shows how to use normalizing flows to enrich the variational family for amortized variational inference."
feedback: true
---

[Editor’s Note: This class was a part of the 2019 DFL [Jane Street](https://www.janestreet.com/) Fellowship.]

Firstly, a huge thank-you to the participants in the study group that led to this guide, for their enthusiastic participation, interesting perspectives and insights, and useful feedback and contributions: Scott Cameron, Jean Michel Sarr, Suvarna Kadam, James Allingham, Bharati Srinivasan, Lood van Niekerk, and Witold Szejgis.

Thank you too to the Depth First Learning team for bringing me on board, and especially to Avital Oliver for helping get things started, keeping them on the rails, organizing guests for study group sessions, and gently but insistently nudging me to wrap things up after the study group had concluded.

Finally, thank you to Laurent Dinh and Rianne van den Berg for sitting in on our discussion sessions and sharing their inputs, and to them, Avital, and the study group members for their feedback on and contributions to various drafts of this material.

<div class="deps-graph">
<iframe class="deps" src="/assets/VI-with-NFs.svg" width="200"></iframe>
<div>Concept dependency graph. Click to navigate.</div>
</div>

# Why

Variational inference forms a cornerstone of large-scale Bayesian inference.
Large-scale neural architectures making use of variational inference have been enabled by approaches allowing computationally and statistically efficient approximate gradient-based techniques for the optimization required by variational inference - the prototypical resulting model is the variational autoencoder.

A complementary objective to efficient variational inference in a given variational family, is maintaining efficiency while allowing a richer variational family of approximate posteriors.
Normalizing flows are an elegant approach to representing complex densities as transformations from a simple density.

This curriculum develops key concepts in inference and variational inference, leading up to the variational autoencoder, and considers the relevant computational requirements for tackling certain tasks with normalizing flows.  While it provides good background for studying a variety of papers on VI and generative modeling, the key focus of the curriculum is the paper [Variational inference with normalizing flows](https://arxiv.org/pdf/1505.05770), which uses normalizing flows to enrich the representation used for the approximate posterior in amortized variational inference.

<br />

# Outline
The paper that we are working towards combines two key ideas: (1) amortized variational inference, and (2) normalizing flows.

We first introducing the challenge of Bayesian inference in latent variable models (Section 1), then explain variational inference (VI) as an approach for approximate inference (Section 2).
In Section 3, we develop some key ideas from the past decade extending the range of problems and problem sizes where VI can be applied.
These ideas are then combined with the idea of an inference network to develop amortized VI, showcased by the variational autoencoder (VAE), in Section 4.

Normalizing flows (NFs) are a modelling approach which represent a density of interest by a sequence of invertible transformations from a reference distribution, for example a standard Gaussian.
NFs can enable one to model a rich class of distributions by specifying parameters for these transformations.
We introduce the key ideas of NFs in Section 5, and then move on to the main paper (Section 6), which leverages NFs to improve the richness of the family of approximate latent distributions used in amortized VI.

_A [Google Doc containing an expanded version of this curriculum](https://docs.google.com/document/d/1a8WH0D5ZCCeiIus119ROigVgd4t8p2RwZLbMk6Zp3yc/edit?usp=sharing) is also available.  It contains more information on assumed prerequisites, additional rationale for and commentary on various assigned readings, links to supporting material to help mastering the required reading, a couple of extra exercises that did not make the final curriculum, and scribe notes from the group discussion sessions._

# 1 Bayesian inference and latent variable models

**Synopsis**: This part's material covers some general background from probability theory, including *Bayes rule*.  With this background, students should be able to formulate a probabilistic model and understand the *inference and learning* problems.  Of particular interest in this course are *latent variable models*, where the model includes variables which are never observed (and are arguably only modelling artifacts).
In some special cases, *Bayesian inference* (using Bayes rule to update beliefs about variables based on observations) leads to tractable posteriors for the variables, where we can conveniently calculate expectations as required for further inference or decision-making.
Many models make use of *exponential families* of distributions to obtain tractable posteriors through a property called *conjugacy*.
In most practical cases of interest, however, the posterior will be more complicated than we can deal with exactly.
*Monte Carlo methods* based on *sampling* from the posterior are one approach for dealing with this.  Our focus in the coming parts, however, will be another major approach, *variational inference*.

**Objectives**:
After this part, you should:
- be able to apply the change of variable formula to calculate the distribution of a transformation of a random variable;
- understand the tasks of inference of variables and learning of parameters in a probabilistic model;
- be comfortable with manipulating the core quantities used in Bayes rule (prior, likelihood, evidence, posterior) and key information-theoretic quantities;
- be able to convert between a Bayes network representation and a factored joint distribution;
- understand the principle of conjugate priors and the relevance of the exponential family w.r.t conjugacy; and
- be aware of sampling techniques and how a sampler can be used to evaluate a posterior expectation.

**Topics**:

- Important concepts in probability and information theory (Bayes rule, latent variables, multivariate change of variables formula, Kullback-Leibler divergence and entropy)
- (Exact) Bayesian inference, conjugacy, and the exponential family
- Introduction to approximate inference

**Required Reading** 

Important concepts in probability and information theory:
- Ian Goodfellow et al., [Deep Learning, the following portions of Chapter 3](https://www.deeplearningbook.org/contents/prob.html): Sections 3.9.6 and 3.11--3.13 (excluding the portion in Section 3.12 on measure theory). [Note that the content of Chapter 3 before Section 3.9.3 is assumed background knowledge.]
 
(Exact) Bayesian inference, conjugacy, and the exponential family:
- David MacKay, [Information Theory, Inference, and Learning Algorithms](http://www.inference.org.uk/itprnn/book.pdf), Section 3.2.
- David Blei, [The Exponential Family](http://www.cs.columbia.edu/~blei/fogm/2015F/notes/exponential-family.pdf), sections titled "Definition" and "Conjugacy" (until Formula (49), before the subsection "Posterior predictive distribution")

Introduction to approximate inference:
- Dimitris G. Tzikas, Aristidis C. Likas, and Nikolaos P. Galatsanos, [The Variational Approximation for Bayesian Inference](http://www.cs.uoi.gr/~arly/papers/SPM08.pdf), until the end of the section titled "An alternative view of the EM algorithm".
- David MacKay, [Information Theory, Inference, and Learning Algorithms](http://www.inference.org.uk/itprnn/book.pdf), Section 29.1 (excluding the portion on uniform sampling).

**Additional Reading**:


1. The rest of David Blei, [The Exponential Family](http://www.cs.columbia.edu/~blei/fogm/2015F/notes/exponential-family.pdf)
2. More of Chapters 29 and 30 of David MacKay, [Information Theory, Inference, and Learning Algorithms](http://www.inference.org.uk/itprnn/book.pdf)

**Questions**:

1. *Density transformation formula*. Use the formula for transformation of variables to derive the density of the multivariate Gaussian distribution from an *invertible* linear transformation of the standard multivariate Gaussian distribution $$\mathcal{N}(\mathbf{0}, \mathbf{I})$$.
 
2. *Belief networks*. Complete part 1 of Exercise 35 at the end of Chapter 3 in [this PDF pre-print version](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/090310.pdf) of David Barber's "Bayesian Reasoning and Machine Learning".

3. *Posterior inference via conjugacy*. Suppose you have data $$D$$ consisting of i.i.d. observations $$x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2 =1)$$.

    a. Specify the likelihood of the observations $$p(D; \mu)$$.

    b. Derive the maximum likelihood estimate of $$\mu$$.

    c. Suppose we model our uncertainty about the mean with $$\mu \sim \mathcal{N}(0, \sigma_{\mu}^2 = 1)$$. Derive the posterior distribution by making use of conjugacy, and use this to obtain the MAP estimate of $$\mu$$.

4. Prove that the KL divergence $$\mathrm{KL}(q \mid p)$$ is nonnegative.
    <details><summary>Hint</summary>
    Apply the bound $$\log t \leq t-1$$ to $$t=p(x)/q(x)$$.
    </details>

5. *KL divergence for simple normal distributions*. Show that

    $$\text{KL}\left(\mathcal{N}\left((\mu_1, \ldots, \mu_k)^\mathsf{T}, \operatorname{diag} (\sigma_1^2, \ldots, \sigma_k^2)\right) \parallel \mathcal{N}\left(\mathbf{0}, \mathbf{I}\right)\right) = {1 \over 2} \sum_{i=1}^k (\sigma_i^2 + \mu_i^2 - \ln(\sigma_i^2) - 1) \enspace .$$

6. Derive Equation (7) in [The Variational Approximation for Bayesian Inference](http://www.cs.uoi.gr/~arly/papers/SPM08.pdf).


<details><summary>Solutions</summary>
<b>Solutions to these exercises can be found <a href="https://colab.research.google.com/drive/1EGEREIdV0RxF27KKnqC2HXDu3rdxdB7A" target="_blank">here</a></b>
</details>
<br />

# 2 Introduction to Variational Inference (VI)

**Synopsis**: In practice, Bayesian inference yields posteriors which do not have convenient forms.  The traditional approach to calculating or estimating posteriors or posterior expectations is to use Monte Carlo methods based on posterior *sampling*.  These are *asymptotically exact but computationally intensive*, particularly in high dimensions.  An alternative approach is *variational inference (VI)*, which *trades off exactness for tractability*.  In this part, we introduce the core ideas of VI approaches in the context of *mean field VI*.  The VI approach loses exactness by approximating the true posterior with a representative from a *variational family*.  There is a tradeoff between richness of the approximation family (impacting the resulting estimate quality) and the tractability of the VI scheme. *Mean-field factorization assumptions* on the variational family yield an approach for optimizing the variational parameters through *coordinate ascent*. Much of the rest of this curriculum will focus on trying to improve the behaviour of VI in terms of scalability, broadness of applicability, and accuracy (by using more sophisticated variational families). 

**Objectives**:
After this part, you should:
- have an idea of the relationships between the (variational) EM algorithm and (variational) Bayesian inference;
- be able to describe coordinate ascent variational inference (CAVI), and explain its shortcomings in terms of scalability to large models; and
- understand and follow the steps required in deriving a CAVI algorithm for a conditionally conjugate model.

**Topics**:

- Variational expectation-maximization
- Variational inference
- Mean-field variational inference
- Co-ordinate ascent variational inference

**Required Reading**:

Variational expectation-maximization:

- Dimitris G. Tzikas, Aristidis C. Likas, and Nikolaos P. Galatsanos, [The Variational Approximation for Bayesian Inference](http://www.cs.uoi.gr/~arly/papers/SPM08.pdf), the section titled "The Variational EM framework".

Variational inference:

- David Blei, Alp Kucukelbir, and Jon D. McAuliffe, [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf), until the end of Section 4.2.

**Additional Reading**:

1. The rest of Dimitris G. Tzikas, Aristidis C. Likas, and Nikolaos P. Galatsanos, [The Variational Approximation for Bayesian Inference](http://www.cs.uoi.gr/~arly/papers/SPM08.pdf).

**Questions**:

1. *Forward vs reverse KL*. Consider the univariate distribution $$P$$ formed by an equal mixture of unit variance Gaussians with means at -5 and 5. Think about how a Gaussian distribution $$Q$$ would look that minimizes (i) $$\mathrm{KL}(Q\|P)$$ and (ii) $$\mathrm{KL}(P\|Q)$$. Explain your answers.  Which approximation behaviour do you think is preferable for posterior inference, and why? Which approach do you think will be more tractable, and why? __Additional__: implement the required KL calculations - sampling or other tricks will be required - and numerically optimize to fit the optimal Q in each case.
2. *EM vs. variational inference*. Describe how Bayesian inference of latent variables and unknown parameters can be seen as a special case of the EM algorithm.  Extend this analogy to compare coordinate ascent variational inference to mean-field variational EM.
3. *ELBO as a KL divergence?*. Looking at Equation (13) of [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf), it seems one can write $$\mathrm{ELBO}(q) = -\mathrm{KL}(q(\mathbf{z})\|p(\mathbf{z},\mathbf{x}))$$.  Explain what the problem is with this. (Note that this is also essentially done in Equation 15 of [The Variational Approximation for Bayesian Inference](http://www.cs.uoi.gr/~arly/papers/SPM08.pdf).) __Warning__: some would argue this is just nitpicking about a technicality!
4. *ELBO derivations*. Show that the expression $$\mathbb{E}[\log p(x_i \mid c_i,\mathbf{\mu}; \phi_i, \mathbf{m}, \mathbf{s}^2)]$$ in Equation (21) of [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf) equals $$-\frac{1}{2}[\log 2\pi + \sum_{k=1}^K \phi_{ik}(x_i^2 +m_k^2 + s_k^2 -2x_i m_k)]$$.
5. What do you think is the biggest challenge to scalability of CAVI?
6. What is the benefit of your model having complete conditionals in the exponential family if you would like to apply CAVI? 
7. Calculate the rest of the terms in the ELBO of Equation (21) in [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf), and verify the CAVI update equations by setting the components of the ELBO gradient to zero. (__Additional__)
8. Implement CAVI for the example in Sections 2-3 of [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf) using PyTorch or a similar package. Think about how to visualize the behaviour of the algorithm and/or its results.  If you have done the previous exercise, use a threshold on the relative change in the ELBO to control when to terminate; otherwise you can monitor changes in the variational parameters, or the log-predictive density on a hold-out set. If you have implemented the ELBO, compare the behaviour of CAVI to directly optimizing the ELBO by gradient descent. (__Additional__)

<details><summary>Solutions</summary>
<b>Solutions to these exercises can be found <a href="https://colab.research.google.com/drive/1WJSnWguTRCHlJm3TFdpz5bJ9YhBVx-Xu" target="_blank">here</a></b>
</details>
<br />


# 3 Doubly stochastic estimation: VI by Monte-Carlo mini-batch gradient estimation

**Synopsis**: In this part we consider two techniques used to address *major limitations on the applicability and scalability of CAVI*.
The first challenge (to scalability) is that each global parameter update *requires a full pass* through the complete data set, which is problematic for very large data sets.
This is resolved through stochastic variational inference, which uses the same ideas from stochastic approximation that enable the use of *stochastic gradient descent* in training other machine learning models.
The second challenge (to applicability) is that the updates by CAVI need to be *determined manually* for each model.
This is addressed through *black-box variational inference* (BBVI), which uses Monte Carlo estimates to replace the manual derivation.
Since the naive Monte Carlo estimator has very high variance, *variance reduction techniques* for Monte Carlo estimation must be applied to make this approach effective.
When BBVI is combined with SVI by using mini-batches for the gradient estimation, we speak of *doubly stochastic estimation*.

**Objectives**:
After this part, you should:
- be aware of the concept of natural gradient;
- be aware of the Robbins-Munro conditions for stochastic optimization;
- understand how SVI uses mini-batch gradients to efficiently scale up CAVI;
- understand the score function Monte Carlo gradient estimator of the ELBO;
- be aware of what is required to apply BBVI and doubly stochastic estimation;
- be aware of Rao-Blackwellization/conditioning and control variates as variance reduction techniques in Monte Carlo estimation; and
- be able to explain the impact of doubly stochastic estimation on scalability, and what issues further limit scalability.

**Topics**:

- Fisher information and natural gradient
- Stochastic variational inference
- Variance reduction methods for Monte Carlo estimation
- Black box variational inference

**Required Reading**:

Fisher information and natural gradient:

- Andrew Miller, [Natural Gradients and Stochastic Variational Inference](http://andymiller.github.io/2016/10/02/natural_gradient_bbvi.html), until the start of the section “Gaussian example”.

Stochastic variational inference:

- David Blei, Alp Kucukelbir, and Jon D. McAuliffe, [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf), Section 4.3.

Variance reduction methods for Monte Carlo estimation:

- Martin Haugh, [Simulation Efficiency and an Introduction to Variance Reduction Methods](http://www.columbia.edu/~mh2078/MonteCarlo/MCS_Var_Red_Basic.pdf).  Read from the beginning until the end of Example 1 on page 4, and then Section 4 until the end of Example 9 on page 12.

Black box variational inference:

- Rajesh Ranganath, Sean Gerrish, and David M. Blei, [Black Box Variational Inference](https://arxiv.org/pdf/1401.0118.pdf).
Section 5 is optional, but note the dramatic effect of the variance reduction techniques shown in Figure 2.
 (Also check the derivation of the ELBO gradient in Equation 2 presented in Section 7, but note that there is a missing gradient sign in the expectation in the line where Equation (13) is labelled.)  

**Additional Reading**:

1. Matthew D. Hoffman, David M. Blei, Chong Wang, and John Paisley, [Stochastic Variational Inference](http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf). (The most relevant portion is until the end of Section 2, with Section 3 discussing applications to two topic models: latent Dirichlet allocation and the hierarchical Dirichlet process.)

**Questions**:

1. Extend your CAVI implementation from the previous section to VI using natural gradient descent, and consider the impact of the minibatch size on the convergence time in terms of number of examples seen.  Use the autodifferentiation capability of PyTorch to perform stochastic gradient descent on the ELBO (i.e. not following the natural gradient), and compare the performance of this to the previous approach. (__Additional__)

2. *The score function*. For a parameterized distribution $$p(x; \theta)$$, the score is defined as the gradient (w.r.t. $$\theta$$) of the log-density, and the covariance matrix of the score under this distribution is called the Fisher information matrix.

    a. Derive the score function for a univariate Gaussian.

    b. Show that the expected score (w.r.t.$$p$$) is zero.

3. *Fisher as the Hessian of relative entropy*. Assuming $$\log q_{\lambda}$$ is twice differentiable, one has that the entries of the Fisher can also be written as $$[F_\lambda]_{ij} = -\mathbb{E}_{x \sim q_{\lambda}}[\frac{\partial^2}{\partial \lambda_i \partial \lambda_j} \log q_{\lambda} (x)]$$. (_Additional_: derive this.) Use this formulation to show that the Fisher is the Hessian (w.r.t. $$\lambda^{\prime}$$) of the KL divergence $$\mathrm{KL}(q_\lambda \mid q_{\lambda^\prime})$$  at $$\lambda^\prime = \lambda$$.

4. *Fisher for exponential families*. Given that $$F_\eta = - \mathbb{E}_{x} \nabla_\eta^2 \log p(x \mid \eta)$$ (the matrix form of the representation in the previous exercise), show that the Fisher equals the Hessian of the log normalizer (\nabla_\eta^2 a(\eta)) when $$p(x \mid \eta)$$ is from an exponential family.

5. *Score function gradient estimation, a.k.a. the log-derivative trick*. Consider the problem of using gradient descent to find the mean of a unit variance Gaussian with minimum second moment $$\mathbb{E}(X^2)$$.
We thus seek the value of $$\nabla_{\mu} \mathbb{E}_{N(\mu,1)}(X^2)$$ at a candidate value $$\mu_0$$.
Exchange the order of differentiation and integration, and then use the score function to obtain an expression for this derivative that is an expectation amenable to Monte Carlo estimation.
Note how the derivation of the ELBO gradient for BBVI used this approach, along with the expectation of the score being zero.
(This idea is essentially the key idea enabling BBVI, so it is probably the most important of this part's exercises to get your head around.)

6. *Incremental SVI*. Suppose you have already fit a model to a huge data set with doubly stochastic VI, and then receive new data.  How would you go about obtaining the estimated posterior over the latent variables for the new data?  How would you go about updating the model to incorporate the new data?

7. *Law of total variance*. Derive the formula in Equation 5 on page 10 of [Simulation Efficiency and an Introduction to Variance Reduction Methods](http://www.columbia.edu/~mh2078/MonteCarlo/MCS_Var_Red_Basic.pdf).
    <details><summary>Hint</summary>
    Begin by writing the variance as a difference in the traditional way, and applying the law of total expectation (the formula above Equation 5) to each term.  
    From there you should be able to manipulate expectations and variances w.r.t. $$Z$$ and $$X|Z$$ to get the required expression - i.e. there should be no need to write these out as integrals.
    </details>

8. *Efficacy of conditional Monte Carlo*. Answer Exercise 2 on page 11 of [Simulation Efficiency and an Introduction to Variance Reduction Methods](http://www.columbia.edu/~mh2078/MonteCarlo/MCS_Var_Red_Basic.pdf)

9. Implement naive Monte Carlo sampling as well as using the control variate and conditioning methods as per Examples 1 and 9 in [Simulation Efficiency and an Introduction to Variance Reduction Methods](http://www.columbia.edu/~mh2078/MonteCarlo/MCS_Var_Red__Basic.pdf) to see the variance reduction effect of these strategies. (__Additional__)

10. Consider mean-field variational inference of an hierarchical Bayesian model as in Equation (12) of [Black Box Variational Inference](https://arxiv.org/pdf/1401.0118.pdf).
Note that $$\beta$$ appears in all terms of the log-joint, while any specific $$z_i$$ only appears in two terms.  What effect does this have when one calculates Rao-Blackwellized estimators of the gradient component for the variational parameters corresponding to $$\beta$$ vs. those for the $$z_i$$ according to Equation (6) of the paper?  How does incorporating stochastic estimation via minibatching/observation sampling make these updates more efficient?  (Focus on the overall effect, equations are not required!)

11. Implement BBVI for the Bayesian Gaussian mixture model, and compare its performance to the previous techniques (both with and without variance reduction techniques). (__Additional__)

<details><summary>Solutions</summary>
<b>Solutions to these exercises can be found <a href="https://colab.research.google.com/drive/1XKk95WUSzWXYd9idD2ei6A3K3EdnMVj7" target="_blank">here</a></b>
</details>
<br />

# 4 Inference networks and amortized VI

**Synopsis**: This part presents developments in VI allowing *further scalability* as well as use in *online settings*.
Traditional VI analyses all the data together, and individually optimizes the latent variables corresponding to each observation.
This means that *new observations require refitting* the entire model.
A way to bypass this is to model the transformation from an observation to its posterior distribution using an *inference network* or recognition model.
Instead of optimizing variational parameters for each observation, those *variational parameters are output by the inference network* when it is given the observation as input, and the model parameters of the inference network are trained to optimize these predictions during the learning phase.
This allows direct, efficient, prediction of the latent variable posterior (i.e. inference) on previously unseen samples - so-called *amortized VI*.
Previous work had trained such inference networks before, but the other development here was combining the inference and generative networks end-to-end in a neural network, and using the *evidence bound (ELBO) as a combined training objective*.
This was enabled, for continuous variables, by an alternative Monte Carlo estimator of the gradient, based on the so-called *reparameterization trick*.
The most well-known such model now is the *variational autoencoder*.

**Objectives**:
After this part, you should be comfortable with:
- explaining the reparameterization trick and what problem it tries to solve;
- understanding in principle how the reparameterization trick is implemented in machine learning libraries with auto-differentiation facilities;
- combining an inference network with a generator network, and training them end to end;
- the idea of the inference network outputting parameters describing the posterior distribution corresponding to the network input;
- the specific choice of loss function used for end-to-end training;
- the use of amortized VI for variational autoencoders and deep latent variable models; and
- discussing the scalability of such systems, and their limitations.

**Topics**:

- Inference networks
- Amortized VI
- The reparameterization trick
- Variational autoencoders

**Required Reading**:

- Diederik Kingma and Max Welling, [An Introduction to Variational Autoencoders](https://arxiv.org/pdf/1906.02691), Sections 1.7-2.8 (but you can omit Section 2.6).

**Additional Reading**:

The first two papers listed below were independent proposals of the variational autoencoder.

1. Danilo J. Rezende, Shakir Mohamed, and Daan Wierstra, [Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/pdf/1401.4082.pdf).
2. Diederik Kingma and Max Welling, [Auto-Encoding variational Bayes](https://arxiv.org/abs/1312.6114).
3. Shakir Mohamed, Mihaela Rosca, Michael Figurnov, and Andriy Mnih, [Monte Carlo Gradient Estimation in Machine Learning](https://arxiv.org/pdf/1906.10652). Studies various approaches to estimating gradients of function expectations with respect to parameters defining the distribution with Monte Carlo methods.  Properties of the score function and pathwise (i.e. reparameterization trick) gradient estimators are discussed in considerable detail.
4. James Allingham, Deep Learning Indaba Practical 3b on [Deep Generative Models](https://colab.research.google.com/drive/1QYej56ctstejvRx6uK9qTTlF44NXzqH1#forceEdit=true&offline=true&sandboxMode=true) - Colab notebook introducing VAEs and Generative Adversarial Networks (GANs).

**Questions**:

1. *Reparameterization trick*. Explain the reparameterization trick in your own words and what problem it tries to solve.

2. *Applying the reparameterization trick* Exercise 5 of Section 3 uses the score function gradient estimator.
Now use the reparameterization trick to get an alternative expression for this gradient in terms of an expectation w.r.t. a standard Gaussian distribution (i.e. zero-mean and unit variance).
Implement both estimators (the one from the previous section and the one from this section), and plot the variance of each against the number of Monte Carlo samples.
(To obtain the variances, repeatedly estimate the gradient with independent Monte Carlo samples of the relevant size.)

3. *Discrete latent variables and the reparameterization trick*. Black-box variational inference can fit models with discrete latent variables, but the VAE can not. Explain why.

4. *The ELBO as training objective*. In previous sections, we considered the situation where the generative model was known, and we focused on estimating the variational parameters by optimizing the ELBO.
In the VAE, the ELBO is used to jointly optimize the parameters of the encoder and the decoder.
Consider the decomposition of the marginal likelihood in Equation 2.8 of [An Introduction to Variational Autoencoders](https://arxiv.org/pdf/1906.02691).
 
    Suppose $$\theta$$ is held fixed, and $$\phi$$ is optimized w.r.t. the ELBO.
    This is similar to other VI approaches, except that an inference network is now used for amortized analysis.
    This has no effect on the marginal likelihood of the generative model (which should be expected, since $$\theta$$ is fixed), but makes the variational posterior better.

    Suppose now that $$\phi$$ is held fixed, and $$\theta$$  is optimized w.r.t. the ELBO.
    This may make the variational posterior less accurate. Why is it nevertheless a good idea?

    Finally, note that end-to-end optimization of the ELBO across the encoder and decoder essentially corresponds to interleaving stochastic gradient descent w.r.t. the two above steps.

5. *VAE implementation and exploration*. Complete the VAE implementation in [``vae.py``](/assets/vae.py).

    a. Note how the provided code uses the VAE to sample new images.

    b. Plot the variational parameters (means and log-variances) for a number of MNIST digits.  Do they seem to have some kind of information about the classes present in the data set? (__Additional__)

6. *Relationship to nonlinear PCA*. An earlier approach to constructing low-dimensional representations (for compression or further analysis) was nonlinear PCA.  This used a low-dimensional bottleneck layer in an autoencoder model, and then extracted the representation at this layer for the lower-dimensional representation.  Modify your VAE implementation above by ignoring the log-variances, and simply returning the predicted mean in the reparameterization step.  This corresponds to setting the variance for the latent Gaussian to zero, and the resulting model then *almost* corresponds to non-linear PCA. The final adjustment to obtain nonlinear PCA is to set the loss function to only use the reconstruction loss, and not to also penalize deviations of the variational family from the prior. (__Additional__)

    a. Compare the sampling output for nonlinear PCA and the VAE, and  contrast their suitability for sampling.

    b. Contrast nonlinear PCA and the VAE w.r.t. their suitability for compression.

<details><summary>Solutions</summary>
<b>Solutions to these exercises can be found <a href="https://colab.research.google.com/drive/1uX39PbxiMHChty8gSF5IVw_c72AhpR4y" target="_blank">here</a></b>
</details>
<br />


# 5 Normalizing Flows

**Synopsis**: There are various approaches to probabilistic modelling of complex phenomena.
In the previous parts, we have considered *variational inference for directed graphical models with latent variables*.
These models postulate meaningful latent variables and are amenable to ancestral sampling once we have fit the required conditional distributions, but a challenge for this approach is that the posterior distribution of latent variables may exhibit complex dependencies, which may not be well modeled by the variational family.
In this part, we consider a different approach to probabilistic modelling which dispenses with the latent variables, and directly models the *data density as a sequence of parameterized invertible transformations* starting from a (simple) base density.
Such a sequence of transformations (from a complicated to a simple density) is called a *normalizing flow*.
A key aspect of this approach is to ensure that applying the transformations and obtaining their gradients are *computationally efficient* to allow efficient training and sampling.
Thus, normalizing flows in the machine learning literature usually refers to an approach to parameterizing a fairly complex distribution as a sequential transformation of a simple one with some attractive computation properties.
In the setting we consider here, a *single* flow is fitted directly to the (often high-dimensional) data.
The next section will combine these modelling approaches by using normalizing flows to refine the posteriors in amortized VI.

**Objectives**:
After this part, you should:
- be comfortable with the change of variable formula and the use of the Jacobian when transforming nonlinear densities;
- understand the distinction between inference and sampling in flow models, and how inference enables density estimation;
- know which operations need to be efficient for efficient inference vs efficient sampling in flow models; and
- understand how the coupling layers used in NICE enable both efficient inference and efficient sampling.

**Topics**:

- Normalizing flows
- Efficient sampling vs. efficient inference with normalizing flows

**Required Reading**:

Normalizing Flows:

- Ivan Kobyzev, Simon J.D. Prince, and Marcus A. Brubaker, [Normalizing Flows: Introduction and Ideas](https://arxiv.org/pdf/1908.09257), until midway through Section 3.2.1, “Triangular”. (Skip Section 2.1.1.)
 This review introduces the foundational concepts of normalizing flows, their main forms of application, and the properties we desire for efficient computation with normalizing flows.

Efficient sampling vs. efficient inference with normalizing flows:

- Laurent Dinh, David Krueger, and Yoshua Bengio, [NICE: Non-linear independent components estimation](https://arxiv.org/pdf/1410.8516.pdf). (Feel free to skim over portions in the Related Methods section that you are not familiar with.)

**Additional Reading**:

1. Eric Jang, [Tips for Training Likelihood Models](https://blog.evjang.com/2019/07/likelihood-model-tips.html).

2. Eric Jang, [Normalizing Flows Tutorial, Part 2: Modern Normalizing Flows](https://blog.evjang.com/2018/01/nf2.html).

3. Lilian Weng, [Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html).

4. Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio, [Density Estimation using Real NVP](https://arxiv.org/pdf/1605.08803.pdf).

5. Gustavo Deco and Wilfried Brauer, [Nonlinear higher-order statistical decorrelation by volume-conserving neural architectures](https://www.sciencedirect.com/science/article/pii/089360809400108X). this is an early forerunner of normalizing flows, with proposed flows that seems to match (volume-preserving) autoregressive flows.

6. George Ho, [Autoregressive Models in Deep Learning - A Brief Survey](https://eigenfoo.xyz/deep-autoregressive-models) - an introduction to a variety of deep autoregressive networks.

**Questions**:

1. Figure 2 of [NICE: Non-linear independent components estimation](https://arxiv.org/pdf/1410.8516.pdf) labels the computation graph of a coupling layer using concepts from cryptography.  Explain why this is a suitable metaphor.

2. Consider a VAE where we use a standard isotropic Gaussian as the prior for the latent variable, and where the conditional $$p(x|z) \sim \mathcal{N}(f_\theta(z), I)$$.
Consider the following perspective on the forward pass through a VAE.
The first (encoder) phase takes as input a pair $$(x, \epsilon)$$, and outputs a pair $$(x,z)$$ - this can be seen as an affine coupling layer (*a la* NICE).
The second (decoder) phase takes as input the pair $$(x,z)$$ and outputs the pair $$\varepsilon, z)$$ (where $$\varepsilon = x - f_{\theta}(z)$$ in a sense encodes how $$x$$ might be generated from $$f_{\theta}(z)$$ with a change of variables) - this can also be seen as an affine coupling layer.
The VAE estimates its parameters by optimizing Monte Carlo estimates of the ELBO with the reparameterization trick, while the normalizing flow estimates its parameters by optimizing the data log-likelihood (assuming isotropic Gaussian priors on $$z$$ and $$\varepsilon$$).
Considering that in the above, the input data points to the normalizing flow are $$(x,\epsilon)$$ (and not just $$x$$), show/convince yourself that these two approaches to estimating the parameters are equivalent.

3. Suppose one fitted a normalizing flow with a Gaussian base density for some domain.
Consider a model using this normalizing flow as an encoder, and the inverse of the flow as a decoder.
Discuss the relationships between this model and a VAE (and nonlinear PCA, if you tackled Exercise 6 in the previous section).

4. Implement NICE in PyTorch using affine coupling layers.  Prevent the multiplicative factor in the scaling of each layer being zero by exponentiating the output of a ReLU MLP.  This approach, also used in RealNVP, removes the need for the final scaling layer in NICE. (__Additional__)

5. Use your NICE implementation from the previous question (or modify an implementation from online) to allow you to experiment with varying numbers of coupling layers while trying to model some somewhat complicated distributions. If you are doing it from scratch yourself, begin by modelling 2-D distributions, like that in the example at the bottom of [https://blog.evjang.com/2018/01/nf1.html](https://blog.evjang.com/2018/01/nf1.html), or that from [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html), before considering tackling higher-dimensional cases such as MNIST. (__Additional__)

6. Consider Table 1 and Figure 3 of [Variational inference with normalizing flows](https://arxiv.org/pdf/1505.05770).
In this setting, we have the (unnormalized) target density, but we do not have samples from the density.
Thus we can not fit a normalizing flow by optimizing the data log-likelihood w.r.t. the flow parameters.
Yet Figures 3(b) and 3(c) present results for fitted flows.
Can you think of a sensible objective function to fit the parameters of a normalizing flow in this case?
    <details><summary>Hint</summary>
    A Gaussian is a flow with zero transformations - how might you fit a Gaussian to such a distribution?
    </details>

<details><summary>Solutions</summary>
<b>Solutions to these exercises can be found <a href="https://colab.research.google.com/drive/1ndyYNmPfAVQlBslmV7X0HPMTBZr1YCoN" target="_blank">here</a></b>
</details>
<br />

<br />

# 6 Normalizing flows for variational inference

**Synopsis**: We now turn to the main paper considered in this curriculum.
The techniques covered so far allow training *combined generative and inference networks* by stochastic backpropagation.
However, the posterior family was generally fairly simple to ensure scalable inference.
This paper leverages the normalizing flows considered in the previous section to transform the simple distributions whose parameters were originally output by the inference network to much more complex posterior distributions.
As before, computational efficiency of the normalizing flow is essential, but due to the way in which the flows are deployed in the VI setting, the requirements for efficiency differ somewhat from those for the normalizing flows considered above.

**Objectives**:
After this part you should:
- understand the idea of using a normalizing flow to obtain a richer family of variational posteriors;
- understand why the flow parameters should also be output by the encoder, rather than being learnt separately;
- have an appreciation for the different requirements on the flows that are tractable for direct density modelling vs. for use with variational inference; and
- understand the decomposition of the inference gap into the approximation and amortization gap, and have some intuition about the effects of the choice of variational posterior family, encoder architecture, and decoder architecture on these gaps.

**Topics**:

- Normalizing flows for variational inference
- Understanding the inference gap

**Disclaimer**:

In the reading for this part, there are a few concepts we have not yet covered - if you are not familiar with them, simply skim over the relevant portions - they are not crucial.

What you should know:

* Auxiliary variables (see this section's optional Section 3.2.1 in [An Introduction to Variational Autoencoders](https://arxiv.org/pdf/1906.02691)) are an alternative technique for adding additional latent variables to a model which allow a richer class of variational posteriors.  It can also be combined with normalizing flows.
* [Annealed importance sampling](https://arxiv.org/pdf/physics/9803008.pdf) is an approach that can be used to estimating the marginal likelihood/evidence.  The resulting estimate is with high probability a lower bound on the actual marginal likelihood.  One can also use the importance weighted autoencoder (IWAE) objective (which we skipped over in Section 2.6 of [An Introduction to Variational Autoencoders](https://arxiv.org/pdf/1906.02691)) as an estimate - this is also a lower bound, which becomes tighter as the number of samples used to calculate it increases.
* [Real NVP](https://arxiv.org/pdf/1605.08803.pdf) is an extension of NICE which incorporates various enhancements which are particularly appropriate for image data.
* [Hamiltonian Monte Carlo](https://arxiv.org/pdf/1701.02434.pdf) (HMC) is a Markov Chain Monte Carlo approach which uses the mathematics of Hamiltonian dynamics from physics to propose transitions.  Hamiltonian dynamics describe motion in terms of kinetic and potential energy.  For HMC, the potential energy corresponds to the distribution we wish to sample from, while the kinetic energy helps control how the space is explored.  If one views the dynamics in continuous time, the parameters of the potential and kinetic energy will correspond to an infinitesimal flow for the latent variables  and auxiliary latent variables , respectively.
* [Stochastic differential equations](https://en.wikipedia.org/wiki/Stochastic_differential_equation) can be used to model the evolution of a probability distribution over time.

**Required reading**:

Normalizing flows for variational inference:

*The first reading reviews what is required of the inference network, before presenting the key idea of normalizing flows for variational inference.  Pay attention to how the proposed flows keep the required operations efficient.  The second reading is the main paper for this curriculum.*

- Diederik Kingma and Max Welling, [An Introduction to Variational Autoencoders](https://arxiv.org/pdf/1906.02691), Chapter 3 until the end of Section 3.2 (with Section 3.2.1 optional).

- Danilo Rezende and Shakir Mohamed, [Variational inference with normalizing flows](https://arxiv.org/pdf/1505.05770). (Only skim Section 3.2 and other portions discussing infinitesimal flows.) [Note: Equation (20) has a missing $$\beta_t$$ coefficient in the last term of the first line.]

Understanding the inference gap:

- Chris Cremer, Xuechen Li, and David Duvenaud, [Inference Suboptimality in Variational Autoencoders](https://arxiv.org/abs/1801.03558). [Note: In Equation 11, the T's in the first factor in the denominator of the log should be zeros, and there should be a product over t from 1 to T of the ensuing determinants.]

**Additional Reading/Resources**:

1. Ben Lambert, [The intuition behind the Hamiltonian Monte Carlo algorithm](https://www.youtube.com/watch?v=a-wydhEuAm0&list=PLwJRxp3blEvZ8AKMXOy0fc0cqT61GsKCG&index=69).
2. Diederik Kingma and Max Welling, [An Introduction to Variational Autoencoders](https://arxiv.org/pdf/1906.02691): The rest of Chapter 3 and Chapter 4 give an overview of further developments using amortized VI for deep generative models beyond the introduction of normalizing flows.
3. David Duvenaud’s University of Toronto course on [Differentiable Inference and Generative Models](https://www.cs.toronto.edu/~duvenaud/courses/csc2541/index.html).
4. George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan, [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/pdf/1912.02762.pdf): a review on the use of normalizing flows in modeling and inference, which came out after completion of the reading group this curriculum was based on.

**Questions**:

1. Explain why it is necessary that the flow parameters, and not just the parameters of the base density used in the flow, also be output by the inference network, rather than simply having global parameters for the flow parameters that are optimized.
    <details><summary>Hint</summary>
    How can the latter case be viewed as a regular VAE without a normalizing flow?
    </details>

2. What is the impact of having the encoder output the flow parameters on using the trained model as a generative model, i.e. for sampling new observations, compared to a VAE.

3. Reproduce figures similar to those in Figure 1 of [Variational inference with normalizing flows](https://arxiv.org/pdf/1505.05770) with your own implementation. (__Additional__)

4. Two key aims of general generative models are density estimation and sampling.
In normalizing flow models for density estimation, we need to evaluate $$p(x)$$ for any potential choice of $$x$$.
This requires that it be efficient to move from the observation space to the latent space, where the base density can be evaluated, i.e. efficient inference.
In sampling, we wish to efficiently move from the latent space to the observation space.
Requiring both of these operations be efficient constrains the choice of possible flows - in general, one must sacrifice efficiency in one of these tasks, or have an easily invertible flow (such as in NICE).
The planar and radial flows used for variational inference in the main paper are not easily invertible, but yet we can efficiently perform the sampling and density estimation that we require.

    a. Explain how this is achieved in light of which "observations" we perform density estimation on.

    b. How does this influence the choice of flows we can use for variational inference compared to those where we require general efficient density estimation?

5. Implement VI with NFs, and experiment with your implementation. (__Additional__)

6. [An Introduction to Variational Autoencoders](https://arxiv.org/pdf/1906.02691) points out that the change to $$z$$ in planar flows can be viewed as a single-hidden-layer multi-layer perceptron (MLP) with a single hidden unit, and say this "does not scale well to a high-dimensional latent space: since information goes through the single bottleneck, a long chain of transformations is required to capture high-dimensional dependencies."  One way to tackle this is to change the MLP to have more hidden units.

    a. Give the resulting modified formula for these generalized flows.

    b. Note that one can no longer use the vanilla form of the matrix determinant lemma to calculate the determinant of this generalized transformation’s Jacobian.  Fortunately, there is a [generalized matrix determinant lemma](https://en.wikipedia.org/wiki/Matrix_determinant__lemma#Generalization) which enables us to calculate the determinant. Write down the determinant, and specify the order complexity of calculating it in terms of the number of hidden units.  (As with planar flows, not all such flows will be invertible.
[Sylvester normalizing flows](https://arxiv.org/pdf/1803.05649.pdf) arise as special forms of the above transformations where one obtains invertibility based on specific assumed forms for the weight matrices in the MLP - note that these forms also need to be maintained throughout training.) (__Additional__)

7. Inequality 12 of [Inference Suboptimality in Variational Autoencoders](https://arxiv.org/abs/1801.03558) gives the IWAE lower bound on the marginal likelihood.
Derive this result by using Jensen's inequality after using $$q(z|x)$$ as a proposal distribution for importance sampling from $$p(z|x)$$. (If you are not familiar with importance sampling, the relevant formula (with $$q$$ as proposal for $$p$$) is the second one on [this page](https://towardsdatascience.com/importance-sampling-introduction-e76b2c32e744).) (__Additional__)

8. How do you think the authors might have gotten the “true posteriors” in Figure 2 of [Inference Suboptimality in Variational Autoencoders](https://arxiv.org/abs/1801.03558)?

9. Try to explain in your own words the issue of encoder overfitting discussed in Section 5.5.1 [Inference Suboptimality in Variational Autoencoders](https://arxiv.org/abs/1801.03558), and when you should prefer using flows to increase the complexity of the variational approximation to increasing the expressiveness of the encoder.

<details><summary>Solutions</summary>
<b>Solutions to these exercises can be found <a href="https://colab.research.google.com/drive/1CiZcHcshztxqVgxx1qk_e_1oN2fnnUNg" target="_blank">here</a></b>
</details>
<br />


<br />
