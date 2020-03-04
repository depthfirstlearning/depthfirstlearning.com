---
layout: post
title:  "Resurrecting the sigmoid: Theory and practice"
date:   2020-01-01 6:00:00 -0400
categories: neural-nets
authors: ['piyush', 'vinay', 'riley'] 
blurb: "With the success of deep networks across task ranging from vision to
language, it is important to understand how to properly trian very deep neural
networks with gradient-based methods. This paper studies, from a rigorous theoretical perspective, which combinations of network weight initializations and network activation functions can result in deep networks which are trainable."
feedback: true
---

This guide would not have been possible without the help and feedback from many people. 

Special thanks to Yasaman Bahri for her useful contributions to this guide, her feedback, support, and mentoring.

Thank you to Kumar Agrawal, Sam Schoenholz, and Jeffrey Pennington for their valuable input and guidance.

Finally, thank you to our students Chris Akers, Brian Friedenberg, Sajel Shah, Vincent Su, Witold Szejgis, for their curiosity and commitment to the course material, and their useful feedback on the curriculum.

<div class="deps-graph">
<iframe class="deps" src="/assets/nodes-deps.svg" width="200"></iframe>
<div>Insert Image.</div>
</div>

# Why

As deep networks continue to make progress in a variety of tasks such as vision and language processing, it is important to understand how to properly train very deep neural networks with gradient-based methods.  This paper studies, from a rigorous theoretical perspective, which combinations of network weight initializations and network activation functions can result in deep networks which are trainable.  The analysis framework used is applicable to more general network architectures, including deep convolutional neural networks which are state-of-the-art in image classification tasks.  

In this currriculum, we will go through all the background topics necessary to understand the calculations in the paper.  By the end, you will have an understanding of analysis techniques used to study the dynamics of signal propagation in very wide neural networks and be able to perform some simple calculations using random matrix theory.  

<br />

# General resources:

The two foundations on which this paper is based are: (1) random matrix theory (RMT), and (2) 'mean-field' analysis of signal propagation in wide neural networks.  The first resource below is a friendly introduction to RMT, while the second and third are the papers in which the mean-field analysis for deep neural networks was developed.  These are good resources to return to throughout the course, though they cover much more than we need to understand the current paper.  The deep learning book of Goodfellow et al. is a good reference for fundamentals of deep learning.

1. Livan, Novaes & Vivo: [Introduction to Random Matrices - Theory and Practice
](https://arxiv.org/abs/1712.07903)
2. Poole, Lhiri, Raghu, Sohl-Dickstein & Ganguli: [Exponential expressivity in deep neural networks through transient chaos
](https://arxiv.org/abs/1606.05340)
3. Schoenholz, Gilmer, Ganguli, & Sohl-Dickstein: [Deep information propagation](https://arxiv.org/pdf/1611.01232.pdf)
4. Goodfellow, Bengio & Courville: [Deep Learning](http://www.deeplearningbook.org)

Other helpful resources:
1. [Course Outline](/assets/sigmoid/misc/Course Outline.docx)



# 1 Introduction

**Motivation**: The paper we will study in this DFL course is part of a body of work with the broad goal of understanding what combination of network architecture and initialization allow a neural network to be trained with gradient-based methods.  This week, you will read about this problem of trainability, specifically its manifestation in deep neural networks and potential solutions developed by the community.

We also suggest that you skim the paper itself, specifically the introductory sections, to understand the relevance of vanishing/exploding gradients to trainability of neural networks.

**Objectives**:
After doing these readings, we would like you to understand the following background:
- Explain the vanishing/exploding gradient problem, and why it worsens as networks become deeper
- Relate vanishing/exploding gradients to the spectrums of various Jacobians
- Explain heuristics used by the community to circumvent the problem of vanishing/exploding gradients, in particular:
  - common initialization schemes, such as Xavier initialization
  - pre-training
  - skip connections / residual neural networks
  - non-saturating activation functions (ReLU and its variants)
We would also like you to have an overview of the paper's structure:
- Motivate and explain the problem the paper is trying to solve
		 - concentrating the entire spectrum of the network's Jacobian around unity
- Understand the setup of the paper, specifically why the following tools are necessary:
     - Mean-field signal propagation analysis
     - Random matrix theory

**Topics**:

- Trainability of networks, specifically the vanishing/exploding gradient problem
- Introduction to the paper / course overview

**Required Reading** 

Prerequisite: 
For those of you that aren’t familiar with deep learning, please read the following sections from the [Deep Learning book](http://www.deeplearningbook.org):
- 2.7 (Eigendecomposition)
- 2.8 (Singular value decomposition)
- 3.2 (Random variables)
- 3.3 (Probability distributions)
- 3.7 (Independence and conditional independence)
- 3.8 (Expectation, variance, and covariance)
- 5.7 (Supervised learning algorithms)

Initialization: 
From the [Deep Learning book](http://www.deeplearningbook.org)
- 8.2 (Challenges in neural network optimization)
- 8.4 (Parameter initialization strategies)

Other: 
[All you need is a good init](https://arxiv.org/pdf/1511.06422.pdf) by Mishkin et al., sections 1 and 2  
[Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) by Glorot and Bengio
Wikipedia [article](https://en.wikipedia.org/wiki/Residual_neural_network) on residual networks (skip connections)


**Optional Reading**:

1. [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
2. [Depth-first learning : NeuralODEs](http://www.depthfirstlearning.com/2019/NeuralODEs#3-resnets), section 3 on ResNets

<br />

# 2 Signal propagation

**Motivation**: One of the foundations which the analysis in _Resurrecting the Sigmoid_ rests on, is signal propagation in wide neural networks.  Understanding this mean-field analysis framework, and its results, also connects to more recent investigations of, e.g., neural networks as Gaussian processes, and the neural tangent kernel (see the [Jacot et al., 2018](https://arxiv.org/abs/1806.07572) and [Lee et al., 2019](https://arxiv.org/pdf/1902.06720).

**Topics**:

- Mean-field analysis of signal propagation in deep neural networks.

**Required Reading**:

Since this analysis is relatively new, the main sources of information online are the original papers in which the framework was developed, namely: 
- Poole, Lhiri, Raghu, Sohl-Dickstein & Ganguli: [Exponential expressivity in deep neural networks through transient chaos
](https://arxiv.org/abs/1606.05340) (Sections 1, 2, and 3)
- Schoenholz, Gilmer, Ganguli, & Sohl-Dickstein: [Deep information propagation](https://arxiv.org/pdf/1611.01232.pdf) (Sections 1, 2, 3, and 5)

These are very useful references, but for those unfamiliar with the field, not necessarily pedagogical.  The problem set listed below is designed to walk you through understanding the formalism in a self-contained manner.  We **strongly** suggest working through the problem set before reading the papers above, and only consulting afterwards or for reference.  Certain problems in the problem set point to sections of the above papers for hints.

**Optional Reading**:

Once you understand the mean-field analysis framework, you will have a good foundation for understanding the following papers.  These are strictly 'bonus', i.e. not connected to the main paper of this course.
1. [Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165)
2. [Wide Neural Networks of Any Depth Evolve as Linear Models under Gradient Descent](https://arxiv.org/abs/1902.06720)

**Questions**:

The full problem set, from which the below problems are taken, is [here](/assets/sigmoid/problem-sets/pdfs/1.pdf).

1. Problem 2 from full set: **The mean field approximation**

In this problem, we use the knowledge we gained in problem 1 to properly choose to initialize the weights and biases according to \(W^l \sim \mathcal{N}(0, \sigma_w^2/N)\) and \(b^l \sim \mathcal{N}(0, \sigma_b^2)\). We'll investigate some techniques that will be useful in understanding precisely how the network's random initialization influences what the net does to its inputs; specifically, we'll be able to take a look at how the _depth_ of the network together with the initialization governs the propagation of an input point as it flows forward through the network's layers.

a. A natural property of input points to study as the input flows through the net layer by layer is its length. Intuitively, this is closely related to how the net transforms the input space, and to how the depth of the net relates to that transformation. Compute the length \(q^l\) of the activation vector outputted by layer \(l\). When considering non-rectangular nets, where layer \(l\) has length \(N_l\), we want to distinguish this activation norm from the width of individual layers, so what's a more appropriate quantity we can track to understand how the lengths of activation vectors change in the net?
<details><summary>Solution</summary>
    <p>
        The length is simply the Euclidean magnitude, i.e. \(\sum_{i = 1}^N (h_i^l)^2\). We can stabilize this quantity, especially when \(N\) differs across layers, by normalizing:
            $$ q^l = \frac{1}{N_l} \sum_{i = 1}^{N_l} (h_i^l)^2 $$
    </p>
</details>

b. What probabilistic quantity of the neuronal activations does \(q^l\) approximate (with the approximation improving for larger \(N\))?
<details><summary>Hint</summary>
Recall that all neuronal activations \(h^l_i\) are zero-mean, and consider the definition of \(q^l\) from part (a) in terms of the empirical distribution of \(h^l_i\).
</details>

<details><summary>Solution></summary>
    <p>
        \(q^l\) is the second moment of the empirical distribution of layer \(l\) activations, and hence approximates the variance. Indeed, as \(N \to \infty\), the empirical average can be written \(q^l = \mathbb{E} \left( (h^l_i)^2 \right) = \text{Var}(h^l_i)\).
    </p>
</details>

c. Calculate the variance of an individual neuron's pre-activations, that is, the variance of \(h_i^l\).  Your answer should be a recurrence relation, expressing this variance in terms of \(h^{l-1}\) (and the parameters \(\sigma_w\) and \(\sigma_b\)).
<details><summary>Solution></summary>
    <p>
        Because the means of both the weight and bias distributions are zero, to calculate the variance we just need to calculate the second moment.  We can use the fact that the weights and biases are initialized independently, so that the variance of \(h_i^l\) is the sum of a bias term and a variance term:
            $$ \begin{align*}
                \langle (h_i^l)^2 \rangle &= \left\langle \left( \sum_j W_{ij}^l x_j^{l-1} \right) ^2 \right\rangle \\
                &= \left\langle \sum_{jj'} W_{ij}^l W_{ij'}^l x_j^{l-1} x_{j'}^{l-1} \right\rangle + \langle (b_i^l)^2 \rangle\\
                &= \left\langle \sum_{jj'} W_{ij}^l W_{ij'}^l x_j^{l-1} x_{j'}^{l-1} \right\rangle + \sigma_b^2\\
                &= \frac{\sigma_w^2}{N} \sum_j \langle (x_j^{l-1})^2 \rangle  + \sigma_b^2\\
                &= \sigma_w^2 \langle (x^{l-1})^2 \rangle  + \sigma_b^2\\
                &= \sigma_w^2 \langle \phi(h^{l - 1})^2 \rangle + \sigma_b^2
            \end{align*} $$
    </p>
</details>

d. Now consider the limit that the number of hidden neurons, \(N\), approaches infinity. Use the central limit theorem to argue that in this limit, the pre-activations will be zero-mean Gaussian distributed. Be explicit about the conditions under which this result holds.
<details><summary>Solution></summary>
    <p>
        The basic idea here is to use the central limit theorem since the pre-activation is a sum of a large number of random variables, i.e.:
            $$ h_i^l = \sum_j^N W_{ij}^l x_j^{l-1} + b_i^l $$
        There are \(N\) terms in the sum, so as \(N\) goes to infinity, we should have a sum of a large number of random variables which should be well-approximated by a Gaussian.

        However, there are a few things we need to be careful of:
        1. CLT can show that the sum \(\sum_j^N W_{ij}^l x_j^{l-1}\) is Gaussian-distributed, but there is still the bias term \(b_i^l\).  So we do have to assume that the bias term is Gaussian-distributed as well.
        2. In order to use CLT, we need each of the variables being added to have finite variance.  These individual variables are \(W_{ij}^{l}x_j^{l-1}\).  By construction the weights have finite variance; what about the previous layer's activations, \(x_j^{l-1}\)?  Unless the activation function \(\phi\) is pathological, if we \emph{assume that the previous-layer pre-activations have finite variance}, there should not be a problem here. In fact, if we just assume that the input distribution, i.e. \(x^0\), has finite variance, all the layers' activations do too. Certainly the commonly used activation functions sigmoid, ReLU, etc. cannot turn a finite-variance sample of pre-activations into an infinite-variance sample of activations.
        3. In order to use the CLT, we also need each of the variables being added to have identical distributions.  This is true by symmetry.
        4. The final condition for use of CLT is that the variables being added are all independent. Taking another look at the definition of \(q^l\),
            $$ q^l = \frac{1}{N} \sum_{i = 1}^N (h_i^l)^2 $$
        we want to show that each \(h_i^l\) is independent (from which the independence of their squares follows). Each \(h_i^l\) is in turn defined
            $$ h_i^l = \sum_{j = 1}^N W^l_{ij} \phi(h^{l - 1}_j) + b^l_i $$
        By assumption, \( W^l_{ij} \) and \(b^l_i\) are independent from each other and, over all \(i, j\), from any quantities in previous layers, including \(\phi(h^{l - 1}_j)\). But are the \(W^l_{ij}\) independent of the \(h^{l - 1}_j\)? To justify this, observe that we can view the sum above as a linear combination of the random variables \(W^l_{ij}\); even though, technically, the linear combination is also over random variables \(\phi(h^{l - 1}_j)\), the key is that over \(1 \leq i \leq N\), all the \(h^{l - 1}_j\)'s are the same. In other words, each neuronal activation in layer \(l\) depends on the same exact realization of the random variables that are the activations of the previous layer. So, \(h^l_i\) is essentially a linear combination of the (independent) \(W^l_{ij}\) with deterministic weights, at least with respect to \(i\). So, we can justify the use of the CLT in analyzing \(\lim_{N \to \infty} q^l\).
    </p>
</details>

e. With this zero-mean Gaussian approximation of \(q^l\), we have a single parameter characterizing this aspect of signal propagation in the net: the variance, \(q^l\), of individual neuronal activations (a proxy for squared activation vector lengths). Let's now look at how this variance changes from layer to layer, by deriving the relationship between \(q^l\) and \(q^{l - 1}\). In part (c), your answer should have included a term \(\langle (x^{l-1})^2 \rangle\).  In terms of the activation function \(\phi\) and the variance \(q^{l-1}\), write this expectation value as an integral over the standard Gaussian measure.
<details><summary>Solution></summary>
    <p>
        Since \(x_i^{l-1} = \phi(h_i^{l-1})\), we can write the variance \(\langle (x^{l-1})^2 \rangle\) as 
            $$ \begin{align*}
                \langle (x^{l-1})^2 \rangle &= \langle \phi(h^{l-1})^2 \rangle \\
                &= \int_\mathbb{R}~dx ~\phi(x)^2 ~p_{h^{l-1}}(x), 
            \end{align*} $$
        where \(p_{h^{l-1}}(x)\) is the pdf of the pre-activations \(h^{l-1}\).  By assumption this is a zero-mean Gaussian of variance \(q^{l-1}\), i.e.
            $$ p_{h^{l-1}}(x) = \frac{1}{\sqrt{2\pi q^{l-1}}} e^{-\frac{x^2}{2q^{l-1}}} $$
        This can be written in terms of the standard Gaussian distribution \(\rho(x)\) via the change of variables
            $$ p_{h^{l-1}}(x) = \frac{1}{\sqrt{q^{l-1}}} \rho(x/\sqrt{q^{l-1}}) $$
        meaning that the variance \(\langle (x^{l-1})^2 \rangle\) becomes
            $$ \langle (x^{l-1})^2 \rangle = \frac{1}{\sqrt{q^{l-1}}} \int_\mathbb{R}~dx ~\phi(x)^2  ~\rho(x/\sqrt{q^{l-1}}) $$
        Let \(y = x/\sqrt{q^{l-1}}\), then 
            $$\langle (x^{l-1})^2 \rangle =\int_\mathbb{R}~dy ~\phi(y\sqrt{q^{l-1}})^2  ~\rho(y).$$
    </p>
</details>

f. Use this result to write a recursion relation for \(q^l\) in  terms of \(q^{l-1}\), \(\sigma_w\), and \(\sigma_b\).
<details><summary>Solution></summary>
    <p>
        We just plug in, to get 
            $$ q^l = \sigma_w^2 \int_\mathbb{R}~dy ~\phi(y\sqrt{q^{l-1}})^2  ~\rho(y) + \sigma_b^2 $$
    </p>
</details>

2. Problem 3 from full set: Fixed points and stability

In the previous problem, we found a recurrence relation relating the length of a vector at layer \(l\) of a network to the length of the vector at the previous layer, \(l-1\) of the network.  In this problem, we are interested in studying the properties of this recurrence relation.  In the _Resurrecting the sigmoid_ paper, the results of this problem are used to understand at which bias point to evaluate the Jacobian of the input-output map of the network.

Note that in this problem, we are just taking the recurrence relation as a given, i.e. we do not need to worry about random variables or probabilities; all of that went into determining the recurrence relation. Instead, we'll use tools from the theory of dynamical systems to investigate the properties - in particular, the asymptotics - of this recurrence relation.

a. A simple example of a dynamical system is a recurrence defined by some initial value \(x_0\) and a relation \(x_n = f(x_{n-1})\) for all \(n>0\).  This system defines the resulting sequence \(x_n\).  Sometimes, these systems have _fixed points_, which are values \(x^*\) such that \(f(x^*) = x^*\). **If the value of the system,** \(x_m\)**, at some time-step** \(m\) **, happens to be a fixed point** \(x^*\)**, what is the subsequent evolution of the system?**
<details><summary>Solution></summary>
    <p>
        Since \(f(x^*) = x^*\), for all times greater than \(m\), the system simply stays at \(x^*\).
    </p>
</details>

b. **For the recurrence relation you derived in the previous problem, what is the equation which a fixed-point of the variance,** \(q^*\)**, must satisfy? Under some conditions (i.e. for some values of** \(\sigma_w\) **and \(\sigma_b\)), the value** \(q^*=0\) **is a fixed point of the system.  What are these conditions?**
<details><summary>Solution></summary>
    <p>
        A fixed point has to satisfy 
            $$ q^* = \sigma_w^2 \int_\mathbb{R} \phi \left( \rho \sqrt{q^*} \right)^2 \text{ d}\rho + \sigma_b^2 $$
        where \( \text{d}\rho \) is the standard Gaussian measure. If \(\sigma_b = 0\), i.e. there is no bias term, and the nonlinearity has a zero y-intercept, then there is a trivial fixed point of \(q^* = 0\).
    </p>
</details>

c. Now let us be concrete, and look at the recurrence relation in the special case of a nonlinearity \(\phi(h)\) which is both monotonically increasing and satisfies \(\phi(0) = 0\).  Note that both of the nonlinearities considered in the paper we are studying, the \(\tanh\) and ReLU nonlinearities, satisfy this property. **Show that those two properties (monotonicity and \(\phi(0)=0\)) imply that the length map \(q^l(q^{l-1})\) is monotonically increasing. What is the maximum number of times any concave function can intersect the line \(y = x\)?  What does this imply about the number of fixed points the length map \(q^l(q^{l-1})\) can have?**
<details><summary>Solution></summary>
    <p>
        To prove that the function is monotonically increasing with its argument \(q\), we take the derivative:
            $$ \begin{align*}
                f(q) &= \sigma_w^2 \int_\mathbb{R}~\phi(\rho\sqrt{q})^2~d\rho + \sigma_b^2 \\
                f'(q) &= \frac{\sigma_w^2}{\sqrt{q}} \int_\mathbb{R}~\phi(\rho\sqrt{q}) \phi ' (\rho\sqrt{q}) \rho d\rho 
            \end{align*} $$
        The derivative is positive since by assumption \(\phi '\) is positive everywhere, and \(\phi \rho\) is also positive everywhere. So the function is monotonically increasing.
        Note that since a fixed point is defined as a point, \(x^*\), such that \(f(x^*) = x^*\), graphically the fixed point can be found from the intersection of the length map \(q^l(q^{l-1})\) with the line \(y = x\).

        If you think about the definition of a concave function (specifically, the version of the definition which states that beween any two points \(x=a\) and \(x=b\), the graph of the function must lie above the line defined by \(f(a)\) and \(f(b)\)), you will realize that a concave function cannot intersect any line more than twice.  Thus, concavity implies that the function can have at most two fixed points.
    </p>
</details>

d. Let's be concrete now and consider the nonlinearity to be a ReLU. **Compute (analytically) the length map** \(q^l = f(q^{l-1})\)**, which will also depend on** \(\sigma_w\) **and** \(\sigma_b\) **.  For what values of** \(\sigma_w\) **and** \(\sigma_b\) **does the system have fixed point(s)? How does the value of the fixed point depend on** \(\sigma_w\) **and** \(\sigma_b\)?**
<details><summary>Solution></summary>
    <p>
        Starting from 
            $$ f(q) = \sigma_w^2 \int_\mathbb{R}~\phi(\rho\sqrt{q})^2~d\rho + \sigma_b^2 $$
        and explicitly inserting the nonlinearity \(\phi\) gives 
            $$ f(q) = \sigma_w^2 \int_0^\infty \rho^2 q~d\rho + \sigma_b^2 $$
        Note that since the ReLU nonlinearity is zero when the argument is zero and just the identity function when the argument is greater than zero, we can take its effect into account simply by changing the above limits of integration so that we only integrate over the region in which the argument is positive.  Now we can pull \(q\) out of the integral, 
            $$ f(q) = q \sigma_w^2 \int_0^\infty \rho^2 ~d\rho + \sigma_b^2 $$
        and to evaluate the integral, note that by symmetry of the Gaussian distribution, it's half of what it would be if we had the limits from \(-\infty\) to \(\infty\), in which case it would just be the variance of a standard Gaussian, and so
            $$ f(q) = q \frac{\sigma_w^2}{2} + \sigma_b^2 $$

        The important things to note here are that because \(f(q)\) is a simple linear function, there is at most a single fixed point of the system.  If \(\sigma_b^2\) is zero, that fixed point is at \(q=0\).  If \(\sigma_b^2 > 0\), then there is a fixed point only if \(\sigma_w < \sqrt{2}\).  Otherwise, the system does not have any fixed point.  This is a qualitative difference from the \(\tanh\) case, in which there is always a fixed point.  

        A slightly strange case is when \(\sigma_w = \sqrt{2}\) exactly, and \(\sigma_b = 0\).  In this case, the recurrence relation gives \(q^l(q^{l-1}) = q^{l-1}\), meaning that every point is a fixed point.
    </p>
</details>

e. Now let's consider the sigmoid nonlinearity \(\phi(h) = \tanh(h)\).  In this case the length map cannot be computed analytically, but it can be done numerically. **Numerically plot the length map, \(q^l=f(q^{l-1})\), for a few values of \(\sigma_w\) and \(\sigma_b\) in the following regimes: (i) \(\sigma_b=0\) and \(\sigma_w < 1\), (ii) \(\sigma_b = 0\) and \(\sigma_w > 1\), and (iii) \(\sigma_b > 0\).  Describe qualitatively the fixed points of the map in each regime.**
<details><summary>Solution></summary>
    <p>
        The following Python code should work:
            ```
            import numpy as np
            import scipy.integrate as integrate

            def integrand(x):
                gaussian = np.sqrt(2 * np.pi), -np.inf, np.inf) * np.exp(-0.5 * x**2)
                return np.tanh(x * np.sqrt(q))**2 * gaussian

            def fint(q):
                result = integrate.quad(integrand)
                return result[0]

            def lengthmap(q, sigma_w, sigma_b):
                return sigma_w**2 * fint(q) + sigma_b**2
            ```
        The behavior that should be seen is the following, as described in the transient chaos paper (ignoring the parts about stability because we haven't covered that yet.  See next part of the problem):
        _For \(\sigma_b = 0\) and \(\sigma_w < 1\), the only intersection is at \(q^*=0\). In this bias-free, small weight regime, the network shrinks all inputs to the origin. For \(\sigma_w > 1\) and \(\sigma_b = 0\), the \(q*=0\) fixed point becomes unstable and the length map acquires a second nonzero fixed point, which is stable. In this bias-free, large weight regime, the network expands small inputs and contracts large inputs. Also, for any nonzero bias σb, the length map has a single stable non-zero fixed point. In such a regime, even with small weights, the injected biases at each layer prevent signals from decaying to 0._
    </p>
</details>

f. Let’s now talk about the stability of fixed points. In a dynamical system, once the system reaches (or starts at) a fixed point, by definition it can never leave. But what happens if the system gets or starts near a fixed point?  In real physical systems, this question is very relevant because physical systems almost always have some noise which pushes the system away from a fixed point. In general, the fixed point can be either stable or unstable. For a stable fixed point, initializing the system near the fixed point will result in behavior which converges to the fixed point, i.e reducing the magnitude of the perturbation away from the fixed point. Conversely, for an unstable fixed point, the system initialized nearby will be repelled from the fixed point. **Use the derivative of the length map at a fixed point to derive conditions on the stability of the fixed point.**
<details><summary>Solution></summary>
    <p>
        If the absolute value of the derivative \(\frac{df}{dx}\), evaluated at the fixed point \(x^*\), is less than \(1\), then the system is stable.  This can be seen from considering initializing the system near the fixed point, say at \(x^* + \Delta x\).  After going through the length map, the value will be 
        $$ \begin{aligned}
            f(x^* + \Delta x) &\approx f(x^*) + f'(x^*) \Delta x \\
            &=  x^* + f'(x^*) \Delta x
        \end{aligned} $$

        So the deviation from the fixed point \(x^*\) has changed to \(f'(x^*) \Delta x\).  If the magnitude of \(f'(x^*)\) is less than \(1\), then the magnitude of this deviation is lower than \(\Delta x\), the system is getting closer to the fixed point, and the fixed point is said to be stable.

        Conversely, if the magnitude of \(f'(x^*)\) is greater than \(1\), then the deviations away from equilibrium grow, and the equilibrium is unstable.
    </p>
</details>

g. With this understanding of stability, revisit your result in part (e) for the \(\tanh\) nonlinearity. **Specifically, discuss the stability of the fixed points in each of the three regimes.  You can estimate the derivative of the length map by looking at the graphs.**
<details><summary>Solution></summary>
    <p>
        See the italicized paragraph in the solutions above, from the transient chaos paper.  In regime (i), there is a single fixed point, \(q^*=0\), and it is stable.  In regime (ii), there are two fixed points, \(q^*=0\) (unstable) and some other positive value (stable), and in regime (iii), there is only a positive fixed point, which is stable.
    </p>
</details>

h. **Do the same stability analysis for the ReLU network.**
<details><summary>Solution></summary>
    <p>
        In the \(\sigma_b = 0\) case, where the only fixed point is at \(q=0\), that point is stable if \(\sigma_w < \sqrt{2}\) (because then the slope of the line is less than unity) and unstable if \(\sigma_w > \sqrt{2}\).  Even for non-zero \(\sigma_b\), the fixed point (which will now be non-zero) is stable if \(\sigma_w < \sqrt{2}\).  

        The slightly strange case is when \(\sigma_w = \sqrt{2}\) exactly, and \(\sigma_b = 0\).  In this case, the recurrence relation gives \(q^l(q^{l-1}) = q^{l-1}\), meaning that every point is a fixed point.  In this case, the fixed points are neither stable nor unstable, since perturbations from them will neither grow or shrink.  
    </p>
</details>

i. **(Optional) You should have found above that the both the ReLU and** \(\tanh\) **systems never had more than one stable fixed point.  Show that this is a consequence of the concavity of the length map.**
<details><summary>Hint</summary>
    You can just draw a picture for this one. Consider using the fact that the length map is concave, which we discussed in part c).
</details>
<details><summary>Solution></summary>
    <p>
        Having two stable fixed points would mean having two intersection points with the line \(y=x\) at which the slope of the function is less than unity.  But this means that in both cases we approach the function from above, which means that there must have been a third intersection point in the middle.  But we already proved that because of the concavity of the length map, the system can have at most two fixed points.
        ![stable fixed point](../assets/sigmoid/problem-sets/source/1/stable_fixed_point.png)
    </p>
</details>

<br />

# 3 Random Matrix Theory: First glance

**Motivation**: The crux of the paper uses tools from the field of random matrix theory, which studies ensembles of matrix-valued random variables. Here, we will take a first stab at analyzing some of the relevant questions surrounding random matrices, getting a feel for how they and their spectra differ from deterministic matrices, depend on the way we sample the matrices, and what random matrices from different ensembles have in common.

**Objectives**:
- Gain familiarity with working with the spectra of random matrices.
- Understand the typical behavior of a random matrix's eigenvalues.
- Understand how standard RMT eigenvalue distributions are influenced by both level repulsion and confinement.
- Understand why RMT is used in the Resurrecting the Sigmoid paper

**Topics**:
- Eigenvalue spacing in random matrices

**Readings**:

1. [Livan RMT textbook, sections 2.1 - 2.3](https://arxiv.org/pdf/1712.07903.pdf)

**Optional Readings**:

1. [Random Matrix Theory and its Innovative Applications](http://math.mit.edu/~edelman/publications/random_matrix_theory_innovative.pdf) by Edelman and Yang
2. [Livan RMT textbook, chapters 3, 6, and 7](https://arxiv.org/pdf/1712.07903.pdf)

**Questions**:

The full problem set, from which the below problems are taken, is [here](/assets/sigmoid/problem-sets/pdfs/2.pdf).  


1. Avoided crossings in the spectra of random matrices
In the first DFL session’s intro to RMT, we mentioned that eigenvalues of random matrices tend to repel each other. Indeed, as one of the recommended textbooks on RMT states, this interplay between confinement and repulsion is the physical mechanism at the heart of many results in RMT. This problem explores that statement, relating it to a concept which comes up often in physics: the avoided crossing.

a. The simplest example of an avoided crossing is in a \(2 \times 2\) matrix. Let’s take the matrix

$$ \begin{pmatrix} 
    \Delta & J \\
    J & -\Delta 
\end{pmatrix} $$

i. Since this matrix is symmetric, its eigenvalues will be real. **What are its eigenvalues?**
<details><summary>Solution></summary>
    <p>
        The polynomial to solve for the eigenvalues \(\lambda\) is 
            $$ \begin{eqnarray}
                (\Delta - \lambda) (-\Delta - \lambda ) - J^2 &=& 0 \\
                \lambda^2 - (\Delta^2 + J^2) &=& 0
            \end{eqnarray} $$
So the eigenvalues are \(\pm \sqrt{\Delta^2 + J^2}\).
    </p>
</details>

ii. To see the avoided crossing here, **plot the eigenvalues as a function of \(\Delta\), first for \(J=0\), then for a few non-zero values of \(J\).**
<details><summary>Solution></summary>
    <p>
        Here is an example graph, showing \(J\) values \(0\), \(1\), and \(2\). The blue line shows no gap when \(J = 0\), and the gap opens up when \(J\) is non-zero.
        ![avoided crossing](../assets/sigmoid/problem-sets/source/2/avoided_crossing.png)
    </p>
</details>

iii. You should see a gap (i.e. the minimal distance between the eigenvalue curves) open up as \(J\) becomes non-zero. **What is the size of this gap?**
<details><summary>Solution></summary>
    <p>
        To get the gap, evaluate the expression for the eigenvalues when \(\Delta\) is zero, and you find that the gap is \(2|J|\).
    </p>
</details>

b. Now take a matrix of the form

$$ \begin{pmatrix} 
    A & C \\
    C & D 
\end{pmatrix}. $$

In terms of \(A\), \(C\), and \(D\), **what is the absolute value of the difference between the two eigenvalues of this matrix?**
<details><summary>Solution></summary>
    <p>
        The difference in eigenvalues won't shift if we add a multiple of the identity matrix to our original matrix, meaning that the eigenvalue difference is the same as that of the matrix  
            $$ \begin{pmatrix} 
                \frac{1}{2}\left(A - D\right) & C \\
                C & -\frac{1}{2}\left(A - D\right)
            \end{pmatrix} $$
        The eigenvalue difference is thus (using the eigenvalues we calculated from the previous part):
            $$ s = \sqrt{4C^2 + \left( A - D \right) ^2} $$
    </p>
</details>

c. Now let’s make the matrix a random matrix.  We will take \(A\), \(C\), and \(D\) to be independent random variables, where the diagonal entries \(A\) and \(D\) are distributed according to a normal distribution with mean zero and variance one,  while the off-diagonal entry \(C\) is also a zero-mean Gaussian but with a variance of \(\frac{1}{2}\).

i. **Use the formula you derived in the previous part of the question to calculate the probability distribution function for the spacing between the two eigenvalues of the matrix.**
<details><summary>Solution></summary>
    <p>
        From the previous part we know the spacing as a function of the random variables \(A\), \(B\), \(C\):
            $$ s = \sqrt{4C^2 + \left( A - D \right) ^2} $$
        So we can write in terms of the joint probability density function of \(A\), \(B\), and \(C\) that 
            $$ \begin{eqnarray}
                p_s(x) &=& \int~da~db~dc~p_{A,B,C}(a, b, c) ~ \delta(x - s(a,b,c)) \\
                &=& \frac{1}{2\pi \sqrt{\pi}}\int_{-\infty}^{\infty}da \int_{-\infty}^{\infty}db \int_{-\infty}^{\infty}dc~ e^{-a^2/2} e^{-b^2/2} e^{-c^2} ~ \delta(x - s(a,b,c))
            \end{eqnarray} $$
        where \(\delta\) is the Dirac delta function. Now perform the change of variables to cylindrical coordinates \(r\), \(\theta\), \(z\), with 
            $$ \begin{eqnarray}
                r \cos{\theta} &=& a - d \\
                r \sin{\theta} &=& 2c \\
                z &=& a + d 
            \end{eqnarray} $$
        The inverse of this transformation is 
            $$ \begin{eqnarray}
                a &=& \frac{1}{2} ( r\cos(\theta) + z) \\
                b &=& \frac{1}{2} ( z - r\cos(\theta))  \\
                c &=& \frac{1}{2} r\sin(\theta)
            \end{eqnarray} $$
        For later, note that 
            $$ a^2 + b^2 = \frac{1}{2} (r^2 \cos^2(\theta) + z^2) $$
        And the Jacobian can be calculated as \(J=-r/4\) (see the Livan book, section 1.2).

        In terms of the new variables, the spacing \(s\) becomes \(r\), and the integration becomes
            $$ \begin{eqnarray}
                p_s(x) &=& \frac{1}{2\pi \sqrt{\pi}}\int_{-\infty}^{\infty}da \int_{-\infty}^{\infty}db \int_{-\infty}^{\infty}dc~ e^{-a^2/2} e^{-b^2/2} e^{-c^2} ~ \delta(x - s(a,b,c)) \\
                &=& \frac{1}{8\pi \sqrt{\pi}}\int_{0}^{\infty} r~dr \int_{0}^{2\pi}d\theta \int_{-\infty}^{\infty}dz~ e^{-a^2/2} e^{-b^2/2} e^{-c^2} ~ \delta(x - r) \\
                &=& \frac{1}{8\pi \sqrt{\pi}}\int_{0}^{\infty} r~dr \int_{0}^{2\pi}d\theta \int_{-\infty}^{\infty}dz~ e^{-\frac{1}{2} (a^2 + b^2 + 2c^2)} ~ \delta(x - r) \\
                &=& \frac{1}{8\pi \sqrt{\pi}}\int_{0}^{\infty} r~dr \int_{0}^{2\pi}d\theta \int_{-\infty}^{\infty}dz~ e^{-\frac{1}{4} (r^2 \cos^2(\theta) + z^2  + r^2\sin^2\theta)} ~ \delta(x - r) \\
                &=& \frac{1}{8\pi \sqrt{\pi}}\int_{0}^{\infty} r~dr \int_{0}^{2\pi}d\theta \int_{-\infty}^{\infty}dz~ e^{-\frac{1}{4} (r^2  + z^2)} ~ \delta(x - r) \\
                &=& \frac{1}{4\sqrt{\pi}}\int_{0}^{\infty} re^{-\frac{1}{4} r^2}~\delta(x - r)~dr \int_{-\infty}^{\infty}dz~ e^{-\frac{z^2}{4}} ~  \\
                &=& \frac{1}{2}\int_{0}^{\infty} re^{-\frac{1}{4} r^2}~\delta(x - r)~dr \\
                &=& \frac{x}{2}e^\frac{-x^2}{4}
            \end{eqnarray} $$
    </p>
</details>

ii. **What is the behavior of this pdf at zero?  How does this relate to the avoided crossing you calculated earlier?**
<details><summary>Solution></summary>
    <p>
        Clearly the pdf we calculated above is exactly zero at \(s=0\), and grows linearly with \(s\).  This absence of spacings at zero is the same phenomenon as the avoided crossing noted above for deterministic matrices.  Another way to see this is to note that from the first part of the problem, the only way to have a spacing of zero is to have the diagonal elements equal each other while the off-diagonal element needs to be zero.  The set of points satisfying this condition is a line in the full 3D space of points, so will have a very low probability of occurring.
    </p>
</details>

d. **Verify using numerical simulation that the pdf you found in the previous part is correct.**
<details><summary>Solution></summary>
    <p>
        The following Python code should work; by generating plots using the two functions, we can verify that they match.
            ```
            import numpy as np

            def eigenvalue_spacing():
                A = np.random.normal(scale=1)
                D = np.random.normal(scale=1)
                C = np.random.normal(scale=np.sqrt(0.5))
                
                M = np.array([[A, C], [C, D]]
                eigenvalues, _ = np.linalg.eig(M)
                return abs(eigenvalues[0] - eigenvalues[1])

            def pdf(x):
                return x / 2 * np.exp(- x**2 / 4)
            ```
    </p>
</details>

<br />

# 4 Random Matrix Theory: Central tools and concepts

**Motivation**: In this section we cover the final topic before we can get to the calculations in the paper: free probability, specifically its instantiation in random matrix theory.  This is a huge topic, and quite difficult, but to understand the paper, luckily we don’t need to learn too much of the field.  The basic question to think about is: given two random matrices whose spectral densities we know, when can we calculate the spectral density of their sum or product?  

We also cover some canonical results in random matrix theory, like the semicircular law.

**Objectives**:
- Understand some basic properties that are of interest when working with random matrices.
- Specifically, for this paper, understand why we are interested in the eigenvalue/singular-value distribution of matrices.
- Be able to describe some canonical ensembles of random matrices, and their properties.
- Be able to explain why we need the theory of freely-independent matrices in this paper. 

**Topics**:
- Free independence.
- The \(R\)- and \(S\)-transforms.
- The semicircle law.

**Reading**:

As before, the primary learning tool this week is the problem set below.  The following readings will help contextualize the problems.  

1. [Livan RMT textbook, chapter 17](https://arxiv.org/pdf/1712.07903.pdf)
2. Section 2.3 of the [Resurrecting the Sigmoid](https://arxiv.org/pdf/1711.04735.pdf) paper 
3. [Partial Freeness of Random Matrices](https://arxiv.org/abs/1204.2257) by Chen et al., Sections 1, 2, 3, and 5

**Optional Readings**:

It is tough to find an exposition of free probability theory (i.e., the theory of non-commuting random variables) at an elementary level.  The chapter in the Livan textbook listed above is a great resource, and the following papers might also help shed light on the subject.

1. [Financial Applications of Random Matrix Theory: a short review](https://arxiv.org/pdf/0910.1205.pdf) by Bouchaud and Potters, section III
2. [Applying Free Random Variables to Random Matrix Analysis of Financial Data Part I: A Gaussian Case](https://arxiv.org/pdf/physics/0603024.pdf) by Burda et al. 

**Questions**:

The full problem set, from which the below problems are taken, is [here](/assets/sigmoid/problem-sets/pdfs/3.pdf).

1. Why we need free probability

In the upcoming lectures, we will encounter the concept of free independence of random matrices.  As a reminder, in standard probability theory (of scalar-valued random variables), two random variables \(X\) and \(Y\) are said to be independent if their joint pdf is simply the product of the individual marginals, i.e.

$$ p_{X,Y}(x,y) = p_X(x) p_Y(x) $$

When we have independent scalar random variables \(X\) and \(Y\), then in principle it is possible to calculate the distribution of any function of these variables, say the sum \(X + Y\) or the product \(XY\). 

When it comes to random matrices, we are often interested in calculating the spectral density (the probability density of eigenvalues) of the sum or product of random matrices.  In the _Resurrecting the Sigmoid_ paper, for example, we will calculate the spectral density of the network's input-output Jacobian, which is the product of several matrices for each layer.  So we need an analogue of independent variables for matrices (this condition is known as _free independence_), such that if we know the spectral densities of each one, we can calculate spectral densities of sums and products.

The simplest condition we might imagine under which two matrix-valued random variables (or, equivalently, two matrix ensembles) being freely independent is that all of the entries of each matrix are mutually independent.  However, it turns out that this condition is not good enough! In other words, independent entries sometimes are not enough to destroy all possible angular correlations between the eigenbases of two matrices. Instead, the property that generalizes statistical independence to random matrices is stronger and known as _freeness_.

In this problem, we will see a concrete example of matrix ensembles with mutually independent entries, yet knowing the eigenvalue spectral density of each ensemble is not enough to determine the eigenvalue spectral density of the sum. 

Define three different ensembles of 2 by 2 matrices:

- Ensemble 1: To sample a matrix from ensemble 1, sample a standard Gaussian scalar random variable \(z\) and multiply it by each element in the matrix \(\sigma_z\), where 

$$ \sigma_z = \left( \begin{array}{cc} 1 & 0 \\ 0 & -1 \end{array} \right) $$

  Thus the sampled matrix will be \(z \sigma_z\).
- Ensemble 2: To sample a matrix from ensemble 2, sample a standard Gaussian  scalar random variable \(z\) and multiply it by each element in the matrix \(\sigma_x\), where 

$$ \sigma_x = \left( \begin{array}{cc} 0 & 1 \\ 1 & 0 \end{array} \right) $$

  Thus the sampled matrix will be \(z \sigma_x\).

a. What is the spectral density \(\rho_1(x)\) of eigenvalues of matrices sampled from ensemble 1?
<details><summary>Solution</summary>
    <p>
    The eigenvalues of \(\sigma_z\) are \(\pm 1\),  so the spectral density \(\rho_1(x)\) will be identical to the probability density of \(z\) except for a factor of \(2\) (because it has to integrate to \(2\), the number of eigenvalues, instead of \(1\) namely
    $$\rho_1(x) = \frac{\sqrt{2}}{\sqrt{\pi}} e^{-x^2/2}$$
    </p>
</details>

b. What is the spectral density \(\rho_2(x)\) of eigenvalues of matrices sampled from ensemble 2?
<details><summary>Solution</summary>
    <p>
    The eigenvalues of \(\sigma_x\) are exactly the same as those of \(\sigma_z\), namely \(\pm 1\),  so the spectral density \(\rho_2(x)\) is the same as \(\rho_1(x)\):
    $$\rho_2(x) = \frac{\sqrt{2}}{\sqrt{\pi}} e^{-x^2/2}$$
    </p>
</details>
You should have found above that the spectral densities of both ensembles are the same.  However, we will see now that simply knowing the spectral density is not enough to determine the spectral density of the sum.

c. Let \(A\) and \(B\) be two matrices independently sampled from ensemble 1.  Calculate _analytically_ the spectral density of the sum, \(A + B\).
<details><summary>Solution</summary>
    <p>
    We can write this matrix as \((z_1 + z_2)\sigma_z\), where \(z_1\) and \(z_2\) are standard normal variables.  The eigenvalues are thus \(\pm (z_1 + z_2)\).  Since \(z_1\) and \(z_2\) are independent, their sum will be a zero-mean Gaussian with variance \(2\).  So the spectral density of eigenvalues will be twice that of such a Gaussian, namely:
    $$\rho_{A+B}(\lambda) = \frac{1}{\sqrt{\pi}} e^{-\lambda^2/4}$$
  </p>
</details>
d. Now let \(C\) be a matrix sampled from ensemble 2.  In the next part, you will calculate the spectral density of the sum \(A + C\), where \(A\) is drawn from ensemble 1 and \(C\) is drawn from ensemble 2.  However, to see immediately that the distributions of \(A+B\) and \(A+C\) will be different, consider the behavior of the spectral density of \(A+C\) at zero.  Based on your knowledge of avoided crossings from the previous problem set, **describe the spectral density of \(A+C\) at \(\lambda =0\) and contrast this to the spectral density of \(A+B\)**.
<details><summary>Solution</summary>
    <p>
    Notice that the matrix \(A+C\) will have the same form as the matrix considered in the previous problem set, and we found that for such a matrix the presence of the off-diagonal term caused their to be a level repulsion.  So, the eigenvalue spectral density should go to zero as \(\lambda\) approaches zero, for the matrix \(A+C\).  However, in the above part, we calculated that for matrices \(A+B\), there is no avoided crossing and the pdf is finite at \(\lambda = 0\).
    </p>
</details>
e. Now let \(C\) be a matrix sampled from ensemble 2.  Calculate the spectral density of the sum, \(A + C\).  Make sure this is consistent with what you argued above about the behavior at \(\lambda = 0\).
<details><summary>Solution</summary>
    <p>
    Notice that the matrix \(A+C\)will have the same form as the matrix considered in the previous problem set, and we found that for such a matrix the presence of the off-diagonal term caused their to be a level repulsion.  So, the eigenvalue spectral density should go to zero as \(\lambda\)approaches zero, for the matrix \(A+C\).  However, in the above part, we calculated that for matrices \(A+B\), there is no avoided crossing and the pdf is finite at \(\lambda = 0\).
    </p>
</details>
Notice that the answers you got in the previous two parts were different, even though the underlying matrices that were being added had the same spectral density and independent entries.

2. Using the tools of free probability theory

From the last problem, you learned that if you're given two different random matrix ensembles, and you know the spectral density of the eigenvalues of each one, that might not be enough to determine the eigenvalue distribution of the sum (or product) of the two random matrices, _even if all of the entries of the two matrices are mutually independent!_ As we mentioned in the last problem, the (stronger) condition that we are after is known as _free independence_.  In general, proving that two matrix ensembles are "free" (freely independent) is quite tough, so we will not do that here.  Instead, we will look at the tools we use to do calculations _assuming_ we have random matrix ensembles which are freely independent.

Specifically, we will show that the sum of two freely independent random matrices, each of whose spectral density is given by a semicircle, is also described by the semicircle distribution.

a. Recall that the spectral density of the Gaussian orthogonal ensemble (in the large \(N\) limit) is given by the semicircle law:

$$ \rho_{sc}(x) = \frac{1}{\pi}\sqrt{2-x^2} $$

(sometimes you see this  with a \(4\) or \(8\) in the square root and a different factor accompanying \(\pi\) in the denominator.  This is just a matter of choosing which Gaussian ensemble---orthogonal, unitary, or symplectic---to use, and doesn't really matter for this problem)

In a previous problem set, you calculated the Stieltjes transform associated with the spectral density for the Gaussian _unitary_ ensemble.  Recall that the Stieltjes transform, \(G(z)\), is defined via the relation

$$ G(z) = \int_\mathbb{R}~dt \frac{\rho(t)}{z - t} $$

(In the previous problem set, this was called \(s_{\mu_N}(z)\).  In literature you often see the \(G(z)\) notation, since the Stieltjes transform is also known as the _resolvent_ or _Green's function_.)

You should have calculated in the last problem set that under the Stieltjes transform, 

$$ \frac{1}{2\pi}\sqrt{4-x^2} \mapsto \frac{z - \sqrt{z^2 - 4}}{2} $$


**Use the above fact to calculate the Stieltjes transform of the GOE semicircle given at the beginning of this problem (part (a)).  This is the first step to calculating the spectral density of the sum.**

<details><summary>Solution</summary>
    <p>
    Define 
    $$\begin{eqnarray*}
        f(x) &=& \frac{1}{2\pi}\sqrt{4 - x^2} \\
        g(x) &=& \frac{1}{\pi}\sqrt{2 - x^2}
    \end{eqnarray*}$$
    Notice that 
    \begin{equation}
        g(x) = \sqrt{2} f(x\sqrt{2}).
    \end{equation}
    If we define \(G_g(z)\) and \(G_f(z)\) as the Green's functions corresponding to \(g(x)\) and \(f(x)\), respectively, then we can get a relation between the two:
    \begin{eqnarray}
        G_g(z) &=& \int~dt~\frac{g(t)}{z - t} \\
        &=& \sqrt{2}\int~dt~\frac{f(t\sqrt{2})}{z - t} \\
        &=& \sqrt{2}\int~\frac{dy}{\sqrt{2}} \frac{f(y)}{z - y/\sqrt{2}} \\
        &=& \sqrt{2}\int~dy~\frac{f(y)}{z\sqrt{2} - y} \\
        &=& \sqrt{2}G_f(z\sqrt{2}).
    \end{eqnarray}
    Since we have previously calculated that 
    \begin{equation}
        G_f(z) = \frac{z - \sqrt{z^2 - 4}}{2},
    \end{equation}
    this immediately gives us that
    \begin{equation}
        G_g(z) = z - \sqrt{z^2 - 2}.
    \end{equation}
    </p>
</details>
b.  We have calculated the Stieltjes transform or Green's function of the semicircle.  Now we proceed to calculate the so-called Blue's function, which is just defined as the functional inverse of the Green's function.  That is, the Green's function \(G(z)\) and the Blue's function \(B(z)\) satisfy 

$$ G(B(z)) = B(G(z)) = z $$

**Calculate the Blue's function corresponding to the semicircle Green's function you derived above.**
<details><summary>Solution</summary>
    <p>
    The inverse function is defined by the relation 
    \begin{equation}
        z = B - \sqrt{B^2 - 2}.
    \end{equation}
    Then 
    \begin{eqnarray}
        \sqrt{B^2 - 2} &=& B - z \\
        B^2 - 2 &=& B^2 - 2Bz + z^2 \\
        2Bz &=& z^2 + 2 \\
        B &=& \frac{z}{2} + \frac{1}{z}
    \end{eqnarray}
    </p>
</details>
c. You should have noticed that the Blue's function you calculated had a singularity at the origin, that is, a term given by \(1/z\).  The \(R\)-transform is defined as the Blue's function minus that singularity; that is, 

$$ R(z) = B(z) - \frac{1}{z} $$

**What is the \(R\)-transform of the GOE semicircle?**
<details><summary>Solution</summary>
    <p>
    Since
    \begin{equation}
        B = \frac{z}{2} + \frac{1}{z},
    \end{equation}
    we can immediately write
    \begin{equation}
        R(z) = \frac{z}{2}
    \end{equation}
    </p>
</details>
d. Finally we come to the law of addition of freely independent random matrices:  If we are given freely independent random matrices \(X\) and \(Y\), whose \(R\)-transforms are \(R_X(z)\) and \(R_Y(z)\), respectively, then the \(R\)-transform of the sum (or more precisely, the \(R\)-transform of the spectral density of the sum \(X + Y\)) is simply given by \(R_X(z) + R_Y(z)\).

Assume that two standard GOE matrices, say \(H_1\) and \(H_2\), are freely independent. **What is the \(R\)-transform of the spectral density of the sum \(H_+ = pH_1 + (1 - p) H_2\)?**
<details><summary>Solution</summary>
    <p>
    $$R_{H_+}(z) = z$$
    </p>
</details>
e. **Using the results above, argue that the sum of two freely-independent ensembles described by the semicircular law is also described by the semicircular law.**
<details><summary>Solution</summary>
    <p>
The \(R\)-transform of the sum (\(z\)) has the same functional form as the individual \(R\)-transforms (\(z/2\)), so it seems plausible that this means that when we invert it, we get a semicircle.  Let's make sure of this fact.

We should first figure out how the scaling of the \(R\)-transform affects the scaling of the matrix it is describing.  We can guess that this amounts to a simple scaling of the matrix itself.  Under the scaling of a general matrix \(H \mapsto cH\), the eigenvalue distribution goes from \(\rho(\lambda) \mapsto \rho(\lambda/c) /c\).  Then, by the same logic we used in part (a) of this problem, the Green's function goes \(G \mapsto G(z/c)/c\) (in part (a), \(c\) was \(\sqrt{2}\)).  To figure out the change in the Blue's function, we can write:
\begin{eqnarray}
    G_{pH}(B_{pH}(z)) = \frac{1}{p} G_{H}(B_{H}(z)/p) = z \\
     G_{H}(B_{H}(z)/p) = pz \\
     B_{pH}(z) = p B_H(pz) \\
\end{eqnarray}
And finally we can get the scaling of the \(R\)-transform:
\begin{eqnarray}
    R_{pH}(z) &=& pB_H(pz) - \frac{1}{z} \\
    &=& p\left( R_H(pz) + \frac{1}{pz} \right) - \frac{1}{z} \\
    &=& p R_H(pz)
\end{eqnarray}
With this result, we know that if we multiply the GOE matrix by \(\sqrt{2}\), the \(R\)-transform goes from \(z/2\) to \(z\).  This means that the spectral density of a sum of two GOE matrices is still semicircular, just with a \(\sqrt{2}\) scaling.  Another way of saying this is that the semicircular law is stable under free addition.
    </p>
</details>


<br />

# 5 Calculations in Resurrecting the Sigmoid

**Motivation**: Now, we are finally ready to actually perform the calculations from the paper, regarding the input-output Jacobian of randomly initialized neural nets, using random matrix theory and building off of the signal propagation concepts from section 2. Using this analysis, we will be able to predict the conditions under which dynamical isometry - which guarantees that inputs and gradients neither vanish nor explode as they pass through the net - is achievable.

**Objectives**:
- Be able to use the \(S\)-transform to calculate \(\sigma_{JJ^T}^2\) and \(\lambda_\text{max}\) for Gaussian nets.
- For Gaussian-initialized neural networks, explain why dynamical isometry is unattainable.
- Be able to use the \(S\)-transform to calculate \(\sigma_{JJ^T}^2\) and \(\lambda_\text{max}\) for orthogonal nets.
- Explain why orthogonal-initizlied neural networks can be initialized attain dynamical isometry when used with a sigmoidal activation function.
- Understand how to choose initialization parameters of an orthogonal, sigmoidal net of a given depth to ensure dynamical isometry

**Topics**:
- Jacobian spectra of neural networks with  Gaussian- and orthogonal- initialized random weight matrices.
- Decomposing neural network Jacobians via weight matrices and diagonal "nonlinearity" matrices.

**Required Reading**:

1. [_Resurrecting the Sigmoid_, sections 2.2 and 2.5](https://arxiv.org/pdf/1711.04735.pdf)

**Questions**:

The full problem set, from which the below problems are taken, is [here](/assets/sigmoid/problem-sets/pdfs/4.pdf).

1. Set up
In this problem set, we perform the main calculations from the  _Resurrecting the Sigmoid_ paper.  The ultimate aim is to look for conditions under which we can achieve _dynamical isometry_, the condition that all of the singular values of the network's Jacobian have magnitude \(1\).  Thus, the problems in this set are all aimed at calculating the eigenvalue spectral density \(\rho_{JJ^T}(\lambda)\)  of nets' Jacobians for specific choices of nonlinearities and weight-matrix initializations. We accomplish this by using the rule we learned from free probability:  \(S\)-transforms of freely-independent matrix ensembles multiply under matrix multiplication.  Following this logic, we will calculate \(S\)-transforms for the matrices \(WW^T\) and \(D^2\), combine these results to arrive at \(S_{JJ^T}\), and from that calculate \(\rho_{JJ^T}(\lambda)\).  In this problem set, as in the paper, we do not prove that the matrices are freely independent, but instead take that as an assumption.

Recall that our neural network is defined by the relations:

$$ \begin{aligned}
    h^l &= W^l x^{l-1} + b^l \\
    x^l &= \phi(h^l)
\end{aligned} $$

where the input is denoted \(h^0\) and the output is given by \(x^L\).  

a. **What is the Jacobian \(J\) of the input-output relation of this network?**
_
<details><summary>Hint</summary>
    See eq. 2 of the paper.
</details>
<details><summary>Solution</summary>
    <p>
    Using the chain rule gives:
    $$ J = \prod_{l=1}^L D^l W^l$$
    where \(D^l\) is a matrix of pointwise derivatives of the nonlinearity \(\phi\) at layer \(l\):
    \begin{equation}
        (D^l)_{ij} = \frac{dx^l_j}{dh^l_i} = \phi ' (h^l_i)\delta_{ij}.
    \end{equation}
    </p>
</details>

b. As the paper discusses, we are interested in the spectrum of singular values of \(J\), but all of the tools we have developed so far deal with the eigenvalue spectrum. 

**In terms of the singular values of \(J\), what are the eigenvalues of \(JJ^T\)?**
<details><summary>Solution</summary>
    <p>
    The definition of dynamical isometry, the condition we're after, is that the magnitude of the singular values of \(J\) should concentrate around 1. 
    </p>
</details>
**What is the dynamical isometry condition in terms of the eigenvalues of \(JJ^T\)?**
<details><summary>Solution</summary>
    <p>
    The singular values of a matrix \(A\) are the square roots of the eigenvalues of \(AA^T\), so the eigenvalues of \(JJ^T\) are the squared singular values of \(J\).

    Quick proof: By SVD, \(A=U \Sigma V^\dagger\) and so
    \(AA^T = (U\Sigma V^\dagger)(U\Sigma V^\dagger)^\dagger = U \Sigma^T \Sigma U^\dagger = U \Sigma^2 U^\dagger\) where \(\Sigma^2\) is composed of squared singular values and \(V^\dagger\) is matrix \(V\)'s conjugate transpose. Note that \(\Sigma^2\) equals matrix \(D\) from a spectral decomposition of \(AA^T\), which contains eigenvalues of \(AA^T\). Thus the squared singular values of \(A\) equal the eigenvalues of \(AA^T\). \qedsymbol

    So, the dynamical isometry condition is that the spectrum of eigenvalues of \(JJ^T\) concentrates around unity.
    </p>
</details>
c. Now that we're focused on \(JJ^T\) instead of \(J\), read the following section reproduced from the main paper, about the \(S\)-transform of \(JJ^T\)'s spectral density: 

$$ S_{JJ^T} = \prod_{l=1}^L S_{W_lW_l^T} S_{D_l^2} = S_{WW^T}^L S_{D^2}^L $$
_where we have used the identical distribution of the weights to define \(S_{WW^T} = S_{W_l W_l^T}\) for all \(l\), and we have also used the fact the pre-activations are distributed independently of depth as \(h_l \sim \mathcal{N}(0,q^*)\), which implies that \(S_{D_l^2} = S_{D^2}\) for all \(l\).Eqn. (12) provides a method to compute the spectrum \(\rho_{JJ^T} (\lambda)\). Starting from \(\rho_{W^T W} (\lambda)\) and \(\rho_{D^2}\), we compute their respective \(S\)-transforms through the sequence of equations eqns. (7), (9), and (10), take the product in eqn. (12), and then reverse the sequence of steps to go from \(S_{JJ^T}\) to \(\rho_{JJ^T} (\lambda)\) through the inverses of eqns. (10), (9), and (8). Thus we must calculate the \(S\)-transforms of \(WW^T\) and \(D^2\), which we attack next for specific nonlinearities and weight ensembles in the following sections. In principle, this procedure can be carried out numerically for an arbitrary choice of nonlinearity, but we postpone this investigation to future work._

**Prove the equation at the top of the box.**

<details><summary>Hint</summary>
    This is done in the first appendix of the paper. Note that you should assume free independence of the \(D\)'s and \(W\)'s.
</details>

The upshot of this problem is that we need to calculate the quantities \(S_{WW^T}\) and \(S_{D^2}\) for whatever nonlinearities and weight initialization schemes we're interested in.  
<details><summary>Solution</summary>
    <p>
    \(JJ^T =\left( \prod_{l=1} D^l W^l\right) \left(\prod_{l=1} D^l W^l\right)^T = \left(D_L W_L \ldots D_1 W_1\right) \left(D_L W_L \ldots D_1 W_1\right)^T\) by expanding the product.  So 
    $$ S_{JJ^T} = S_{\left( D_L W_L \ldots D_1 W_1\right) \left( D_L W_L \ldots D_1 W_1\right)^T} $$
    Since the \(S\)-transform is defined in terms of moments of the eigenvalue distribution, it is invariant to cyclic permutations (since the trace, which defines moments, is invariant to cyclic permutations).  So, we can re-order matrices in the product, yielding:
    $$ S_{JJ^T} = S_{(W_L^T D_L^T D_L W_L) (D_{L-1} W_{L-1} \ldots D_1 W_1)(D_{L-1} W_{L-1} \ldots D_1 W_1)^T} $$
    Then, assuming free independence, the \(S\)-transforms multiply:
    $$ S_{JJ^T} = S_{(W_L^T D_L^T D_L W_L)} S_{(D_{L-1} W_{L-1} \ldots D_1 W_1)(D_{L-1} W_{L-1} \ldots D_1 W_1)^T}.$$

    Again using invariance to cyclic permutations:
    $$ S_{JJ^T} = S_{(D_L^T D_L W_L W_L^T)} S_{(D_{L-1} W_{L-1} \ldots D_1 W_1)(D_{L-1} W_{L-1} \ldots D_1 W_1)^T}.$$
    And again assuming free independence:
    $$ S_{JJ^T} = S_{D_L^T D_L} S_{W_L W_L^T} S_{(D_{L-1} W_{L-1} \ldots D_1 W_1)(D_{L-1} W_{L-1} \ldots D_1 W_1)^T}.$$
    Since \(D\) is diagonal,
    $$ S_{JJ^T} = S_{D_L^2} S_{W_L W_L^T} S_{(D_{L-1} W_{L-1} \ldots D_1 W_1)(D_{L-1} W_{L-1} \ldots D_1 W_1)^T}.$$
    Continuing this procedure we get
    $$ S_{JJ^T} = \prod_{l=1}^L S_{D_l^2} S_{W_l^T W_l} $$
    Since the weight matrices \(W^l\) for each layer are identically distributed, their \(S\)-transforms are equal, so we can drop the subscript and write:
    $$ S_{JJ^T} =  S_{W^T W}^L \prod_{l=1}^L S_{D_l^2}$$
    Finally, using the fact that \(D^l\) matrices are identically distributed gives the desired expression.
    $$ S_{JJ^T} =  \left(S_{W^T W}\right)^L \left(S_{D^2}\right)^L$$
    </p>
</details>

3. \(S_{D^2}\) for ReLU and hard-tanh networks

In this problem, we turn to networks with nonlinearities. We look at two nonlinearities here, the ReLU function and a piecewise approximation to the sigmoid known as the hard-tanh.  These functions are defined as follows: 

$$ f_{\mathrm{ReLU}}(x) = \begin{cases} 
    0 & x\leq 0 \\
    x & x\geq 0
\end{cases} $$
$$ f_{\mathrm{HardTanh}}(x) = \begin{cases} 
    -1 & x\leq -1 \\
    x & -1\leq x\leq 1 \\
    1 & x\geq 1
\end{cases} $$

We want the spectral density, \(\rho_{JJ^T}(\lambda)\), of \(JJ^T\), where \(J\) is the Jacobian.  We will find this by first calculating its \(S\)-transform, \(S_{JJ^T}\).  As discussed in the introduction, this involves two separate steps: finding  \(S_{D^2}\) and finding \(S_{WW^T}\). Note that finding \(S_{D^2}\)'s closed form relies primarily on choice of nonlinearity, and finding \(S_{WW^T}\)'s closed form relies only on choice of weight initialization (and not on choice of nonlinearity).  In this problem, we focus on the nonlinearities (\(S_{D^2}\)); the next problems focus on the weight initializations (\(S_{WW^T}\)), and how to combine these to get the \(S\) transform of the Jacobian.

a. The probability density function of the \(D\) matrix depends on the distributions of inputs to the nonlinearity.  To calculate this, we will make a couple simplifying assumptions.  The first assumption is that we initialize the network at a critical point (defined in problem set 2).  

**If we are interested in finding conditions for achieving dynamical isometry, why is it a good assumption that the network is initialized at criticality?**
<details><summary>Solution</summary>
    <p>
    The criticality condition, \(\chi = 1\), implies that the <em>mean</em> squared singular value of \(J\), or equivalently that the mean eigenvalue of \(JJ^T\), is unity.  Dynamical isometry means that the <em>entire</em> spectrum of squared singular values of \(J\) is concentrated around unity.  So criticality is a prerequisite for dynamical isometry. 
    </p>
</details>
b. The second assumption we make in calculating the distribution of inputs to the nonlinearity is that the we have settled to a stationary point of the length map (the variance map). **Reread section 2.2 of _Resurrecting the Sigmoid_, and argue why this is also a good assumption.**
<details><summary>Solution</summary>
    <p>
    As described in both <em>Exponential Expressivity in Deep Neural Networks Through Transient Chaos</em> and in section 2.2 of <em>Resurrecting the Sigmoid</em>, the empirical distribution of network pre-activations approximates a \(0\)-mean, \(q^l\)-variance Gaussian distribution in the large-width limit.  The length map describing the evolution of \(q^l\) has a fixed point, which the papers show empirically is rapidly converged to.  Because of this rapid convergence, it is natural to assume that only a few initial layers are not characterized by this variance, and that we can neglet them in computing the spectrum of the network's Jacobian.  Conveniently, assuming we are at a fixed point makes \(D^2\) is independent of \(l\), greatly simplifying our analysis. 
    </p>
</details>
c. To find the critical points of both the ReLU and hard-tanh networks, recall from problem set 2 that criticality was defined by the condition \(\chi = 1\), where \(\chi\) is defined in eqn. (5) of the main paper. 
As in the paper, define \(p(q^*)\) as the probability, given the variance \(q^*\), that a given neuron in a layer is in its linear (i.e. not constant) regime. **Show that \(\chi = \sigma_w^2 p(q^*)\).**

<details><summary>Hint</summary>
    Plug the nonlinearity into the equation for \(\chi\) and reduce.
</details>
<details><summary>Solution</summary>
    <p>
    $$\chi = \sigma_w^2 \int D h \phi ' ((\sqrt(q^*)h)^2$$
    Where \(D h\) is the standard Gaussian measure. Note that when \(\phi' = 0\) (the slope of the activation function is zero) then \(\chi=0\). Thus, since \(\chi\) only takes on values in \({0,1}\). Thus the Gaussian measure integral, which represents probability that \(\phi ' \neq 0\) reduces to \(p(q^*)\), the probability that \(\phi' = 0\), so \(\chi = \sigma_w^2 p(q^*)\).
    </p>
</details>
d. **In terms of \(p(q^*)\), what is the spectral density \(\rho_{D^2}(z)\) (for both ReLU and hard-tanh networks) of the eigenvalues of \(D^2\)? **.
<details><summary>Solution</summary>
    <p>
    Bernoulli with parameter equal to the probability of being in the linear regime. The Dirac delta expresses the fact that both ReLU and hard-tanh are piecewise linear with sections at value \(0\), so their probability of being in the linear regime is a step function -- it allows us to express a discrete pdf (in this case with two values, \(0\) and \(1\)).
    </p>
</details>
e. **Following equations 7-10 in the main paper, derive the Stieltjes transform \(G_{D^2}(z)\), the moment-generating function  \(M_{D^2}(z)\), and the \(S\)-transform  \(S_{D^2}(z)\) in terms of \(p(q^*)\)**. 
Note: This should be the same for both ReLU and hard-tanh networks.
<details><summary>Solution</summary>
    <p>
    Recall that:
    $$ \rho_{D^2} (z) = (1-p(q^*)) \delta (z) + p(q^*) \delta(z-1)$$

    Then recall the definition 
    $$ G_{D^2} (z) = \int_\mathcal{R} \frac{\rho_x (t) dt}{z-t} = \frac{\rho_x(0)}{z} + \frac{\rho_x (1)}{z-1} = \frac{1-p(q^*)}{z} + \frac{p(q^*)}{z-1}$$

    Then 
    \begin{eqnarray*}
    M_{D^2}(z) &=& z G_{D^2}(z) - 1 \\
    &=& z \left(\frac{1-p(q^*)}{z} + \frac{p(q^*)}{z-1}\right) - 1 \\
    &=&  -p(q^*) + \frac{z p(q^*)}{z-1} \\
    &=& p(q^*) \left(\frac{z}{z-1} - 1\right) \\
    &=& \frac{p(q^*)}{{z-1}}
    \end{eqnarray*}

    Next use the definition 
    \begin{equation*}
        S_{D ^2} (z) = \frac{1+z}{z M_{D^2}^{-1} (z)}.
    \end{equation*}
    The inverse \(M_{D^2}^{-1}(z)\) is \(\frac{p(q^*)}{z} + 1\). Thus:
    $$ S_{D^2}(z) = \frac{1+z}{z \left(\frac{p(q^*)}{z} + 1\right)}  = \frac{z+1}{z+ p(q^*)}$$
    </p>
</details>
f. Now that we've calculated the transforms we wanted in terms of \(p(q^*)\), let us see what the critical point (which determines \(q^*\) and \(p(q^*)\)) looks like for our two nonlinearity options. **For ReLU networks, what is \(p(q^*)\)?  Show that this implies that the only critical point for ReLU networks is \((\sigma_w, \sigma_b) = (\sqrt{2},0).\)** 
<details><summary>Solution</summary>
    <p>
    For ReLUs, the nonlinearity is half in the positive linear regime and half at \(0\). Assuming \(0\)-mean symmetric activation distributions, the probability of being in the linear regime is \(p(q^*) = \frac{1}{2}\).
    Using the above result that \( \chi = \sigma_w^2 p(q^*) \) immediately tells us that \( \sigma_w^2 = 2 \).  
    Using equation (4) in the <em>Resurrecting the Sigmoid</em> paper,
    $$q^* = \sigma_w^2 \int \mathcal{D} h ~\phi(\sqrt{q^*}h)^2 + \sigma_b^2$$
    and using the fact that \(\phi\) is a ReLU, we can write  -->
    $$q^* = q^* \sigma_w^2 \int_{h>0} \mathcal{D} h~ h^2 + \sigma_b^2.$$
    Since the integrand is an even function, it can be evaluated easily
    $$q^* = \frac{1}{2} q^* \sigma_w^2 \int \mathcal{D} h~ h^2 + \sigma_b^2.$$
    The integral now is the variance of \(h\), which is unity by construction, so we simply get
    $$q^* = \frac{1}{2} q^* \sigma_w^2  + \sigma_b^2.$$
    Plugging in \(\sigma_w^2=2\) gives \(q^* = q^* + \sigma_b^2\), meaning \(\sigma_b^2 = 0\).
     </p>
</details>
g. For hard-tanh networks, the behavior is a bit more complex, but we can calculate it numerically.  As we saw in problem set 2, for the smooth tanh network there is a 1D curve in the \((\sigma_w, \sigma_b)\) plane which satisfies criticality.  The same is true for the hard tanh network, as we'll now see.  We are interested in three quantities, all of which are functions of \(\sigma_w\) and \(\sigma_b\):  \(q^*\), \(p(q^*)\), and \(\chi\). We've already seen (in part (c) above) that if we know \(\sigma_w\) and \(p(q^*)\), we can easily determine \(\chi\).  It turns out that there is also a simple relation between \(q^*\) and \(p(q^*)\). **Show that for the hard tanh network, \(p(q^*) = \mathrm{erf}(1/\sqrt{2q^*})\).**
<details><summary>Solution</summary>
    <p>
    TODO(seems like there's no solution in the latex)
    </p>
</details>
Now all that's left is to determine \(q^*\) as a function of \(\sigma_w\) and \(\sigma_b\), and then we can get both \(q^*\) and \(p(q^*)\). Remember that in problem set 2, you derived the relation 

$$ q^* = \sigma_w^2 \int~ \mathcal{D}h~ \phi(\sqrt{q^*}h)^2 + \sigma_b^2 $$
<details><summary>Solution</summary>
    <p>
    TODO(seems like there's no solution in the latex) 
    </p>
</details>
**Use this relation to get an implicit expression for \(q^*\) in terms of \(\sigma_w\) and \(\sigma_b\).**

**Using the three relations, and any programming language or numerical package of your choice, plot (in the \(\sigma_w\), \(\sigma_z\) plane) the three quantities of interest, and identify the critical line \(\chi = 1\).**
<details><summary>Solution</summary>
    <p>
    For hard-tanh, \(p(q^*)\) is the probability that a normally distribution set of activations takes on values in hard-tanh's linear regime (recall this is between \(-1\) and \(1\)). Thus we integrate \(\int_{-1}^{1} z dz\) where \(z\) is a zero-mean Gaussian with variance \(q^*\). The integral of the Gaussian is given by the error function. The error function (denoted \(erf\) and defined as the integral of the standard Gaussian) is commonly defined without the leading factor \(\frac{2}{\pi}\), so \(\int z dz = erf(\sqrt(1/2q^*)\) (the parameter \(1/2q^*\) is arrived at by substituting \(t=h/\sqrt{2q^*}\)). Thus \(p(q^*) = erf(\sqrt(1/2q^*)\).
    </p>
</details>
4. Can Gaussian initialization achieve dynamical isometry?

In this problem, we will consider weights with a Gaussian initialization, and use the results from the previous problems to investigate whether dynamical isometry can be achieved for such nets over our two main activation functions of interest (ReLU and hard-tanh).

a. As we've seen in the decomposition from the previous problems, the \(S\)-transform of \(\mathbf{J} \mathbf{J}^T\)  depends on the \(S\)-transform of \(D^2\), which was computed above, and that of \( WW^T \), which is a _Wishart random matrix_, i.e. the product of two random Gaussian matrices.

**Prove that \(S_{WW^T}(z) = \frac{1}{\sigma_w^2 \cdot (z + 1)}\), using the following connection between the moments of a Wishart matrix and the Catalan numbers:**
$$ m_k = \frac{\sigma_w^{2k}}{k + 1} {2k \choose k} $$

**where \(m_k\) is the \(k^\text{th}\) moment of \(WW^T\).**
<details><summary>Solution</summary>
    <p>
    Given the moments, we can easily form the moment-generating function
    $$ M_{WW^T}(z) &= \sum_{k = 1}^\infty \frac{m_k}{z^k} = \sum_{k = 1}^\infty \left( \frac{\sigma_w^2}{z} \right)^k \frac{1}{k + 1} {2k \choose k} = \sum_{k = 1}^\infty \left( \frac{\sigma_w^2}{z} \right)^k C_k $$
    where \(C_k\) is the \(k^\text{th}\) Catalan number. So, we can now exploit the defining recurrence relation for the Catalan numbers, that \(C_k = \sum_{j = 0}^{k - 1} C_j C_{k - j - 1}\) (if you think of the \(k^\text{th}\) Catalan number as the number of ways to balance \(2k\) parentheses, this recurrence is pretty intuitive). To start off, this recurrence starts with the \(C_0\), though our MGF does not, and this might make the calculation more difficult; let's temporarily work with
    $$ f(x) := \sum_{k = 0}^\infty \left( \frac{\sigma_w^2}{z} \right)^k C_k = 1 + M_{WW^T}(z) $$
    Next, the recurrence is in a sum of products of Catalan numbers; specifically, products whose indices have a constant sum. Seeing as \(f(x)\) is basically an infinitely long polynomial, and polynomial multiplication also involves such product sums, a good first attempt to apply this recurrence is to square our function. Indeed, we have:
    $$ f(x)^2 = \sum_{k = 0}^\infty \sum_{j = 0}^\infty \left( \frac{\sigma_w^2}{z} \right)^{k + j} C_k C_j $$
    which after collecting like terms, is
    $$ f(x)^2 = \sum_{k = 0}^\infty \left( \frac{\sigma_w^2}{z} \right)^k \sum_{j = 0}^{k - 1} C_j C_{k - j} = \sum_{k = 0}^\infty \left( \frac{\sigma_w^2}{z} \right)^k C_{k + 1} $$
    Thus,
    $$ \frac{\sigma_w^2}{z} f(x)^2 = \frac{\sigma_w^2}{z} \left( M_{WW^T}(z) + 1 \right)^2  = \sum_{k = 1}^\infty \left( \frac{\sigma_w^2}{z} \right)^k C_k = M_{WW^T}(z) $$
    Solving the quadratic equation yields
    $$ M_{WW^T}(z) = \frac{z}{2 \sigma_w^2} - 1 - \frac{1}{2} \sqrt{1 - \frac{4 \sigma_w^2}{z}} $$
    Now that we've reduced the MGF to a quadratic polynomial, inverting it is easy enough, and we are left with
    $$ M_{WW^T}^{-1}(z) = \sigma_w^2 \frac{(z + 1)^2}{z} $$
    $$ S_{WW^T}(z) = \left( \sigma_w^2 \cdot (z + 1) \right)^{-1} $$
    </p>
</details>
b. We now have enough pieces to begin attacking the calculation of the Jacobian singular value distribution - recall that due to the decomposition

$$ S_{JJ^T} = (S_{WW^T})^L \cdot (S_{D^2})^L $$

once we've calculated the \(S\)-transforms for \(D^2\) and \(WW^T\), we can easily obtain the \(S\)-transform of \(\mathbf{J} \mathbf{J}^T\).

**Using your solution to the previous part and the calculation of** \(S_{D^2}\) **from the earlier problems, show that**

$$ S_{JJ^T} = \sigma_w^{-2L} \cdot (z + p(q^*))^{-L} .$$
<details><summary>Solution</summary>
    <p>
    We calculated \(S_{D^2}\) in part (e) of problem 3, showing that 
    $$S_{D^2}(z) = \frac{z+1}{z+ p(q^*)}$$
    And from the previous part we know that 
    $$S_{WW^T}(z) = \frac{1}{\sigma_w^2(z+1)},$$
    so combining these gives
    $$ S_{JJ^T} = (S_{WW^T})^L (S_{D^2})^L = \left( \sigma_w^{-2} (1 + z)^{-1} \right)^L \left( \frac{1 + z}{z + p(q^*)} \right)^L = \sigma_w^{-2L} (z + p(q^*))^{-L} $$
    </p>
</details>
c. From the \(S\)-transform, one route to getting information about the spectrum of \(JJ^T\) is to compute the spectral density \(\rho_{JJ^T}(\lambda)\). While that calculation is too involved, we can get the answer to the question of achieving dynamical isometry by a slightly more indirect route.

**Use the \(S\)-transform you calculated above to calculate \(M_{JJ^T}^{-1}\) (the inverse of the moment-generating function for \(\mathbf{J} \mathbf{J}^T\)).**

<details><summary>Hint</summary>
    To compute the inverse MGF, recall the definition of the \(S\)-transform given in the paper (section 2.3, eqn. 10).
</details>
<details><summary>Solution</summary>
    <p>
     The \(S\)-transform is defined so that \(S_{JJ^T} = \frac{1 + z}{z M^{-1}_{JJ^T}(z)}\), so
    $$ M^{-1}_{JJ^T}(z) = \frac{1 + z}{z S_{JJ^T}(z)} = \frac{1 + z}{z} \left(z + p(q^*)\right)^L \sigma_w^{2L} $$
    </p>
</details>
d. We can now compute the variance of the \(JJ^T\) eigenvalue distribution, \(\sigma_{JJ^T}^2\). You should have calculated above that

$$ M_{JJ^T}^{-1}(z) = \frac{1 + z}{z} \cdot (z + p(q^*))^L \cdot \sigma_w^{2L} $$

Using the definition that

$$ M_{JJ^T}(z) = \sum_{k = 1}^\infty \frac{m_k}{z^k} $$

and the expression for the functional inverse of \(M_{JJ^T}\) to compute that the first two moments are

$$ m_1 = \sigma_w^{2L} p(q^*)^L $$
$$ m_2 = m_1^2 \cdot \frac{L + p(q^*)}{p(q^*)} $$

<details><summary>Hint</summary>
    Use the [Lagrange inversion theorem](https://en.wikipedia.org/wiki/Lagrange_inversion_theorem) (eqn. 18 in the paper) to obtain a power series for the inverse MGF and equate corresponding coefficients with our calculated expressions.
</details>
<details><summary>Solution</summary>
    <p>
    Note that we have a formula for \(M^{-1}(z)\) (suppressing the \(JJ^T\) subscript for clarity, but the moments are defined in terms of \(M(z)\).  In the paper, the Lagrange inversion theorem is used to express the constant and \(1/z\) coefficients of \(M^{-1}(z)\) in terms of the \(m_1\) and \(m_2\).  Here is a slightly hand-wavy proof of that result (the rigorous proof turns out to be quite difficult):
    

    Assume that the \(M^{-1}(z)\) can be written as a Taylor series with an additional \(1/z\) term  (This assumption is one of the weaknesses of this proof).  So
    $$
        M^{-1}(z) = \frac{a}{z} + b + cz + dz^2 + \cdots
    $$
    Since we know that 
    $$
        M(z) = \frac{m_1}{z} + \frac{m_2}{z^2} + \cdots,
    $$
    we can write
    $$
        z = \frac{m_1}{M^{-1}(z)} + \frac{m_2}{M^{-1}(z)^2} + \cdots
    $$
    Plugging in our ansatz above gives 
    $$
        z = \frac{m_1}{\left( \frac{a}{z} + b + cz + dz^2 + \cdots \right)} + \frac{m_2}{\left( \frac{a}{z} + b + cz + dz^2 + \cdots \right)^2} + \cdots
    $$
    We'll expand the RHS of the above equations assuming \(z\) to be small, and then equate coefficients of the RHS and LHS.  Specifically, we will expand the RHS to second order in \(z\).
    $$ 
    z  = \frac{m_1 z}{a} \bigg( 1 + (b/a) + (c/a)z + (d/a)z^2 + \cdots \bigg)^{-1} + \cdots
    $$
    $$ 
    \frac{m_2 z^2}{a^2} \bigg( 1 + (b/a) + (c/a)z + (d/a)z^2 + \cdots \bigg)^{-2} + \cdots = \frac{m_1}{a} z - \frac{m_1 b}{a^2} z^2 + \frac{m_2}{a^2} z^2 + O(z^3) 
    $$
        
    Since the coefficient of \(z\) above has to be unity, and the coefficient of \(z^2\) has to be zero, this implies that \(a = m_1\) and \(b .= m_2/m_1\).  This implies that our sought-after expression for \(M^{-1}(z)\) is 
    $$M^{-1}(z) = \frac{m_1}{z} + \frac{m_2}{m_1} + \cdots$$
    
    
    With this expression in hand, we can directly extract the constant and \(1/z\) coefficients of the function \(M^{-1}(z)\):
    
    Given the result of our earlier calculation that 
    $$ M_{JJ^T}^{-1}(z) = \left(1+\frac{1}{z}\right) \cdot (z + p(q^*))^L \cdot \sigma_w^{2L}, $$
    we see that the only place to get a \(1/z\) term here is from the constant term when the \((z+p(q^*))^L\) is expanded.  This constant term will simply be \(p(q^*)^L\), so the \(1/z\) term here, which is \(m_1\), is
    $$
        m_1 = \sigma_w^{2L} p(q^*)^L
    $$
    The constant term comes from two places.  One, the \(p(q^*)^L\) multiplies the \(1\) in the first term, and the term \(Lzp(q^*)^{L-1}\), coming from the binomial expansion, multiplies the \(1/z\) in the first term.  
    So this means that the constant coefficient, \(m_2/m_1\), is given by
    $$
        \frac{m_2}{m_1} = \sigma_w^{2L} p(q^*)^L + \sigma_w^{2L} L p(q^*)^{L-1}.
    $$
    Recognizing the first term in the RHS sum as \(m_1\), we can factor to get 
    $$
        \frac{m_2}{m_1} = m_1 \left( 1 + \frac{L}{p(q^*)} \right), 
    $$
    or,
    $$
        m_2 =  m_1^2 \left( 1 + \frac{L}{p(q^*)} \right),
    $$
    as desired.
    </p>
</details>
<br />

# 5 Paper Experiments & Results, Future / Related Work

**Motivation**: To wrap up our study of this paper, we'll replicate some of their experiments, programatically validating the theoretical results we derived above. We've provided some starter code [here](https://drive.google.com/open?id=1ocuk_mH4fJrFkDKhMT4ZdAqIMIgnFDet) in an IPython notebook.

**Objectives**:
- Experimentally confirm the linear dependence of the singular value spectrum of a neural net's Jacobian under various random initializations.
- Experimentally confirm the positive impact of dynamical isometry at initialization on the trainability of a neural net.

**Follow-up reading**:
Here are a couple papers you might enjoy, which build upon the results of this paper:
1. [Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks](https://arxiv.org/pdf/1806.05393.pdf) by Xiao et al.
2. [The Emergence of Spectral Universality in Deep Networks](https://arxiv.org/pdf/1802.09979.pdf)

<br />
