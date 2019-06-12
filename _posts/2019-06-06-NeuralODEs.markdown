---
layout: post
title:  "Neural ODEs"
date:   2019-06-06 10:00:00 -0400
categories: neural-nets
author: luca
blurb: "Neural ODEs are neural network models which generalize standard layer to layer propagation to continuous depth models. The forward propagation in many neural networks models can be seen as one step of discretation of an ODE; starting from this observation, one can construct and efficiently train models by numerically solving ODEs. On top of furnishing a new whole family of architectures, neural ODEs can be applied to gain memory efficiency in supervised learning tasks, construct a new class of invertible density models and model continuous time-series"
feedback: true
---

This guide would not have been possible without the help and feedback from many people. 

Special thanks to Prof. Joan Bruna and his class at NYU, [Mathematics of Deep Learning](https://github.com/joanbruna/MathsDL-spring19), and to Cinjon Resnick, who introduced me to DFL and helped me putting up this guide.

Thank you to Avital Oliver, Matt Johnson, Dougal MacClaurin, David Duvenaud, and Ricky Chen for useful contributions to this guide.

Thank you to Tinghao Li, Chandra Prakash Konkimalla, Manikanta Srikar Yellapragada, Shan-Conrad Wolf, Deshana Desai, Yi Tang, Zhonghui Hu for helping me prepare the notes.

Finally, thank you to all my fellow students who attended the recitations and provided valuable feedback.


# Why

Neural ODEs are neural network models which generalize standard layer to layer propagation to continuous depth models. The forward propagation in many neural networks models can be seen as one step of discretation of an ODE; starting from this observation, one can construct and efficiently train models by numerically solving ODEs. On top of furnishing a new whole family of architectures, neural ODEs can be applied to gain memory efficiency in supervised learning tasks, construct a new class of invertible density models and model continuous time-series.

In this curriculum, we will go through all the background topics necessary to understand these models. At the end, you should be able to implement neural ODEs and apply them to different tasks.

<br />

# 1 Numerical solution of ODEs - Part 1
  **Motivation**: ODEs are used to mathematically model a number of natural processes and phenomena. The study of their numerical 
    simulations is one of the main topics in numerical analysis and of fundamental importance in applied sciences. A first step to understand Neural ODEs is their definition and basic numerical techniques to solve them.

  **Topics**:

  1. Initial values problems
  2. One-step methods
  3. Consistency and convergence

  **Required Reading**:

  1. Sections 12.1-4 from [An Introduction to Numerical Analysis](https://www.cambridge.org/core/books/an-introduction-to-numerical-analysis/FD8BCAD7FE68002E2179DFF68B8B7237#) (S\"uli & Mayers)
  2. Sections 11.1-3 from [Numerical Mathematics](https://www.springer.com/us/book/9783540346586?token=holiday18&utm_campaign=3_fjp8312_us_dsa_springer_holiday18&gclid=Cj0KCQiAvebhBRD5ARIsAIQUmnlViB7VsUn-2tABSAhIvYaJgSEqmJXD7F4A7EgyDQtY9v_GeUsNif8aArGAEALw_wcB) (Quarteroni et al.)

  
  **Optional Reading**:

  1. Runge-Kutta methods: Section 12.5 from (S\"uli & Mayers)
  2. [Prof. Trefethen's class ODEs and Nonlinear Dynamics 4.2](http://podcasts.ox.ac.uk/odes-and-nonlinear-dynamics-42)

  **Questions**:

  1. Exercise 1 in Section 11.12 of (Quarteroni et al.)
     <details><summary>Solutions</summary>
     <p>
     ...
     </p>
     </details>

  2. Exercises 12.3,12.4, 12.7 in Section 12 of (S\"uli & Mayers)
     <details><summary>Solution</summary>
     <p>
     ...
     </p>
     </details>

  3. Consider the following method for solving \(y' = f(y)\):
       
     $$y_{n+1} = y_n + h*(theta*f(y_n) + (1-theta)*f(y_{n+1}))$$
    
     Assuming sufficient smoothness of \(y\) and \(f\), for what value of \(0 \leq\theta\leq 1\) is the truncation error the smallest? What does this mean about the accuracy of the method?
     <details><summary>Solution</summary>
     <p>
     ...
     </p>
     </details>
       

  4. [Colab Notebook](https://colab.research.google.com/drive/1bNg-RzZoelB3w8AUQ6mefRQuN3AdrIqX)
     <details><summary>Solution</summary>
     <p>
     ...
     </p>
     </details>

**Notes**: Here is a [link](/assets/nodes_notes/week1.pdf) to our notes for the lesson. 

<br />

# 2 Numerical solution of ODEs - Part 2
  **Motivation**: In the previous class we introduced some simple schemes to numerically solve ODEs. In this class we go through some more involved schemes and their convergence analysis. 

  **Topics**:

  1. Runge-Kutta methods
  2. Multi-step methods
  3. System of ODEs and absolute converge

  **Required Reading**:

  1. Runge-Kutta methods: Section 11.8 from (Quarteroni et al.) or Sections 12.{5,12} from (S\"uli & Mayers)
  2. Multi-step methods: Sections 12.6-9 from (Quarteroni et al.) or Section 11.5-6 from (S\"uli & Mayers)
  3. System of ODEs: Sections 12.10-11 from (Quarteroni et al.) or Sections 11.9-10 from (S\"uli & Mayers)

  
  **Optional Reading**:

  1. [Prof. Trefethen's class ODEs and Nonlinear Dynamics 4.1](http://podcasts.ox.ac.uk/odes-and-nonlinear-dynamics-41)
  2. Predictor-corrector methods: Section 11.7 from (Quarteroni et al.)
  3. Richardson extrapolation: Section 16.4 from [Numerical Recipes](http://numerical.recipes/)
  4. [Automatic Selection of Methods for Solving Stiff and Nonstiff Systems of Ordinary Differential Equations](https://epubs.siam.org/doi/pdf/10.1137/0904010?
  
  **Questions**:

  1. Exercises 12.11, 12.12, 12.19 in Section 12 of (S\"uli & Mayers)
     <details><summary>Solution</summary>
     <p>
     ...
     </p>
     </details>

**Notes**: Here is a [link](/assets/nodes_notes/week2.pdf) to our notes for the lesson. 

<br />

# 3 ResNets
  **Motivation**: The introduction of Residual Networks (ResNets) made possible to train very deep networks. In this section we study some residual architectures variants and their properties. We then look into how ResNets approximates ODEs and how this interpretation can motivate neural net architectures and new training approaches. 

  **Topics**:

  1. ResNets
  2. ResNets and ODEs

  **Required Reading**:

  1. ResNets: 
     * [ResNets](https://www.coursera.org/lecture/convolutional-neural-networks/resnets-HAhz9) 
     * [An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)
  2. ResNets and ODEs: 
     * Sections 1-3 from [Multi-level Residual Networks from Dynamical Systems View](https://arxiv.org/pdf/1710.10348.pdf)
     * [Reversible Architectures for Arbitrarily Deep Residual Neural Networks](https://arxiv.org/abs/1709.03698)
  
  **Optional Reading**:

  1. The original ResNets paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
  2. Another blog post on ResNets: [Understanding and Implementing Architectures of ResNet and ResNeXt for state-of-the-art Image Classification](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624)
  3. Invertible ResNets: [The Reversible Residual Network: Backpropagation Without Storing Activations](https://arxiv.org/pdf/1707.04585.pdf)
  4. [Stable Architectures for Deep Neural Networks](https://arxiv.org/pdf/1705.03341.pdf)
  
  **Questions**:

  1. Can you think of any other neural network architectures which can be seen as discretisations of some ODE?
     <details><summary>Solution</summary>
     <p>
     ...
     </p>
     </details>

  2. Do you understand why adding ‘residual layers’ should not degrade the network performance?
     <details><summary>Solution</summary>
     <p>
     See the notes below.
     </p>
     </details>

  3. How do the authors of the (Multi-level Residual Networks from Dynamical Systems View) explain the phenomena of still having almost as good performances in residual networks when removing a layer?
     <details><summary>Solution</summary>
     <p>
     ...
     </p>
     </details>

  4. Implement your favourite ResNet variant

**Notes**: Here is a [link](/assets/nodes_notes/week3.pdf) to our notes for the lesson. 

<br />

# 4 Normalising Flows
  **Motivation**: In this class we take a little detour through the topic of Normalising Flows. This is used for density estimation and generative modeling, and it is another model which can be seen a time-discretisation of its continuous-time counterpart.

  **Topics**:

  1. Normalising Flows
  2. End-to-end implementations with neural nets

  **Required Reading**:

  1. [Density Estimation by Dual Ascent of the Log-likelihood](https://math.nyu.edu/faculty/tabak/publications/CMSV8-1-10.pdf) (you can skip Section 3) (DE)
  2. [A family of non-parametric density estimation algorithms](https://math.nyu.edu/faculty/tabak/publications/Tabak-Turner.pdf) 
  3. [A post on Normalising flow](http://akosiorek.github.io/ml/2018/04/03/norm_flows.html)
  
  **Optional Reading**:

  1. Variational Inference with Normalizing Flows](https://arxiv.org/pdf/1505.05770.pdf)
  2. [High-Dimensional Probability Estimation with Deep Density Models](https://arxiv.org/pdf/1302.5125.pdf)
  
  **Questions**:

  1. In DE, what is the difference between \(rho_t\) and \(\tilde{\rho}_t\), i.e. what do they represent?
     <details><summary>Solution</summary>
     <p>
     ...
     </p>
     </details>

  2. In DE, why does eq. (4.2) imply convergence \(\rho_t\to\mu\) as \(t\to\infty\) ?
     <details><summary>Solution</summary>
     <p>
     ...
     </p>
     </details>

  3. What is the computational complexity of evaluating a determinant of a \(N\times N\) matrix, and why is that relevant in this context?
     <details><summary>Solution</summary>
     <p>
     ...
     </p>
     </details>

**Notes**: Here is a [link](/assets/nodes_notes/week4.pdf) to our notes for the lesson. 

<br />

# 5 The adjoint method (and auto-diff)
  **Motivation**: The adjoint method is a numerical method for efficiently computing the gradient of a function in numerical optimization problems. Understanding this method is essential to understand how to train ‘continuous depth’ nets. We also review the basics of Automatic Differentiation, which will help us understand the efficiency of the algorithm proposed in the NeuralODE paper. 

  **Topics**:

  1. Adjoint method
  2. Auto-diff

  **Required Reading**:

  1. Section 8.7 from [Computational Science and Engineering](http://math.mit.edu/~gs/cse/) (CSE)
  2. Sections 2,3 from [Automatic Differentiation in Machine Learning: a Survey](http://www.jmlr.org/papers/volume18/17-468/17-468.pdf)

  
  **Optional Reading**:

  1. [Prof. Steven G. Johnson's notes on adjoint method](http://math.mit.edu/~stevenj/notes.html)
  
  **Questions**:

  1. Exercises 1,2,3 from Section 8.7 of CSE
     <details><summary>Solution</summary>
     <p>
     ...
     </p>
     </details>

  2. Consider the problem of optimizing a real-valued function \(g\) over the solution of the ODE \(y' = Ay\), \(y(0) = y_0\) at time \(T>0\): \(min_{y0, A} g(y(T))\). What is the solution of the adjoint equation?
     <details><summary>Solution</summary>
     <p>
     See notes below.
     </p>
     </details>

  3. How do you get eq. (14) in Section 8.7 of CSE?
     <details><summary>Solution</summary>
     <p>
     See notes below.
     </p>
     </details>

**Notes**: Here is a [link](/assets/nodes_notes/week5.pdf) to our notes for the lesson. 

<br />

# 6 Neural ODEs
  **Motivation**: Let’s read the paper!

  **Topics**:

  1. Normalising Flows
  2. End-to-end implementations with neural nets

  **Required Reading**:

  1. [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
  2. [A blog post on NeuralODEs](https://rkevingibson.github.io/blog/neural-networks-as-ordinary-differential-equations/)
  
  **Optional Reading**:

  1. A follow-up paper by the authors on scalable continuous normalizing flows: [Free-form Continuous Dynamics for Scalable Reversible Generative Models](https://arxiv.org/abs/1810.01367)

**Notes**: Here is a [link](/assets/nodes_notes/week6.pdf) to our notes for the lesson. 

<br />
