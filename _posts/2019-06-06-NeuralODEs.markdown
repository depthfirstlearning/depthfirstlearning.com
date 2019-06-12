---
layout: post
title:  "Neural ODEs"
date:   2019-06-06 10:00:00 -0400
categories: neural-nets
author: luca
blurb: "Neural Ordinary Differentiable Equations (Neural ODEs) are deep learning architectures which combine neural networks and ordinary differentiable equations, providing new models for the familiar litany of tasks from supervised learning to generative modeling to time series forecasting. In this curriculum, we will dive deep into these models with an end goal of implementing them yourself."
feedback: true
---

This guide would not have been possible without the help and feedback from many people. 

Special thanks to Prof. Joan Bruna and his class at NYU, [Mathematics of Deep Learning](https://github.com/joanbruna/MathsDL-spring19), and to Cinjon Resnick, who introduced me to DFL and helped complete this guide.

Thank you to Avital Oliver, Matt Johnson, Dougal MacClaurin, David Duvenaud, and Ricky Chen for useful contributions to this guide.

Thank you to Tinghao Li, Chandra Prakash Konkimalla, Manikanta Srikar Yellapragada, Shan-Conrad Wolf, Deshana Desai, Yi Tang, Zhonghui Hu for helping me prepare the notes.

Finally, thank you to all my fellow students who attended the recitations and provided valuable feedback.

<div class="deps-graph">
<iframe class="deps" src="/assets/nodes-deps.svg" width="200"></iframe>
<div>Concepts used in Neural ODEs. Click to navigate.</div>
</div>

# Why

Neural ODEs are neural network models which generalize standard layer to layer propagation to continuous depth models. Starting from the observation that the forward propagation in neural networks is equivalent to one step of discretation of an ODE, we can construct and efficiently train models via ODEs. On top of providing a novel family of architectures, notably for invertible density models and continuous time series, neural ODEs also provide a memory efficiency gain in supervised learning tasks.

In this curriculum, we will go through all the background topics necessary to understand these models. At the end, you should be able to implement neural ODEs and apply them to different tasks.

<br />

# Common resources:

1. Süli & Mayers: [An Introduction to Numerical Analysis](https://www.cambridge.org/core/books/an-introduction-to-numerical-analysis/FD8BCAD7FE68002E2179DFF68B8B7237#).
2. Quarteroni et al.: [Numerical Mathematics](https://www.springer.com/us/book/9783540346586?token=holiday18&utm_campaign=3_fjp8312_us_dsa_springer_holiday18&gclid=Cj0KCQiAvebhBRD5ARIsAIQUmnlViB7VsUn-2tABSAhIvYaJgSEqmJXD7F4A7EgyDQtY9v_GeUsNif8aArGAEALw_wcB).

# 1 Numerical solution of ODEs - Part 1
  **Motivation**: ODEs are used to mathematically model a number of natural processes and phenomena. The study of their numerical 
    simulations is one of the main topics in numerical analysis and of fundamental importance in applied sciences. To understand Neural ODEs, we need to first understand how ODEs are solved with numerical techniques.

  **Topics**:

  1. Initial values problems.
  2. One-step methods.
  3. Consistency and convergence.

  **Notes**: In this [class](/assets/nodes_notes/week1.pdf), we touched upon one-step method and their analysis. We also looked at some illustrative examples.

  **Required Reading**:

  1. Sections 12.1-4 from Süli & Mayers.
  2. Sections 11.1-3 from Quarteroni et al.
  
  **Optional Reading**:

  1. Runge-Kutta methods: Section 12.5 from Süli & Mayers.
  2. [Prof. Trefethen's class ODEs and Nonlinear Dynamics 4.2](http://podcasts.ox.ac.uk/odes-and-nonlinear-dynamics-42).

  **Questions**:

  1. Exercise 1 in Section 11.12 of Quarteroni et al.
     <details><summary>Solution</summary>
     <p>
     The truncation error can be split as 
     
     $$h\tau_{n+1} = y_{n+1} - y_n - h\Phi(t_n,y_n;h) = E_1 + E_2$$
     
     where 
     
     $$E_1 = \int_{t_n}^{t_{n+1}} f(s, y(s))\,ds - \frac{h}{2}\left( f(t_n,y_n) + f(t_{n+1},y_{n+1}) \right)$$
     
     and
     
     $$E_2 = \frac{h}{2}\left( f(t_{n+1},y_{n+1}) - f(t_{n+1},y_n + hf(t_n,y_n) \right)$$
     
     We can bound \(E_2\) as
     
     $$|E_2| = \frac{h}{2} \left| f(t_{n+1},y_{n+1}) - f(t_{n+1}, y_n + h f(t_n,y_n)) \right| \leq \frac{hL}{2}|y_{n+1}-y_{n} - hf(t_n,y_n)| = \frac{hL}{2}O(h^2) = O(h^3)$$
     
     where \(L\) is the Lipschitz constant of \(f\). On the other hand, \(E_1\) is bounded above by \(O(h^3)\); see this <a href="https://en.wikipedia.org/wiki/Trapezoidal_rule#Error_analysis">link</a> for a proof. It follows that \(\tau_{n} = O(h^2)\).
     </p>
     </details>

  2. Exercises 12.3,12.4, 12.7 in Section 12 of Süli & Mayers.
     <details><summary>Solution to Exercise 12.3</summary>
     <p>
     Notice that we can write
     
     $$\left(y + \frac{q}{p}\right)'=p\left(y + \frac{q}{p}\right)$$
     
     It follows that \(y(t) = Ce^{pt} - q/p\) for some constant \(C\). Imposing the initial condition \(y(0)=1\), we get \(y(t)=e^{pt} + q/p(e^{pt}-1)\). In particular, we expand \(y\) in its Taylor series: 
     
     $$y(t) = 1 + \left(y + \frac{q}{p}\right)\sum_{k=1}^\infty \frac{(pt)^k}{k!}$$
     
     To conclude the exercise we only need to notice that 
     
     $$y_n(t) = q/p + \left(y + \frac{q}{p}\right)\sum_{k=1}^n \frac{(pt)^k}{k!}$$
     
     satisfies Picard's iteration: \(y_0 \equiv 1\), \(y_{n+1}(t) = y_0 + \int_0^t (py_n(s) + q)\,ds\).
     </p>
     </details>
     <details><summary>Solution to Exercise 12.4</summary>
     <p>
     Applying Euler's method with step-size \(h\), we get \(\hat{y}(0) = 0\), \(\hat{y}(h) = \hat{y}(0) + h \hat{y}(0)^{1/5} = 0\), \(\hat{y}(2h) = \hat{y}(h) + h \hat{y}(h)^{1/5} =0\). Iterating, we see that \(y(nh) = 0\) for all \(n\geq 0\). On the other hand, the implicit Euler's method says that
     
     $$\hat{y}_{n+1} = \hat{y}_n + h \hat{y}_{n+1}^{1/5}$$
     
     for \(n \geq 0\) and \(\hat{y}_0 = 0\). After substituting \(\hat{y}_{n} = (C_nh)^{5/4}\) in the above relation, we only need to check that there exists a sequence \(C_n\) satisfying the requirements.
     </p>
     </details>
     <details><summary>Solution to Exercise 12.7</summary>
     <p>
     First, notice that
     
     $$e_{n+1} = y(x_{n+1}) - y_{n} - \frac{1}{2}h(f_{n+1} + f_n)= e_n - \frac{1}{2}h (f_{n+1}+f_n) + \int_{x_n}^{x_{n+1}} f(s,y(s))\,ds$$
     
     and that the second component of the RHS is the same as \(E_1\) in Exercise 1 above. Therefore the first bound follows. The last inequality is simply obtained by re-arranging the terms.  
     </p>
     </details>

  3. Consider the following method for solving $$y' = f(y)$$:
       
     $$y_{n+1} = y_n + h(\theta f(y_n) + (1-\theta) f(y_{n+1}))$$
    
     Assuming sufficient smoothness of $$y$$ and $$f$$, for what value of $$0 \leq\theta\leq 1$$ is the truncation error the smallest? What does this mean about the accuracy of the method?
     <details><summary>Solution</summary>
     <p>
     By definition, it holds that
     
     $$h\tau_n = y_{n+1} - y_n - h (\theta f_n + (1-\theta) f_{n+1}) = y_{n+1} - y_n - h \theta y_n' - h(1-\theta) y_{n+1}'$$
     
     Taylor-expanding, we get
     
     $$h\tau_n = y_{n} + hy_n' + h^2/2y_n'' + O(h^3) - y_n - h \theta y_n' - h(1-\theta) y_{n}' - h^2(1-\theta) y_{n}'' + O(h^3) = h^2(\theta - 1/2)y_n''+O(h^3)$$
     
     It follows that the truncation error is the smallest for \(\theta=1/2\). For \(\theta = 1/2\), the method has order \(2\), otherwise it has order \(1\).
     </p>
     </details>

  4. [Colab notebook](https://colab.research.google.com/drive/1bNg-RzZoelB3w8AUQ6mefRQuN3AdrIqX).
     <details><summary>Solution</summary>
     <p>
     See this <a href="https://colab.research.google.com/drive/1wTQXy2_4InQH51rEmiCtvl5Q7MiyrC4k">Colab</a>  for the solution.
     </p>
     </details> 

<br />

# 2 Numerical solution of ODEs - Part 2
  **Motivation**: In the previous class, we introduced some simple schemes to numerically solve ODEs. In order to understand which numerical scheme is more proper to apply, it is important to know and understand their different properties. For this reason, in this class, we go through some more involved schemes and analyze them with regards to convergence and stability.

  **Topics**:

  1. Runge-Kutta methods.
  2. Multi-step methods.
  3. System of ODEs and absolute converge.

  **Notes**: In this [class](/assets/nodes_notes/week2.pdf), we went through different ways to construct multi-step methods and their convergence analysis. We then looked into absolute stability regions for different methods. 

  **Required Reading**:

  1. Runge-Kutta methods: Section 11.8 from Quarteroni et al. or Sections 12.{5,12} from Süli & Mayers.
  2. Multi-step methods: Sections 12.6-9 from Quarteroni et al. or Section 11.5-6 from Süli & Mayers.
  3. System of ODEs: Sections 12.10-11 from Quarteroni et al. or Sections 11.9-10 from Süli & Mayers.
  
  **Optional Reading**:

  1. [Prof. Trefethen's class ODEs and Nonlinear Dynamics 4.1](http://podcasts.ox.ac.uk/odes-and-nonlinear-dynamics-41).
  2. Predictor-corrector methods: Section 11.7 from Quarteroni et al.
  3. Richardson extrapolation: Section 16.4 from [Numerical Recipes](http://numerical.recipes/).
  4. [Automatic Selection of Methods for Solving Stiff and Nonstiff Systems of Ordinary Differential Equations](https://epubs.siam.org/doi/pdf/10.1137/0904010?).
  
  **Questions**:

  1. Exercises 12.11, 12.12, 12.19 in Section 12 of Süli & Mayers.
     <details><summary>Solution to Exercise 12.11</summary>
     <p>
     By definition, the truncation error is given by
     
     $$h\tau_n = y_{n+3} + \alpha y_{n+2} -\alpha y_{n+1} - y_n -h\beta y_{n+2}' - h\beta y_{n+1}'$$
     
     Taylor-expanding, we have that
     
     $$y_{n+3} = y_n + 3hy_n' + 9/2h^2 y_n'' + 9/2h^3 y_n''' + 27/8h^4 y_n^{(4)} + O(h^5)$$
     
     $$y_{n+2} = y_n + 2hy_n' + 2h^2 y_n'' + 4/3h^3 y_n''' + 2/3h^4 y_n^{(4)} + O(h^5)$$
     
     $$y_{n+1} = y_n + hy_n' + h^2 y_n'' + h^3 y_n''' + h^4y_n^{(4)} + O(h^5)$$
     
     $$y_{n+2}' = y_n' + 2hy_n'' + 2h^2y_n''' + 4/3 h^3 y_{n}^{(4)}$$
     
     $$y_{n+1}' = y_n' + hy_n'' + h^2y_n''' + h^3 y_{n}^{(4)}$$
     
     Substituting these in the first equation and imposing the terms in \(h^i\), \(i = 0,1,2,3,4\), to be \(0\), we get the equations
     
     $$3 + \alpha - 2\beta = 0$$
     
     $$27 + 7\alpha - 15\beta = 0$$
     
     $$27 + 5\alpha - 12\beta = 0$$
     
     Solving for these, we find \(\alpha = 9\) and \(\beta = 6\). The resulting method reads
     
     $$y_{n+3} + 9(y_{n+2} - y_{n+1}) - y_n = 6h(f_{n+2} + f_{n+1})$$
     
     The characteristic polynomial is given by
     
     $$\rho(z) = z^3 +9z^2 - 9z -1$$
     
     One of the roots of this polynomial satisfies \(|z|>1\) and this implies that the method is not zero-stable.
     </p>
     </details>
     <details><summary>Solution to Exercise 12.12</summary>
     <p>
     By definition, the truncation error is given by
     
     $$h\tau_n = y_{n+1} + b y_{n-1} +a y_{n-2} -h y_{n}'$$
     
     Taylor-expanding, we have that
     
     $$y_{n+1} = y_n + hy_n' + 1/2h^2 y_n'' + O(h^3)$$
     
     $$y_{n-1} = y_n - hy_n' + 1/2h^2 y_n'' + O(h^3)$$
     
     $$y_{n-2} = y_n - 2hy_n' + 2h^2 y_n'' +  O(h^3)$$
     
     Substituting these in the first equation and solving for the terms in \(h^i\), \(i = 0,1\), to be \(0\), we get \(a=1\) and \(b=-2\). In particular
     
     $$\tau_n = 3/2h + O(h^2)$$
     
     and thus the method has order of accuracy \(1\).
     The resulting method reads
     
     $$y_{n+1} -2 y_{n-1} + y_{n-2} = h f_{n}$$
     
     The characteristic polynomial is given by
     
     $$\rho(z) = z^3 -2z -1$$
     
     One of the roots of this polynomial satisfies \(|z|>1\) and this implies that the method is not zero-stable.
     </p>
     </details>
     <details><summary>Solution to Exercise 12.19</summary>
     <p>
     The first equation can be found by substituting \(f(t,y) = \lambda y\) in equation (12.51) in the book and by solving for \(k_1,k_2\) (it is a \(2\times 2\) linear system). Substituting the values of \(A\) and \(b\) from the Butcher tableau in this formula and in the one right before equation (12.51) in the book, and simplifying, we get the formula for \(R(\lambda h)\). Finally, \(p\) and \(q\) are given by \(p,q=-3\pm i \sqrt{3}\). One can see that this implies \(|R(z)|<1\) if \(Re(z) <0\) and thus the method is A-stable.
     </p>
     </details>

<br />

# 3 ResNets
  **Motivation**: The introduction of Residual Networks (ResNets) made it possible to train very deep networks. In this section, we study residual architectures and their properties. We then look into how ResNets approximate ODEs and how this interpretation can motivate neural net architectures and new training approaches.  This is important in order to understand the basic models underlying Neural ODEs and gain some insights into their connection to numerical solutions of ODEs.

  **Topics**:

  1. ResNets.
  2. ResNets and ODEs.

  **Notes**: In this [class](/assets/nodes_notes/week3.pdf), we defined and briefly discussed residual network architecture. We then looked at a stability notion for ResNets, derived from the connection with discretisation of ODEs, and to a simple way to make such architectures reversible.

  **Required Reading**:

  1. ResNets: 
     * [ResNets](https://www.coursera.org/lecture/convolutional-neural-networks/resnets-HAhz9).
     * [An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035).
  2. ResNets and ODEs: 
     * Sections 1-3 from [Multi-level Residual Networks from Dynamical Systems View](https://arxiv.org/pdf/1710.10348.pdf).
     * [Reversible Architectures for Arbitrarily Deep Residual Neural Networks](https://arxiv.org/abs/1709.03698).
     * Invertible ResNets: [The Reversible Residual Network: Backpropagation Without Storing Activations](https://arxiv.org/pdf/1707.04585.pdf)
     * [Stable Architectures for Deep Neural Networks](https://arxiv.org/pdf/1705.03341.pdf).
  
  **Optional Reading**:

  1. The original ResNets paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
  2. Another blog post on ResNets: [Understanding and Implementing Architectures of ResNet and ResNeXt for state-of-the-art Image Classification](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624).
  
  **Questions**:

  1. Do you understand why adding ‘residual layers’ should not degrade the network performance?
     <details><summary>Solution</summary>
     <p>
     Let 
     
     $$x_k = x_{k-1} + f(W_k, x_{k-1})$$
     
     be the output of the \(k\)-th layer of a residual net. Then, adding a residual layer consists of considering $$x_{k+1} = x_{k} + f(W_{k+1}, x_{k})$$ instead of \(x_k\). For most common architectures, it holds that \(f(W, x) \equiv 0\) for \(W=0\). This is why adding a layer should not degrade the performances: any residual network with \(k\) layers can be also written as a residual network with \(k+1\) layers, by simply taking \(W_{k+1}=0\). 
     </p>
     </details>

  2. How do the authors of (Multi-level Residual Networks from Dynamical Systems View) explain the phenomena of still having almost as good performances in residual networks when removing a layer?
     <details><summary>Solution</summary>
     <p>
     Viewing the network output as time-step of the forward Euler's method, we have that 
     
     $$x^{(n+1)}(x_i) = x^{(n)}(x_i) + h F(x^{(n)}(x_i); \theta)$$
     
     where \(x^{(n)}(x_i)\) is the output of the \(n\)-th layer of the network evaluated on the input point \(x_i\). Then
     
     $$x^{(n+2)}(x_i) = x^{(n)}(x_i) + h F(x^{(n)}(x_i); \theta) + h F(x^{(n+1)}(x_i); \theta)$$
     
     Therefore, removing layer \(n+1\) consists of taking
     
     $$x^{(n+2)}(x_i) = x^{(n)}(x_i) + h F(x^{(n)}(x_i); \theta)$$
     
     instead. As \(h\) is small (and this is motivated by the experiments in Section 3.2), the removed term is small and so is the variation in the output layer. Nevertheless, it must be noticed that this analysis is only based on empirical evaluations.
     </p>
     </details>

  3. Implement your favourite ResNet variant.
     <details><summary>Example</summary>
     <p>
     See this <a href="https://keras.io/examples/cifar10_resnet/">tutorial</a> for an example of implementation of a ResNet.
     </p>
     </details>

<br />

# 4 Normalising Flows
  **Motivation**:  In this class, we take a little detour to learn about Normalising Flows. These are used for density estimation and generative modeling, and their implementation is motivated by a discretisation of an ODE. Understanding it at a basic level is necessary to understanding continuous normalizing flows, a central application of neural ODEs.

  **Topics**:

  1. Normalising Flows.
  2. End-to-end implementations with neural nets.
  
  **Notes**: In this [class](/assets/nodes_notes/week4.pdf), we defined nomalising flow, starting from the non-parametric form and then deriving their algorithmic (and parametric) implementation. We concluded by discussing some architectures proposed in the literature and their trade-offs.

  **Required Reading**:

1. *DE*: [Density Estimation by Dual Ascent of the Log-likelihood](https://math.nyu.edu/faculty/tabak/publications/CMSV8-1-10.pdf) (Skip Section 3).
  2. [A family of non-parametric density estimation algorithms](https://math.nyu.edu/faculty/tabak/publications/Tabak-Turner.pdf).
  3. [A post on Normalising flow](http://akosiorek.github.io/ml/2018/04/03/norm_flows.html).
  
  **Optional Reading**:

  1. [Variational Inference with Normalizing Flows](https://arxiv.org/pdf/1505.05770.pdf).
  2. [High-Dimensional Probability Estimation with Deep Density Models](https://arxiv.org/pdf/1302.5125.pdf).
  
  **Questions**:

  1. In *DE*, what is the difference between $$\rho_t$$ and $$\tilde{\rho}_t$$, i.e. what do they represent?
     <details><summary>Solution</summary>
     <p>
     The function \(\tilde{\rho}_t\) is the density of the distribution of the random variable \(\phi_t^{-1}(y)\) where \(y\sim \mu\). The function \(\rho_t\) is the density of the distribution of the random variable \(\phi_t(x)\) where \(x\sim \rho\).
     </p>
     </details>

  2. What is the computational complexity of evaluating a determinant of an $$N\times N$$ matrix, and why is that relevant in this context?
     <details><summary>Solution</summary>
     <p>
     In general, the cost of computing the determinant of an \(N\times N\) matrix is \(O(N^3)\). To compute densities  transported by normalising flows, we need to compute the determinants of the Jacobians; therefore, an important feature of practical normalising flows, is that the Jacobian structure must allow an efficient computation of its determinant. See this week notes for more discussion on this.  
     </p>
     </details>

<br />

# 5 The Adjoint Method (and Auto-Diff)
  **Motivation**: The adjoint method is a numerical method for efficiently computing the gradient of a function in numerical optimization problems. Understanding this method is essential to understand how to train ‘continuous depth’ nets. We also review the basics of Automatic Differentiation, which will help us understand the efficiency of the algorithm proposed in the NeuralODE paper. 

  **Topics**:

  1. Adjoint Method.
  2. Auto-Diff.

  **Notes**: In this [class](/assets/nodes_notes/week5.pdf), we discussed the adjoint method. We started from the case of linear system and went through non-linear equations and recurrent relations. We concluded by discussing their application to ODE constrained optimization problems, which is the case of interest for Neural ODEs.

  **Required Reading**:

  1. Section 8.7 from *CSE*: [Computational Science and Engineering](http://math.mit.edu/~gs/cse/).
  2. Sections 2 and 3 from [Automatic Differentiation in Machine Learning: a Survey](http://www.jmlr.org/papers/volume18/17-468/17-468.pdf).
  
  **Optional Reading**:

  1. [Prof. Steven G. Johnson's notes on adjoint method](http://math.mit.edu/~stevenj/notes.html).
  
  **Questions**:

  1. Exercises 1,2,3 from Section 8.7 of *CSE*.
     <details><summary>Solution to Exercise 1</summary>
     <p>
     This follows immediately by noticing that the number of multiply-add operations of multiplying an \(N\times M\) matrix with an \(M\times P\) matrix is given by \(O(NMP)\). 
     </p>
     </details>
     <details><summary>Solution to Exercise 2</summary>
     <p>
     Apply the chain rule. Since \(\frac{\partial C}{\partial S} = 2S\) and \(\frac{dT}{dS} = \frac{\partial T}{\partial S} + \frac{\partial T}{\partial C}\frac{\partial C}{\partial S}\), we get \(\frac{d T}{d S} = 1 -2S\).  
     </p>
     </details>
     <details><summary>Solution to Exercise 3</summary>
     <p>
     This follows from Exercise 1 by seeing \(u^T\) and \(w^T\) as \(1\times N\) matrices and \(v\) as an \(N\times 1\) matrix. 
     </p>
     </details>

2. Consider the problem of optimizing a real-valued function $$g$$ over the solution of the ODE $$y'(t) = A(p)y(t)$$, $$y(0) = b(p)$$ at time $$T>0$$: $$\min_p\, g(T) \doteq g(y(T; p))$$. Find $$\frac{dg(T)}{dp}$$ by solving the ODE and by applying chain rule. Check the correctness of equations (16-17) in *CSE*.
     <details><summary>Solution</summary>
     <p>
     It holds that
     
     $$y(t) = e^{tA(p)}y(0)$$
     
     Applying the chain rule, we get
     
     $$\frac{dg}{dp} = \frac{dg}{dy}e^{TA(p)}\frac{db}{dp} + T\frac{dg}{dy}\frac{\partial A}{\partial p}e^{TA(p)}b(p)$$
     
     On the other hand, the adjoint ODE reads
     
     $$\lambda'(t) = -A(p)^T\lambda(t)$$
     
     with the final condition \(\lambda(T) = \left(\frac{\partial g}{\partial y}\right)^T\), which gives \(\lambda(t) = e^{A(p)^T(T-t)}\left(\frac{\partial g}{\partial y}\right)^T\). Equation (17) from <i>CSE</i> gives
     
     $$\frac{dg}{dp} = \left(e^{TA(p)^T}\left(\frac{\partial g}{\partial y}\right)^T\right)^T\frac{\partial b}{\partial p} + \int_0^T \frac{\partial g}{\partial y} e^{A(p)(T-t)}\frac{\partial A}{\partial p}e^{tA(p)}b(p)\,dt$$
     
     which coincides with the above.
     </p>
     </details>

  3. Prove equations (14-15) in Section 8.7 of *CSE*.
     <details><summary>Solution</summary>
     <p>
     By definition, it holds that
     
     $$\frac{dG}{dp} = \int_0^T\left(\frac{\partial g}{\partial p} + \frac{\partial g}{\partial u}\frac{\partial u}{\partial p}\right)\,dt $$
     
     On the other hand, it holds that
     
     $$\lambda(0)^T\frac{\partial u}{\partial p}(0) + \int_0^T\lambda^T \frac{\partial f}{\partial p}\,dt = \int_0^T \left( \lambda^T\frac{\partial f}{\partial p} -\frac{d}{dt}\left( \lambda^T \frac{\partial u}{\partial p}\right) \right)\,dt $$
     
     Using equation (14) from <i>CSE</i> and the equality \(\frac{\partial u}{\partial p} = \frac{\partial f}{\partial p} + \frac{\partial f}{\partial u}\frac{\partial u}{\partial p}\), we get
     
     $$\int_0^T \left( \lambda^T\frac{\partial f}{\partial p} -\frac{d}{dt}\left( \lambda^T \frac{\partial u}{\partial p}\right) \right)\,dt = \int_0^T \left( \lambda^T\frac{\partial f}{\partial p} + \lambda^T \frac{\partial f}{\partial u}\frac{\partial u}{\partial p} + \frac{\partial g}{\partial u}\frac{\partial u}{\partial p} - \lambda^T \frac{\partial f}{\partial p} -\lambda^T \frac{\partial f}{\partial u}\frac{\partial u}{\partial p} \right)\,dt$$
     
     which gives 
     
     $$
     \lambda(0)^T\frac{\partial u}{\partial p}(0) + \int_0^T \lambda^T\frac{\partial f}{\partial p}\,dt = \int_0^T \frac{\partial g}{\partial u}\frac{\partial u}{\partial p}\,dt
     $$
     
     and thus completes the proof. 
     </p>
     </details>

<br />

# 6 The Paper
  **Motivation**: Let’s read the paper! Here is a summary of what’s going on to help with your understanding:
  
  Any residual network can be seen as the Explicit Euler's method discretisation of a certain ODE; given the network parameters, any numerical ODE solver can be used to evaluate the output layer. The application of the adjoint method makes it possible to efficiently back-propagate (and thus train) these models. The same idea can be used to train time-continuous normalising flows. In this case, moving to the continuous formulation allows us to avoid the computation of the determinant of the Jacobian, one of the major bottlenecks of normalising flows. Neural ODEs can also be used to model latent dynamics in time-series modeling, allowing us to easily tackle irregularly sampled data.

  **Topics**:

  1. Normalising Flows.
  2. End-to-end implementations with neural nets.

  **Notes**: In this [class](/assets/nodes_notes/week6.pdf), we defined Neural ODEs and derived the respective adjoint method, essential for their implementation. We then discussed continuous normalising flows and the computational advantages offered by Neural ODEs in this setting.
  
  **Required Reading**:

  1. [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366).
  2. [A blog post on NeuralODEs](https://rkevingibson.github.io/blog/neural-networks-as-ordinary-differential-equations/).
  
  **Optional Reading**:

  1. A follow-up paper by the authors on scalable continuous normalizing flows: [Free-form Continuous Dynamics for Scalable Reversible Generative Models](https://arxiv.org/abs/1810.01367).

<br />
