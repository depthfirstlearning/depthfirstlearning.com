---
layout: post
title:  "Wasserstein GAN"
date:   2019-03-23 10:00:00 -0400
categories: generative-adversarial-networks
author: james
blurb: "The Wasserstein GAN (WGAN) is a GAN variant which uses the 1-Wasserstein distance, rather than the KL-Divergence, to measure the difference between the model and target distributions. This seemingly simple change has big consequences! Not only does WGAN train more easily but it also achieves very impressive results &#8212; generating some stunning images."
feedback: true
---

A number of people need to be thanked for their parts in making this happen. Thank you to Martin Arjovsky, Avital Oliver, Cinjon Resnick, Marco Cuturi, Kumar Krishna Agrawal, and Ishaan Gulrajani for contributing to this guide. 

Of course, thank you to Sasha Naidoo, Egor Lakomkin, Taliesin Beynon, Sebastian Bodenstein, Julia Rozanova, Charline Le Lan, Paul Cresswell, Timothy Reeder, and Michał Królikowski for beta-testing the guide and giving invaluable feedback. A special thank you to Martin Arjovsky, Tim Salimans, and Ishaan Gulrajani for joining us for the weekly meetings.

Finally, thank you to Ulrich Paquet and Stephan Gouws for introducing me to Cinjon.

<div class="deps-graph">
  <iframe class="deps" src="/assets/wgan-deps.svg" width="400"></iframe>
  <div>Concepts used in Wasserstein GAN. Click to navigate.</div>
</div>

# Why

The Wasserstein GAN (WGAN) is a GAN variant which uses the 1-Wasserstein distance, rather than the KL-Divergence, to measure the difference between the model and target distributions. This seemingly simple change has big consequences! Not only does WGAN train more easily (a common struggle with GANs) but it also achieves very impressive results &#8212; generating some stunning images. By studying the WGAN, and its variant the WGAN-GP, we can learn a lot about GANs and generative models in general. After completing this curriculum you should have an intuitive grasp of why the WGAN and WGAN-GP work so well, as well as, a thorough understanding of the mathematical reasons for their success. You should be able to apply this knowledge to understanding cutting edge research into GANs and other generative models.

<br />

# 1 Basics of Probability & Information Theory
  **Motivation**: To understand GAN training (and eventually WGAN & WGAN-GP) we need to first have some understanding of probability and information theory. In particular, we will focus on Maximum Likelihood Estimation and the KL-Divergence. This week we will make sure that we understand the basics so that we can build upon them in the following weeks.  

  _This week contains some fairly introductory material. If everyone participating in this curriculum is comfortable with this material you may wish to treat this week as optional or as a pre-requisite. However, this week still covers some very interesting and cool topics, and it is important to have a solid grasp of these concepts in order to build towards understanding the Wasserstein GAN, so skip with caution._

  **Topics**:

  1. Probability Theory
  2. Information Theory
  3. Mean Squared Error (MSE)
  4. Maximum Likelihood Estimation (MLE)

  **Required Reading**:

  1. Chs 3.1 - 3.5 of [Deep Learning](https://www.deeplearningbook.org/) by Goodfellow _et. al_ (the DL book)
     * These chapters are here to introduce fundamental concepts such as random variables, probability distributions, marginal probability, and conditional probability. If you have the time, reading the whole of chapter 3 is highly recommended. A solid grasp of these concepts will be important foundations for what we will cover over the next 5 weeks.
  2. Ch 3.13 of the DL book
      * This chapter covers KL-Divergence & the idea of distances between probability distributions which will also be a key concept going forward.
  3. Chs 5.1.4 and 5.5 of the DL book
      * The aim of these chapters is to make sure that everyone understands Maximum Likelihood Estimation which is a fundamental concept in machine learning. It is used explicitly or implicitly in both supervised and unsupervised learning as well as in both discriminative and generative methods. In fact, many methods using gradient descent are doing approximate MLE. It is important to understanding MLE as a fundamental concept, and its use in machine learning in practice. Note that, if you are not familiar with the notation used in these chapters, you might want to start at the beginning of the chapter. Also note that, if you are not familiar with the concept of estimators, you might want to read Ch 5.4. However, you can probably get by simply knowing that minimizing MSE is a method for optimizing some approximation for a function we are trying to learn (an estimator).

  
  **Optional Reading**:

  1. Ch 2 from [Information Theory, Inference & Learning Algorithms by David MacKay](http://www.inference.org.uk/itprnn/book.pdf) (MacKay's book)
      * This is worth reading if you feel like you didn’t quite grok the probability and information theory content in the DL book. MacKay provides a different perspective on these ideas which might help make things click. These concepts are going to be crucial going forward so it is definitely worth making sure you are comfortable with them.
  2. Chs 1.6 and 10.1 of [Pattern Recognition and Machine Learning by Christopher M. Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (PRML)
     * Similarly, this is worth reading if you don’t feel comfortable with the KL-Divergence and want another perspective.
  3. Aurélien Géron's video [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8)
     * An introductory, but interesting video that describes the KL-Divergence.
  4. [Notes on MLE and MSE](http://people.math.gatech.edu/~ecroot/3225/maximum_likelihood.pdf)
     * An alternative discussion on the links between MLE and MSE.
  5. The first 37ish minutes of Arthur Gretton's MLSS Africa talk on comparing probability distributions &#8212; [video](https://www.youtube.com/watch?v=5sijxSg8P14), [slides](https://drive.google.com/file/d/1RNrgDs5xw-9HTjikFU1L0iO1PBMDaGwE/view)
     * An interesting take on comparing probability distributions. The first 37 minutes are fairly general and give some nice insights as well as some foreshadowing of what we will be covering in the following weeks. The rest of the talk is also very interesting and ends up covering another GAN called the MMD-GAN, but it isn’t all that relevant for us.
  6. [On Integral Probability Metrics, φ-Divergences and Binary Classification](https://pdfs.semanticscholar.org/6af2/fa8887a2cb0386f79e3a2822b661e2dc8369.pdf)
     * For those of you whose curiosity was peaked by Arthur’s talk, this paper goes into depth describing IPMs (such as MMD and the 1-Wasserstein distance) and comparing them the φ-divergences (such as the KL-Divergence). *This paper is fairly heavy mathematically so don't be discouraged if you struggle to follow it*.

  **Questions**:

  1. Examples/Exercises 2.3, 2.4, 2.5, 2.6, and 2.26 in MacKay's book
      * Bonus: 2.35, and 2.36
     <details><summary>Solutions</summary>
     <p>
     Examples 2.3, 2.5, and 2.6 have their solutions directly following them.
     </p>
     <p>
     Exercise 2.26 has a solution on page 44.
     </p>
     <p>
     Exercise 2.35 has a solution on page 45.
     </p>
     <p>
     Exercise 2.36: 1/2 and 2/3.
     </p>
     <p>
     (Page numbers from Version 7.2 (fourth printing) March 28, 2005, of MacKay's book.)
     </p>
     </details>

  2. Derive Bayes' rule using the definition of conditional probability.
      <details><summary>Solution</summary>
      <p>
      The definition of conditional probability tells us that 

       $$p(y|x) = \frac{p(y,x)}{p(x)}$$ 

      and that 

       $$p(x|y) = \frac{p(y,x)}{p(y)}.$$ 
      
      From this we can see that \(p(y,x) = p(y|x)p(x) = p(x|y)p(y)\). Finally if we divide everything by \(p(x)\) we get

       $$p(y|x) = \frac{p(x|y)p(y)}{p(x)}$$

      which is Bayes' rule.
      </p>
      </details>

  3. Exercise 1.30 in PRML
       <details><summary>Solution</summary>
       <p>
       <a href="https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians?rq=1">Here</a> is a solution.
       </p>
       <p> 
       The result should be \(\log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}\).
       </p>
       </details>
       

  4. Prove that minimizing MSE is equivalent to maximizing likelihood (assuming Gaussian distributed data).
       <details><summary>Solution</summary>
       <p>
       Mean squared error is defined as 
       
       $$MSE = \frac{1}{N}\sum^N_{n=1}(\hat{y}_n - y_n)^2$$
       
       where \(N\) is the number of examples, \(y_n\) are the true labels, and \(\hat{y}_n\) are the predicted labels.

       Log-likelihood is defined as \(LL = \log(p(\mathbf{y}|\mathbf{x}))\). Assuming that the examples are independent and identically distributed (i.i.d.) we get 

       $$ LL = \log\prod_{n=1}^Np(y_n|x_n) = \sum_{n=1}^{N}\log p(y_n|x_n). $$

       Now, substituting in the definition of the normal distribution 

       $$ \mathcal{N}(y;\mu,\sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp{-\frac{(y - \mu)^2}{2\sigma^2}}$$

       for \(p(y_n|x_n)\) and simplifying the expression, we get

       $$ LL = \sum_{n=1}^{N} -\frac{1}{2}\log(2\pi) - \log\sigma - \frac{(y_n - \mu_n)^2}{2\sigma^2}.$$

       Finally, replacing \(\mu\) with \(\hat{y}\) (because we use the mean as our prediction), and noticing that maximizing the expression above depends only on the third term (because the others are constants), we arrive at the conclusion that to maximize the log-likelihood we must minimize

       $$ \frac{(y_n - \hat{y}_n)^2}{2\sigma^2} $$

       which is the same as minimising the MSE.
       </p>
       </details>

  5. Prove that maximizing likelihood is equivalent to minimizing KL-Divergence.
     <details><summary>Solution</summary>
     <p>
     KL-Divergence is defined as 

     $$ D_{KL}(p||q) = \sum_x p(x) \log\frac{p(x)}{q(x|\bar{\theta})}$$

     where \(p(x)\) is the true data distribution, \(q(x|\bar{\theta})\) is our model distribution, and \(\bar{\theta}\) are the parameters of our model. We can rewrite this as

     $$ D_{KL}(p||q) = \mathbb{E}_p[\log p(x)] - \mathbb{E}_p[\log q(x|\bar{\theta})]$$

     where the notation \(\mathbb{E}_p[f(x)]\) means that we are taking the expected value of \(f(x)\) by sampling \(x\) from \(p(x)\). We notice that minimizing \(D_{KL}(p||q)\) means maximizing \(\mathbb{E}_p[\log q(x|\bar{\theta})]\) since the first term in the expression above is constant (we can't change the true data distribution). Now, to maximize the likelihood of our model, we need to maximize
     
     $$q(\bar{x}|\bar{\theta}) = \prod_{n=1}^Nq(x_n|\bar{\theta}).$$

     Recall that taking a logarithm does not change the result of optimization which means that we can maximize

     $$\log q(\bar{x}|\bar{\theta}) = \sum_{n=1}^N\log q(x_n|\bar{\theta}).$$

     If we divide this term by a constant factor of \(N\) we the same term that would minimize the to maximize the KLD: \(\mathbb{E}_p[\log q(x|\bar{\theta})]\).
     </p>
     </details>

**Notes**: Here is a [link](todo:add_link) to our notes for the lesson. We were fortunate enough to have Martin Arjovsky sit in on the session!

<br />

# 2 Generative Models
  **Motivation**: This week we’ll take a look at generative models. We will aim to understand how they are similar and how they differ from discriminative models. In particular, we want to understand the challenges that come with training generative models.

  **Topics**:

  1. Generative Models
  2. Evaluation of Generative Models

  **Required Reading**:

  1. The "Overview", "What are generative models?", and "Differentiable inference" sections of the webpage for David Duvenaud’s [course on Differentiable Inference and Generative Models](https://www.cs.toronto.edu/~duvenaud/courses/csc2541/index.html).
      * Here we want to get a sense of the big picture of what generative models are all about. There are also some fantastic resources here for further reading if you are interested.
  2. [A note on the evaluation of generative models](https://arxiv.org/pdf/1511.01844.pdf)
      * This paper is the real meat of this week’s content. After reading this paper you should have a good idea of the challenges involved in evaluating (and therefore training) generative models. Understanding these issues will be important for appreciating what the WGAN is all about. Don’t worry too much if some sections don’t completely make sense yet - we’ll be returning to the key ideas in the coming weeks.

  **Optional Reading**:

  1. Ch 20 of the DL book, particularly:
      * Differentiable Generator Networks (Ch 20.10.2)
          * Description of a broad class of generative models to which GANs belong which will help contextualize GANs when we look at them next week.
      * Variational Autoencoders (Ch 20.10.3)
          * Description of another popular class of differentiable generative model which might be nice to contrast to GANs next week.
      * Evaluating Generative Models (Ch 20.14)
          * Summary of techniques and challenges for evaluating generative models which might put Theis _et al._’s paper into context.

  **Questions**:

  _These first two questions are here to make sure that you understand what a generative model is and how it differs from a discriminative model_ 

  1. Fit a [multivariate Gaussian distribution](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html) to the [Fisher Iris dataset](https://scikit-learn.org/stable/datasets/index.html#iris-plants-dataset) using maximum likelihood estimation (see Section 2.3.4 of PRML for help) then:
      1. Determine the probability of seeing a flower with a sepal length of 7.9, a sepal width of 4.4, a petal length of 6.9, and a petal width of 2.5.
      2. Determine the distribution of flowers with a sepal length of 6.3, a sepal width of 4.8, and a petal length of 6.0 (see section 2.3.2 of PRML for help). 
      3. Generate 20 flower measurements.
      4. Generate 20 flower measurements with a sepal length of 6.3.

      (congrats you’ve just trained and used a generative model)
      <details><summary>Solution</summary>
      <p>
      <a href="https://github.com/JamesAllingham/DFL-WGAN/blob/master/DFL_WGAN_week2.ipynb">Here</a> is a Jupyter notebook with solutions. Open the notebook on your computer or Google colab to render the characters properly.
      </p>
      </details>

  2. Describe in your own words the difference between a generative and a discriminative model.
      <details><summary>Solution</summary>
      <p>
      This is an open ended question but here are some of the differences:
      <ul>
        <li>In the generative setting, we usually model \(p(x)\), our models are usually non-deterministic, and we can sample from them.</li>
        <li>In the discriminative setting, we usually model \(p(y|x)\), our models are often deterministic, and we can't necessarily sample from them.</li>
      </ul>
      </p>
      </details>
 
  _These last two questions are a good barometer for determining your understanding of the challenges involved in training generative models._ 

  3. Theis _et al._ claim that “a model with zero KL divergence will produce perfect samples” &#8212; why is this the case?
      <details><summary>Solution</summary>
      <p>
      As we showed last week, \(D_{KL}(p||q) = 0\) if and only if \(p(x)\), the true data distribution, and \(q(x)\) the model distribution, are the same. 
      </p>
      <p>
      Therefore, if \(D_{KL}(p||q) = 0\), samples from our model will be indistinguishable from the real data.
      </p>
      </details>

  4. Explain why the high log-likelihood of a generative model might not correspond to realistic samples?
     <details><summary>Solution</summary>
      <p>
      Theis <i>et al.</i> outlined two scenarios where this is the case:
      <ul>      
      <li><b>Low likelihood & good samples</b>: our model can overfit to the training data and produce good samples, however, because the model has overfitted it will have a low likelihood for unseen test data.</li>
      <li><b>High likelihood & poor samples</b>: here the issue is that high dimensional data will tend to have higher log-likelihoods than low dimensional data. </li>
      </ul>
      </p>
      </details>

  **Notes**: Here is a [link](todo:add_link) to our notes for the lesson. We were fortunate enough to have Tim Salimans sit in on the session!

  <br />

# 3 Generative Adversarial Networks

  **Motivation**: Let’s read the original GAN paper. Our main goal this week is to understand how GANs solve some of the problems with training generative models, as well as, some of the new issues that come with training GANs.

  _This week has a fairly heavy workload and could plausibly be split over two weeks. This is due to having two papers for required reading. However, it should be noted that the second paper is really somewhere between required and optional &#8212; we think that it contains some  interesting material and sets the state well for looking at WGAN in week 4, but the most important points are covered again in week 4. Depending on your study group you might want to focus on this paper more or less based on your interest (we recommend that most groups don't spend too much time on this paper)._

  **Topics**:

  1. Generative Adversarial Networks
  2. The Jensen-Shannon Divergence (JSD)
  3. Why training GANs is hard

  **Required Reading**:

  1. [Goodfellow's GAN paper](https://arxiv.org/pdf/1406.2661.pdf)
      * This is the paper the started it all and if we want to understand WGAN & WGAN-GP we’d better understand the original GAN.
  2. [Toward Principled Methods for Generative Adversarial Network Training](https://arxiv.org/pdf/1701.04862.pdf)
      * This paper explores the difficulties in training GANs and is a precursor to the WGAN paper that we will look at next week. The paper is quite math heavy so unless math is your cup of tea you shouldn’t spend too much time trying to understand the details of the proofs, corollaries, and lemmas. The important things to understand here are: what is the problem, and how do the proposed solutions solve the problem. Focus on the introduction, the English descriptions of the theorems and the figures. **Don't spend too much time on this paper**.

  **Optional Reading**:

  1. [Goodfellow's tutorial on GANs](https://arxiv.org/pdf/1701.00160.pdf)
      * A more in-depth explanation of GANs from the man himself.
  2. The GAN chapter in the DL book (20.10.4) 
      * A summary of what a GAN is and some of the issues involved in GAN training.
  3. Coursera (Stanford) course on game theory videos: [1-05](https://www.youtube.com/watch?v=-j44yHK0nn4&index=5&list=PLGdMwVKbjVQ8DhP8dgrBO1B5etE81Hxxh), [2-01](https://www.youtube.com/watch?v=BsgnKTfOxTs&list=PLGdMwVKbjVQ8DhP8dgrBO1B5etE81Hxxh&index=11), [2-02](https://www.youtube.com/watch?v=FU6ax5K9HIA&list=PLGdMwVKbjVQ8DhP8dgrBO1B5etE81Hxxh&index=12), and [3-04b](https://www.youtube.com/watch?v=RIneClCKgAw&list=PLGdMwVKbjVQ8DhP8dgrBO1B5etE81Hxxh&index=22)
      * This is really here just for people who are interested in the game theory ideas such as minmax. 
  4. Finish reading [GANs and Divergence Minimization](https://colinraffel.com/blog/gans-and-divergence-minimization.html#citation-gulrajani2018).
      * Now that we know what a GAN is it will be worth it to go back and finish reading this blog. It should help to tie together many of the concepts we’ve covered so far. It also has some great resources for extra reading at the end.
  5. [Overview: Generative Adversarial Networks – When Deep Learning Meets Game Theory](https://ahmedhanibrahim.wordpress.com/2017/01/17/generative-adversarial-networks-when-deep-learning-meets-game-theory/comment-page-1/)
      * A short blog post which briefly summarises many of the topics we’ve covered so far.
  6. [How to Train your Generative Models? And why does Adversarial Training work so well?](https://www.inference.vc/how-to-train-your-generative-models-why-generative-adversarial-networks-work-so-well-2/) and [An Alternative Update Rule for Generative Adversarial Networks](https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/)
      * Two great blog posts from Ferenc Huszar that discuss the challenges in training GANs as well as the differences between the JSD, KLD and reverse KLD.
  7. [Simple Python GAN example](https://github.com/HIPS/autograd/blob/master/examples/generative_adversarial_net.py)
      * This example illustrates how simple GANs are to implement by doing it in 145 lines of Python using Numpy and a simple autograd library.

  **Questions**:

  1. Prove that minimizing the optimal discriminator loss, with respect to the generator model parameters, is equivalent to minimizing the JSD.
      * Hint, it may help to somehow introduce the distribution $$p_m(x) = \frac{p_d(x) + p_g(x)}{2}$$.
      <details><summary>Solution</summary>
      <p>
      The loss we are minimizing is

      $$\mathbb{E}_{x \sim p_d(x)}[\log D^*(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D^*(G(x)))]$$ 

      where \(p_d(x)\) is the true data distribution, \(p_z(z)\) is the noise distribution from which we draw samples to pass through our generator, \(D\) and \(G\) are the discriminator and generator, and \(D^*\) is the optimal discriminator which has the form:

      $$ D^*(x) = \frac{p_d(x)}{p_d(x) + p_g(x)}.$$

      Here \(p_g(x)\) is the distribution of the data sampled from the generator. Substiting in \(D^*(x)\) and \(p_g(x)\), we can rewrite the loss as

      $$ \mathbb{E}_{x \sim p_d(x)}[\log \frac{p_d(x)}{p_d(x) + p_g(x)}] + \mathbb{E}_{x \sim p_g(x)}[\log \frac{p_g(x)}{p_d(x) + p_g(x)}]. $$

      Now we can multiply the values inside the logs by \(1 = \frac{0.5}{0.5}\) to get

      $$ \mathbb{E}_{x \sim p_d(x)}[\log \frac{0.5 p_d(x)}{0.5(p_d(x) + p_g(x))}] + \mathbb{E}_{x \sim p_g(x)}[\log \frac{0.5 p_g(x)}{0.5(p_d(x) + p_g(x))}]. $$

      Recall that \(\log(ab) = \log(a + b)\) and define \(p_m(x) = \frac{p_d(x) + p_g(x)}{2}\), we now get

      $$ \mathbb{E}_{x \sim p_d(x)}[\log \frac{p_d(x)}{p_m(x)}] + \mathbb{E}_{x \sim p_g(x)}[\log \frac{p_g(x)}{p_m(x)}] - 2\log2. $$

      Using the definition of the KL-Divergence, this simplifies to

      $$ D_{KL}(p_d||p_m) + D_{KL}(p_g||p_m) - 2\log2. $$

      Finally, using the definition of the JS-Divergence and noting that for the purposes of minimization the \(2\log2\) term can be ignored, we get

      $$ D_{JS}(p_d||p_g).$$
      </p>
      </details>

  2. Explain why Goodfellow says that $$D$$ and $$G$$ are playing a two-player minmax game and derive the definition of the value function $$V(G,D)$$.
     <details><summary>Solution</summary>
      <p>
      \(G\) wants to maximize the probability that \(D\) thinks the generated samples are real \(\mathbb{E}_{z \sim p_z(z)}[D(G(z))]\). This is the same as minimizing the probability that \(D\) thinks the generated samples are not fake \(\mathbb{E}_{z \sim p_z(z)}[1 - D(G(z))]\). 
      </p>
      <p>
      On the other hand, \(D\) wants to maximise the probability that it assigns the labels correctly \(\mathbb{E}_{x \sim p_d(x)}[D(x)] + \mathbb{E}_{z \sim p_z(z)}[1 - D(G(z))]\). Note that \(D(x)\) should be 1 if \(x\) is real, and 0 if \(x\) is fake.
      </p>
      <p>
      We can take logs without changing the optimization, which gives

      $$ V(G,D) = \min_G\max_D \mathbb{E}_{x \sim p_d(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]. $$
      </p>
     </details>


  3. Why is it important to carefully tune the amount that the generator and discriminator are trained in the original GAN formulation? 
        * Hint, it has to do with the approximation for the JSD & the dimensionality of the data manifolds.
        <details><summary>Solution</summary>
        <p>
        If we train the discriminator too much we get vanishing gradients. This is due to the fact that when the true data distribution and model distribution lie on low dimensional manifolds (or have disjoint support almost everywhere), the optimal discriminator will be perfect &#8212; i.e. the gradient will be zero almost everywhere. This is something that almost always happens.
        </p>
        <p>
        On the other hand, if we train the discriminator too little, then the loss for the generator no longer approximates the JSD. This is because the approximation only holds if the discriminator is near the optimal \(D^*(x) = \frac{p)d(x)}{p_d(x) + p_g(x)}\). 
        </p>
        </details>

  4. Implement a GAN and train it on Fashion MNIST.
      * [This notebook](https://colab.research.google.com/drive/1OWZEeF-SB0r1f6mHm-7-hfxd2zsecEwq#scrollTo=Q8YoJ4mejp97) contains a skeleton with boilerplate code and hints.
      * Try various settings of hyper-parameters, other than those suggested, and see if the model converges.
      * Examine samples from various stages of the training. Rank them without looking at the corresponding loss and see if your ranking agrees with the loss.
      <details><summary>Solution</summary>
        <p>
        <a href="https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py">Here</a> is a GAN implementation using Keras.
        </p>
      </details>
  
  **Notes**: Here is a [link](todo:add_link) to our notes for the lesson. We were fortunate enough to have Martin Arjovsky sit in on the session!

  <br />

# 4 Wasserstein GAN

  **Motivation**: Last week we saw how GANs solve some problems in training generative models but also that they bring in new problems. This week we’ll look at the Wasserstein GAN which goes a long way to solving these problems.

  **Topics**:

  1. Wasserstein Distance vs KLD/JSD
  2. Wasserstein GAN

  **Required Reading**:

  1. [The WGAN paper](https://arxiv.org/pdf/1701.07875.pdf)
      * This should be pretty self-explanatory! We’re doing a DFL on Wasserstein GANs so we’d better read the paper! (This isn’t the end of the road, however, next week we’ll look at WGAN-GP.) The paper builds upon an intuitive idea: the family of Wasserstein distances is a nice distance between probability distributions, that is well grounded in theory. The authors propose to use the 1-Wasserstein distance to estimate generative models. They show that the 1-Wasserstein distance is an IPM with a meaningful set of constraints (1-Lipschitz functions), and can, therefore, be optimized by focusing on discriminators that are “well behaved” (meaning that their output does not change to much if you perturb the input, i.e. they are Lipschitz!).

  **Optional Reading**:

  1. [Summary blog for the paper](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html)
      * This is a brilliant blog post that summarises almost all of the key points we’ve covered over the last 4 weeks and puts them in the context of the WGAN paper. In particular, if any of the more theoretic aspects of the WGAN paper were a bit much for you then this post is worth reading.
  2. [Another good summary of the paper](https://mindcodec.ai/2018/09/23/an-intuitive-guide-to-optimal-transport-part-ii-the-wasserstein-gan-made-easy/)
  3. Wasserstein / Earth Mover distance [blog](https://vincentherrmann.github.io/blog/wasserstein/) [posts](https://mindcodec.ai/2018/09/19/an-intuitive-guide-to-optimal-transport-part-i-formulating-the-problem/)
  4. [Set of](https://www.youtube.com/watch?v=6iR1E6t1MMQ) [three](https://www.youtube.com/watch?v=1ZiP_7kmIoc) [lectures](https://www.youtube.com/watch?v=SZHumKEhgtA) by Marco Cuturi on optimal transport (with accompanying [slides](https://drive.google.com/file/d/1oYX41dIAXhU6EShcid6eYrrK7svi5NXW/view))
      * If you are interested in the history of optimal transport and would like to see where the KR duality comes from (that’s the crucial argument in the WGAN paper which connects the 1-Wasserstein distance to an IPM with a Lipschitz constraint) the Wasserstein distance, or if you feel like you need a different explanation of what the Wasserstein distance and the Kantorovich-Rubinstein duality are, then watching these lectures is recommended. There are some really cool applications of optimal transport here too, and a more exhaustive description of other families of Wasserstein distances (such as the quadratic one) and their dual formulation.
  5. The first 15 or so minutes of [this lecture on GANs](https://www.youtube.com/watch?v=eDWjfrD7nJY) by Sebastian Nowozin
      * Great description of WGAN, including Lipschitz and KR duality. This lecture is actually part 2 of a series of 3 lectures from MLSS Africa. Watching the whole series is also highly recommended if you are interested in knowing more about the bigger picture for GANs (including other interesting developments and future work) and how WGAN relates to other GAN variants. However, to avoid spoilers for next week, you should wait to watch the rest of part 2.
  6. [Computational Optimal Transport](https://arxiv.org/pdf/1803.00567.pdf) by Peyré and Cuturi (Chapters 2 and 3 in particular)
      * If you enjoyed Marco’s lectures above, or want a more thorough theoretical understanding of the Wasserstein distance, then this textbook is for you! However, please keep in mind that this textbook is somewhat mathematically involved, so if you don't have a mathematics background you may struggle with it.

  **Questions**:

  1. What happens to the KLD/JSD when the real data and the generator's data lie on low dimensional manifolds?
     <details><summary>Solution</summary>
      <p>
      The true distribution and model distribution tend to have different supports which causes the KLD and JSD to saturate.
      </p>
     </details>

  2. With this in mind, how does using the Wasserstein distance, rather than JSD, reduce the  sensitivity to careful scheduling of the generator and discriminator?
     <details><summary>Solution</summary>
      <p>
      The Wasserstein distance does not saturate or blow up for distributions with different supports. This means that we still get signals in these cases which in turn means that we don’t have to worry about training the discriminator (or critic) to optimality &#8212; in fact, we <i>want</i> to train it to optimality since it will give better signals.
      </p>
     </details>

  3. Let’s compare the 1-Wasserstein Distance (aka Earth Mover’s Distance - EMD) with the KLD for a few simple discrete distributions. We want to build up an intuition for the differences between these two metrics and why one might be better than another in certain scenarios. You might find it useful to use the Scipy implementations for [1-Wasserstein](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html) and [KLD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html).
      1. Let $$P(x)$$, $$Q(x)$$ and $$R(x)$$ be discrete distributions on $$Z$$ with:
          * $$P(0) = 0.5$$, $$P(1) = 0.5$$,
          * $$Q(0) = 0.75$$, $$Q(1) = 0.25$$, and
          * $$R(0) = 0.25$$ and $$R(1) = 0.75$$.
          <br> Calculate both the KLD and EMD for the following pairs of distributions. You should notice that while Wasserstein is a proper distance metric, KLD is not ($$D_{KL}(P||Q) \ne D_{KL}(Q||P)$$).
          1. $$P$$ and $$Q$$
          2. $$Q$$ and $$P$$
          3. $$P$$ and $$P$$
          4. $$P$$ and $$R$$
          5. $$Q$$ and $$R$$
      2. Let $$P(x)$$, $$Q(x)$$, $$R(x)$$, $$S(x)$$ be discrete distributions on $$Z$$ with:
          * $$P(0) = 0.5$$, $$P(1) = 0.5$$, $$P(2) = 0$$,
          * $$Q(0) = 0.33$$, $$Q(1) = 0.33$$, $$Q(2) = 0.33$$,
          * $$R(0) = 0.5$$, $$R(1) = 0.5$$, $$R(2) = 0$$, $$R(3) = 0$$, and
          * $$S(0) = 0$$, $$S(1) = 0$$, $$S(2) = 0.5$$, $$S(3) = 0.5$$.
        <br> Calculate the KLD and EMD between the following pairs of distributions. You should notice that the EMD is well behaved for distributions with disjoint support while the KLD is not. 
          1. $$P$$ and $$Q$$
          2. $$Q$$ and $$P$$
          3. $$R$$ and $$S$$
      3. Let $$P(x)$$, $$Q(x)$$, $$R(x)$$, and $$S(x)$$ be discrete distributions on $$Z$$ with:
          * $$P(0) = 0.25$$, $$P(1) = 0.75$$, $$P(2) = 0$$,
          * $$Q(0) = 0$$, $$Q(1) = 0.75$$, $$Q(2) = 0.25$$,
          * $$R(0) = 0$$, $$R(1) = 0.25$$, $$R(2) = 0.75$$, and
          * $$S(0) = 0$$, $$S(1) = 0$$, $$S(2) = 0.25$$, $$S(3) = 0.75$$.
          <br> Calculate the EMD between the following pairs of distributions. Here we just want to get more of a sense for the EMD.
          1. $$P$$ and $$Q$$
          2. $$P$$ and $$R$$
          3. $$Q$$ and $$R$$
          4. $$P$$ and $$S$$
          5. $$R$$ and $$S$$
      <details><summary>Solution</summary>
        <p>
        <a href="https://github.com/JamesAllingham/DFL-WGAN/blob/master/DFL_WGAN_week4_q3.ipynb">Here</a> is a Jupyter notebook with solutions.
        </p>
      </details>

  4. Based on the GAN implementation from week 3, implement a WGAN for FashionMNIST. 
      * Try various settings of hyper-parameters. Does this model seem more resilient to the choice of hyper-parameters?
      * Examine samples from various stages of the training. Rank them without looking at the corresponding loss and see if your ranking agrees with the loss.
      <details><summary>Solution</summary>
        <p>
        <a href="https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py">Here</a> is a WGAN implementation using Keras.
        </p>
      </details>

  **Notes**: Here is a [link](todo:add_link) to our notes for the lesson. We were fortunate enough to have Martin Arjovsky sit in on the session!

  <br />

# 5 WGAN-GP

  **Motivation**: Let’s read the WGAN-GP paper (Improved Training of Wasserstein GANs). As has been the trend over the last few weeks, we’ll see how this method solves a problem with the standard WGAN: weight clipping, as well as a potential problem in the standard GAN: overfitting.

  **Topics**:

  1. WGAN-GP
  3. Weight clipping vs gradient penalties
  2. Measuring GAN performance

  **Required Reading**:

  1. [WGAN-GP paper](https://arxiv.org/pdf/1704.00028.pdf)
      * This is our final required reading. The paper suggests improvements to the training of Wasserstein GANs with some great theoretical justifications and actual results.


  **Optional Reading**:

  1. [On the Regularization of Wasserstein GANs](https://arxiv.org/pdf/1709.08894.pdf)
      * This paper came out after the WGAN-GP paper but gives a thorough discussion of why the weight clipping in the original WGAN was an issue (see Appendix B). In addition, they propose other solutions for how to get around doing so and provide other interesting discussions of GANs and WGANs.  
  2. [Wasserstein GAN & WGAN-GP blog post](https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490)
      * Another blog that summarises many of the key points we’ve covered and includes WGAN-GP.
  3. [GAN — How to measure GAN performance?](https://medium.com/@jonathan_hui/gan-how-to-measure-gan-performance-64b988c47732)
      * A blog that discusses a number of approaches to measuring the performance of GANs, including the Inception score, which is useful to know about when reading the WGAN-GP paper.


  **Questions**:

  1. Why does weight clipping lead to instability in the training of a WGAN & how does the gradient penalty avoid this problem?
     <details><summary>Solution</summary>
      <p>
      The instability comes from the fact that if we choose the weight clipping hyper-parameter poorly we end up with either exploding or vanishing gradients. This is because weight clipping encourages the optimizer to push the absolute all of the weights very close to the clipping value. Figure 1b in the paper shows this happening. To explain this phenomenon, consider a simple logistic regression model. Here if any of the features are highly predictive of a particular class it will be assigned as positive a weight as possible, similarly, if a feature is not predictive of a particular class, it will be assigned as negative a weight as possible. Now depending on our choice of the weight clipping value, we either get exploding or vanishing gradients. 
      <ul>
      <li> Vanishing gradients: this is similar to the issues if vanishing gradients in a vanilla RNN, or a very deep feed-forward NN without residual connections. If we choose the weight clipping value to be too small, during back-propagation, the error signal going to each layer will be multiplied by small values before being propagated to the previous layer. This results in exponential decay in the error signal as it propagates farther backward. </li>
      <li> Exploding gradients: similarly, if we choose a weight clipping value that is too large, the error signals will get repeatedly multiplied by large numbers as the propagate backward &#8212; resulting in exponential growth. </li>
      </ul>
      </p>
      <p>
      This phenomena also related to the reason we use weight initialization schemes such as Xavier and He and also why batch normalization is important &#8212; both of these methods help to ensure that information is propagated through the network without decaying or exploding.
      </p>
     </details>

  2. Explain how WGAN-GP addresses issues of overfitting in GANs.
     <details><summary>Solution</summary>
      <p>
      Both WGAN-GP, and indeed the original weight-clipped WGAN, have the property that the discriminator/critic loss corresponds to the sample quality from the discriminator, which lets us use the loss to detect overfitting (we can compare the negative discriminator/critic loss for a validation set to that of the training set of real images &#8212; when the two diverge we have overfitted). The correspondence between the loss and the sample quality can be explained by a number of factors.
      <ul>
      <li> With a WGAN we can train our discriminator to optimality. This means that if the critic is struggling to tell the difference between real and generated images we can conclude that the real and generated images are similar. In other words, the loss is meaningful.</li>
      <li> In addition, in a standard GAN where we cannot train the discriminator to optimality, our loss no longer approximates the JSD. We do not know what function our loss is actually approximating and as a result we cannot say (and in practise we do not see) that the loss is a meaningful measure of sample quality. </li>
      <li> Finally, there are arguments to be made that even if the loss for a standard GAN was approximating the JSD, the Wasserstein distance is a better distance measure for images distributions than the JSD. </li>
      </ul>
      </p>
     </details>

  3. Based on the WGAN implementation from week 4, implement an improved WGAN for MNIST.
      * Compare the results, ease of hyper-parameter tuning, and correlation between loss and your subjective ranking of samples, with the previous two models.
      * _The Keras implementation of WGAN-GP can be tricky. If you are familiar with another framework like TensorFlow or Pytorch it might be easier to use that instead. If not, don't be too hesitant to check the solution if you get stuck._
     <details><summary>Solution</summary>
      <p>
      <a href="https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py">Here</a> is a WGAN-GP implementation using Keras.
      </p>
     </details>

  **Notes**: Here is a [link](todo:add_link) to our notes for the lesson. We were fortunate enough to have Ishaan Gulrajani sit in on the session!
