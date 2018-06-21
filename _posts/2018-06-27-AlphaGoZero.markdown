---
layout: post
title:  "AlphaGo Zero"
date:   2018-06-27 12:00:00 -0400
categories: games
author: cinjon
blurb: "In this curriculum, you will learn about two person zero-sum perfect 
        information games and develop understanding to completely grok AlphaGo Zero."
hidden: true
feedback: true
---

(Thank you to Marc Lanctot, Tim Lillicrap, Joan Bruna, Martin Arjovsky, Xintian Han,
Liang Zhuo, Chirag Maheshwari, Ryan Saxe, Ojas Deshpande, Sanyam Kapoor, Anant Gupta,
Yixiang Luo, and Chen Li for contributing to this guide.)

<div class="deps-graph">
  <iframe class="deps" src="/assets/ag0-deps.svg" width="200"></iframe>
  <div>Concepts used in AlphaGo Zero. Click to navigate.</div>
</div>

# Why

AlphaGo Zero was a big splash when it debuted and for good reason. The grand effort 
was led by David Silver at DeepMind and was an extension of work that he started
during his PhD. The main idea is to solve the game of Go and the approach taken
is to use Monte Carlo Tree Search along with a deep neural network as the value
approximation to cut off the search space. 

In this curriculum, you will focus on the study of two person zero-sum perfect 
information games and develop understanding sufficient to grok AlphaGo.

<br />
# 1 Minimax & Alpha Beta Pruning
  **Motivation**: These original core ideas did so much for the study of games.
  They spurred the field forward starting in the 50s and still to this day have
  mindshare in how to build a computer engine that beats games, including in
  popular chess engines like Stockfish.

  **Topics**:
  1. Perfect Information Games.
  1. Minimax.
  2. Alpha-Beta Pruning.

  **Required Reading**: 
  1. [Cornell Recitation on Minimax & AB Pruning](https://www.cs.cornell.edu/courses/cs312/2002sp/lectures/rec21.htm).
  2. [Knuth](https://pdfs.semanticscholar.org/dce2/6118156e5bc287bca2465a62e75af39c7e85.pdf): 6 (Theorems 1&2, Corollaries 1&3).
    
  **Optional Reading**:
  1. [CMU's MFAI Lecture 1](https://www.cs.cmu.edu/~arielpro/mfai_papers/lecture1.pdf).
  2. [Knuth](https://pdfs.semanticscholar.org/dce2/6118156e5bc287bca2465a62e75af39c7e85.pdf): 1-3.    
  3. [Chess Programming on Minimax](https://chessprogramming.wikispaces.com/Minimax).
  4. [Chess Programming on AB Pruning](https://chessprogramming.wikispaces.com/Alpha-Beta).
    
  **Questions**:
  1. (Knuth) Show that AlphaBetaMin(p, alpha, beta) = -AlphaBetaMax(p, -beta, -alpha). (p. 300)
     <details><summary>Solution</summary>
     <p>
     </p>
     </details>
  2. (Knuth) For Theorem 1.(1), why are the successor positions of type 2? (p. 305)
     <details><summary>Solution</summary>
     <p>
     </p>
     </details>
  3. (Knuth) For Theorem 1.(2), why is it that p’s successor position is of type 3 if p is not terminal?
     <details><summary>Solution</summary>
     <p>
     </p>
     </details>
  4. (Knuth) For Theorem 1.(3), why is it that p’s successor positions are of type 2 if p is not terminal?
     <details><summary>Solution</summary>
     <p>
     </p>
     </details>
  5. (Knuth) Show that Theorem 2.(1, 2, 3) are correct.
     <details><summary>Solution</summary>
     <p>
     </p>
     </details>

<br />
# 2 Multi-Armed Bandits & Upper Confidence Bounds
  **Motivation**: Bandits and UCB are key components of how MCTS was originally formalized. The node selection during the search is achieved through the UCB approach, which is analogues to how its performed in a multi-armed bandit scenario.

  **Topics**:
  1. Basics of Reinforcement Learning.
  2. Multi-Armed Bandit algorithms and their bounds.

  **Required Reading**: 
  1. Sutton: Sections 2.1 - 2.6 (Find on newclasses.nyu.edu in the class materials)
  2. [Jeremy Kun: Optimizing in the Face of Uncertainty](https://jeremykun.com/2013/10/28/optimism-in-the-face-of-uncertainty-the-ucb1-algorithm/)
    
  **Optional Reading**:
  1. [Original UCB1 Paper](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)
  2. [UW Lecture Notes](https://courses.cs.washington.edu/courses/cse599s/14sp/scribes/lecture15/lecture15_draft.pdf)
    
  **Questions**:
  1. Sutton Exercises 2.1, 2.2, 2.4, 2.5
  2. Sutton: What are the pros and cons of the optimistic initial values method?
  3. Kun: In the proof for the expected cumulative regret of UCB1, why is delta(T) a trivial regret bound if the deltas are all the same?
     <details><summary>Solution</summary>
     <p>
     </p>
     </details>
  4. Kun: Do you understand the argument for why the regret bound is O(sqrt(KTlog(T)))?
     <details><summary>Solution</summary>
     <p>
     </p>
     </details>
  5. Reproduce the UCB1 algorithm in code with minimal supervision.

<br />
# 3 Policy & Value Functions
  **Note**: Sutton from here out refers to the [final version](http://incompleteideas.net/book/bookdraft2017nov5.pdf).
  
  **Motivation**: The Policy and Value Functions are at the core of Reinforcement Learning. The Policy function is the set of probabilities you give to each possible move. The Value function is your estimate of how good is the current state. In AlphaGoZero, a single network calculates both a value and a policy, then later updates its weights based off of the difference between those figures and the empirical results.

  **Topics**:
  1. Bellman Equation.
  2. Policy Gradient.
  3. On-Policy / Off-Policy.
  4. Policy Iteration.
  5. Value Iteration.
  
  **Required Reading**: 
  1. Value Function:
     1. Sutton 3.5, 3.6, 3.7
     2. Sutton: 9.1, 9.2, 9.3 (important!)
  2. Policy Function:
     1. Sutton: 4.1, 4.2, 4.3
     2. Sutton: 13.1, 13.2 (important!), 13.3, 13.4
    
  **Optional Reading**:
  1. Sergey Levine: [Berkeley Fall'17: Policy Gradients](https://www.youtube.com/watch?v=tWNpiNzWuO8&feature=youtu.be) →  This is really good.
  2. Sergey Levine: [Berkeley Fall'17: Value Functions](https://www.youtube.com/watch?v=k1vNh4rNYec&feature=youtu.be) → This is really good.
  3. [Karpathy does Pong](http://karpathy.github.io/2016/05/31/rl/).
  4. [David Silver on PG](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf).
  5. [David Silver on Value](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/FA.pdf).
  
  **Questions**:
  1. Why does policy gradient have such high variance?
  2. What is the difference between off-policy and on-policy?
     <details><summary>Solution</summary>
     <p>
     </p>
     </details>
  3. Sutton:
     1. 3.13: What is the Bellman equation for action values, that is, for qπ? ...
     2. 3.14: In the gridworld example … are the signs of these rewards important, or only the intervals between them? Prove ...
     3. 3.15: Now consider adding a constant c to all the rewards in an episodic task … would this have any effect, or would it leave the task unchanged as in the continuing task above? Why or why not? Give an example. 
     4. 3.20: Give the Bellman equation for q∗ for the recycling robot. 
     5. 4.3: What are the equations analogous to (4.3), (4.4), and (4.5) for the action-value function qπ and its successive approximation by a sequence of functions q0, q1, q2, . . . ? 
     6. 4.6 (important!): How would policy iteration be defined for action values? Give a complete algorithm for computing q∗, analogous to that on page 65 for computing v∗. Please pay special attention to this exercise, because the ideas involved will be used throughout the rest of the book.
     <details><summary>Solution</summary>
     <p>
     </p>
     </details>
     7. 13.2 (important!): Prove (13.7) using the definitions and elementary calculus.
     <details><summary>Solution</summary>
     <p>
     </p>
     </details>


<br />
# 4 MCTS & UCT
  **Motivation**: Monte Carlo Tree Search (MCTS) forms the backbone of AlphaGoZero. It’s what lets it reliably explore and then hone in on the best policy. UCT (UCB for Trees) builds on top of what we’ve been learning and, paired with MCTS, is integral to the training process.

  **Topics**:
  1. Conceptual understanding of Monte Carlo Tree Search.
  2. Optimality of UCT.

  **Required Reading**:
  1. [Sutton](http://incompleteideas.net/book/bookdraft2017nov5.pdf): Section 8.11
  2. [Browne](https://gnunet.org/sites/default/files/Browne%20et%20al%20-%20A%20survey%20of%20MCTS%20methods.pdf): Sections 2.2, 2.4, 3.1-3.5, 8.2-8.4.
  3. [Silver Thesis](http://papersdb.cs.ualberta.ca/~papersdb/uploaded_files/1029/paper_thesis.pdf): Sections 1.4.2 and 3.
  
  **Optional Reading**:
  1. [Jess Hamrick on Browne](http://jhamrick.github.io/quals/planning%20and%20decision%20making/2015/12/16/Browne2012.html).
  2. [Original MCTS Paper](https://hal.archives-ouvertes.fr/file/index/docid/116992/filename/CG2006.pdf).
  3. [Original UCT Paper](http://ggp.stanford.edu/readings/uct.pdf).
  4. Browne: 
     1. 4.8: MCTS applied to Stochastic or Imperfect Information Games.
     2. 7.2, 7.3, 7.5, 7.7: Applications of MCTS.
    
  **Questions**:
  1. Can you detail each of the four parts of the MCTS algorithm?
     <details><summary>Solution</summary>
     <p>

     1. Selection: Select child node from the current node based on the tree policy.
     2. Expansion: Expand the child node based on the exploration / exploitation trade-off.
     3. Simulation: Simulate from the child node until termination or upon reaching a suitably small future reward (like from reward decay).
     4. Backup: Backup the reward along the path taken according to the tree policy.

     </p>
     </details>
  2. What characteristics make MCTS a good choice?
  3. What are examples of domain knowledge default policies in Go?
  4. Why is UCT optimal? Can you prove that the failure probability at the root converges to zero at a polynomial rate in the number of games simulated?
     <details><summary>Solution</summary>
     <p>
     </p>
     </details>
  
<br />
# 5 MCTS & RL
  **Motivation**: Up to this point, we’ve learned a lot about how games can be
  solved and how RL works on a foundational level. Before we jump into the paper,
  one last foray contrasting and unifying the games vs learning perspective is
  worthwhile for understanding the domain more fully.

  **Topics**:
  1. Integrating MCTS and RL.
  
  **Required Reading**:
  1. [Vodopivec](http://www.jair.org/media/5507/live-5507-10333-jair.pdf): 
  2. Connection between MCTS and RL → 3.1-3.4
  3. Integrating MCTS and RL → 4.1-4.3
  4. [Why did TD-Gammon Work?](https://papers.nips.cc/paper/1292-why-did-td-gammon-work.pdf)
  
  **Optional Reading**:
  1. Vodopivec: Survey of research inspired by both fields → 5.
  
  **Questions**:
  1. What are key differences between MCTS and RL?
  2. UCT can be described in RL terms as the following "The original UCT searches identically as an offline on-policy every-visit MC control algorithm that uses UCB1 as the policy." What do each of these terms mean?
  3. What is a Representation Policy? Give an example not described in the text.
  4. What is a Control Policy? Give an example not described in the text.

<br />
# 6 The Paper
  **Motivation**: Let's read the paper!

  **Topics**:
  1. MCTS learning and computational capacity.

  **Required Reading**:
  1. [Mastering the Game of Go Without Human Knowledge](https://www.dropbox.com/s/yva172qos2u15hf/2017-silver.pdf?dl=0)
  2. [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/pdf/1705.08439.pdf)
  
  **Optional Reading**:
  1. [Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning](http://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf)
  2. [Silver Thesis](http://papersdb.cs.ualberta.ca/~papersdb/uploaded_files/1029/paper_thesis.pdf): Section 4.6
  3. [Mastering the game of Go with deep neural networks and tree search](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
  4. [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
  
  **Questions**:
  1. What were the differences between the two papers "Mastering the Game of Go Without Human Knowledge" and "Thinking Fast and Slow with Deep Learning and Tree Search"?
  2. What was common to both of "Mastering the Game of Go Without Human Knowledge" and "Thinking Fast and Slow with Deep Learning and Tree Search"?  
