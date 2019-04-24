---
layout: post
title:  "DeepStack"
date:   2018-07-10 12:00:00 -0400
categories: games
author: cinjon
blurb: "In this curriculum, you will explore Game Theory and Counterfactual
        Regret Minimization in order to understand techniques for solving two 
        person zero-sum games of incomplete information."
hidden: false
feedback: true
---

Thank you to Michael Bowling, Michael Johanson, and Marc Lanctot for contributions to this guide.

Additionally, this would not have been possible without the generous support of
Prof. Joan Bruna and his class at NYU, [The Mathematics of Deep Learning](https://github.com/joanbruna/MathsDL-spring18).
Special thanks to him, as well as Martin Arjovsky, my colleague in leading this
recitation, and my fellow students Ojas Deshpande, Anant Gupta, Xintian Han,
Sanyam Kapoor, Chen Li, Yixiang Luo, Chirag Maheshwari, Zsolt Pajor-Gyulai,
Roberta Raileanu, Ryan Saxe, and Liang Zhuo.

<div class="deps-graph">
  <iframe class="deps" src="/assets/deepstack-deps.svg" width="200"></iframe>
  <div>Concepts used in DeepStack. Click to navigate.</div>
</div>

# Why

Along with Libratus, DeepStack is one of two approaches to solving No-Limit 
Texas Hold-em that debuted coincidentally. This game was notoriously difficult
to solve as it has just as large a branching factor
as Go, but additionally is a game of imperfect information.

The main idea behind both DeepStack and Libratus is to use Counterfactual Regret 
Minimization (CFR) to find a mixed strategy that approximates a Nash Equilibrium 
strategy. CFR's convergence properties guarantee that we will yield such a strategy
and the closer we are to it, the better our outcome will be. They differ in
their implementation. In particular, DeepStack uses deep neural networks
to approximate the counterfactual value of each hand at specific points in the
game. While still being mathematically tight, this lets it cut short 
the necessary computation to reach convergence.

In this curriculum, you will explore the study of games with a tour through 
game theory and counterfactual regret minimization while building up the 
requisite understanding to tackle DeepStack. Along the way, you will learn
all of the necessary topics, including what is the 
[branching factor](https://en.wikipedia.org/wiki/Branching_factor), all about
[Nash Equilibria](https://en.wikipedia.org/wiki/Nash_equilibrium), and 
[CFR](https://www.quora.com/What-is-an-intuitive-explanation-of-counterfactual-regret-minimization).

<br>
# Common Resources:
1. MAS: [Multi Agent Systems](http://www.masfoundations.org/mas.pdf).
2. LT: [Marc Lanctot’s Thesis](http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf).
3. ICRM: [Introduction to Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf).
4. PLG: [Prediction, Learning, and Games](http://www.ii.uni.wroc.pl/~lukstafi/pmwiki/uploads/AGT/Prediction_Learning_and_Games.pdf).

<br>
# 1 Normal Form Games & Poker
  **Motivation**: Most of Game Theory, as well as the particular techniques used in
  DeepStack and Libratus, is built on the framework of Normal Form 
  Games. These are game descriptions and are familiarly represented as a matrix,
  a famous example being the Prisoner's Dilemma. In this section, we cover 
  the basics of Normal Form Games. In addition, we go over the rules of Poker and 
  why it had proved so difficult to solve.
  
  **Required Reading**:
  1. MAS: Sections 3.1 & 3.2.
  2. LT: Pages 5-7.
  3. [The Game of Poker](https://arxiv.org/pdf/1701.01724.pdf): Supplementary #1 on pages 16-17.
  
  **Optional Reading**:
  1. [The State of Solving Large Incomplete-Information Games, and Application to Poker](https://www.cs.cmu.edu/~sandholm/solving%20games.aimag11.pdf) (2010)
  
  **Questions**:
  1. LT: Prove that in a zero-sum game, the Nash Equilibrium strategies are interchangeable.
     <details><summary>Hint</summary>
     <p>Use the definition of a Nash Equilibrium along with the fact that 
     \(\mu_{i}(\sigma_{i}, \sigma_{-i}) + \mu_{-i}(\sigma_{i}, \sigma_{-i}) = c\).
     </p>
     </details>
  2. LT: Prove that in a zero-sum game, the expected payoff to each player is the same for every equilibrium.
     <details><summary>Solution</summary>
     <p>We will solve both this problem and the one above here. We have that if
     \(\mu_{i}(\sigma) = \mu(\sigma_{i}, \sigma_{-i})\) and
     \(\mu_{i}(\sigma') = \mu(\sigma_{i}', \sigma_{-i}')\) are both 
     Nash Equilibria, then:</p>
     <p>\(
     \begin{align}
     \mu_{i}(\sigma_{i}, \sigma_{-i}) &\geq \mu_{i}(\sigma_{i}', \sigma_{-i}) \\
     &= c - \mu_{-i}(\sigma_{i}', \sigma_{-i}) \\
     &\geq c - \mu_{-i}(\sigma_{i}', \sigma_{-i}') \\
     &= \mu_{i}(\sigma_{i}', \sigma_{-i}')
     \end{align}
     \)
     </p>
     <p>In a similar fashion, we can show that 
     \(\mu(\sigma_{i}', \sigma_{-i}') \geq \mu(\sigma_{i}, \sigma_{-i})\).
     </p>
     Consequently, \(\mu(\sigma_{i}', \sigma_{-i}') = \mu(\sigma_{i}, \sigma_{-i})\),
     which also implies that the strategies are interchangeable, i.e.
     \(\mu(\sigma_{i}', \sigma_{-i}') = \mu(\sigma_{i}', \sigma_{-i})\).
     </details>
  3. MAS: Prove Lemma 3.1.6. <br />
     $$\textit{Lemma}$$: If a preference relation $$\succeq$$ satisfies the axioms
     completeness, transitivity, decomposability, and monotonicity, and if $$o_1 \succ o_2$$
     and $$o_2 \succ o_1$$, then there exists probability $$p$$ s.t. $$\forall p' < p$$,
     $$o_2 \succ [p': o_1; (1 - p'): o_3]$$ and for all $$p'' > p$$,
     $$[p'': o_1; (1 - p''): o_3] \succ o_2.$$
  4. MAS: Theorem 3.1.8 ensures that rational agents need only maximize the expectation
  of single-dimensional utility functions. Prove this result as a good test of your
  understanding. <br />
  $$\textit{Theorem}$$: If a preference relation $$\succeq$$ satisfies the axioms completeness,
  transitivity, substitutability, decomposability, monotonicity, and continuity, then
  there exists a function $$u: \mathbb{L} \mapsto [0, 1]$$ with the properties that:
     1. $$u(o_1) \geq u(o_2)$$ iff $$o_1 \succeq o_2$$.
     2. $$u([p_1 : o_1, ..., p_k: o_k]) = \sum_{i=1}^k p_{i}u(o_i)$$.

<br>
# 2 Optimality & Equilibrium 
  **Motivation**: How do you reason about games? The best strategies in multi-agent 
  scenarios depend on the choices of others. Game theory deals with this problem 
  by identifying subsets of outcomes called solution concepts. In this section, we
  discuss the fundamental solution concepts: Nash Equilibrium, Pareto Optimality, 
  and Correlated Equilibrium. For each solution concept, we cover what it implies 
  for a given game and how difficult it is to discover a representative strategy.
  
  **Required Reading**:
  1. MAS: Sections 3.3, 3.4.5, 3.4.7, 4.1, 4.2.4, 4.3, & 4.6.
  2. LT: Section 2.1.1.
  
  **Optional Reading**:
  1. MAS: Section 3.4.
  
  **Questions**:
  1. Why must every game have a Pareto optimal strategy?
     <details><summary>Solution</summary>
     <p>Say that a game does not have a Pareto optimal outcome. Then, for every 
     outcome \(O\), there was another \(O'\) that Pareto-dominated \(O\).
     Say \(O_2 > O_1\). Because \(O_2\) is not Pareto optimal, there is some 
     \(O_k > O_2\). There cannot be a max in this chain (because that max would
     be Pareto optimal) and thus there must be some cycle. Consequently, there
     exists for some agent a strategy \(O_j\) s.t. \(O_j > O_j\), which is a 
     contradiction.
     </p>
     </details>
  2. Why must there always exist at least one Pareto optimal strategy in which 
  all players adopt pure strategies?
  3. Why in common-payoff games do all Pareto optimal strategies have the same payoff?
     <details><summary>Solution</summary>
     <p>Say two strategies \(S\) and \(S'\) are Pareto optimal. Then neither
     dominates the other, so either \(\forall i \mu_{i}(S) = \mu_{i}(S')\)
     or there are two players \(i, j\) for which \(mu_{i}(S) < \mu_{i}(S')\)
     and \(mu_{j}(S) > \mu_{j}(S')\). In the former case, we see that the
     two strategies have the same payoff as desired. In the latter case, we have
     a contradiction because \(\mu_{j}(S') = \mu_{i}(S') > \mu_{i}(S) 
     = \mu_{j}(S) > \mu_{j}(S')\). Thus, all of the Pareto optimal strategies 
     must have the same payoff.
     </p>
     </details>
  4. MAS: Why does definition 3.3.12 imply that the vertices of a simplex must
  all receive different labels?
     <details><summary>Solution</summary>
     <p>This follows from the definitions of \(\mathbb{L}(v)\) and \(\chi(v)\).
     At the vertices of the simplex, \(\chi\) will only have singular values in
     its range defined by the vertice itself. Consequently, \(\mathbb{L}\) must
     as well.
     </p>
     </details>
  5. MAS: Why in definition 3.4.12 does it not matter that the mapping is to 
  pure strategies rather than to mixed strategies?
  6. Take your favorite normal-form game, find a Nash Equilibrium, and then find 
  a corresponding Correlated Equilibrium.

<br>
# 3 Extensive Form Games
  **Motivation**: What happens when players don't act simultaneously? 
  Extensive Form Games are an answer to this question. While this representation 
  of a game always has a comparable Normal Form, it's much more natural to reason 
  about sequential games in this format. Examples include familiar ones like Go, 
  but also more exotic games like Magic: The Gathering and Civilization. This 
  section is imperative as Poker is best described as an Extensive Form Game.
  
  **Required Reading**:
  1. MAS: Sections 5.1.1 - 5.1.3.
  2. MAS: Sections 5.2.1 - 5.2.3.
  3. [Accelerating Best Response Calculation in Large Extensive Games](http://martin.zinkevich.org/publications/ijcai2011_rgbr.pdf): 
  This is important for understanding how to evaluate Poker algorithms.
  
  **Optional Reading**: 
  1. LT: Section 2.1.2.

  **Questions**:
  1. What is the intuition for why not all normal form games can be transformed 
  into perfect-form extensive games?
     <details><summary>Solution</summary>
     <p>The problem is one of modeling simultaneity. Perfect information 
     extensive form games have trouble modeling concurrent moves because they
     have an explicit temporal structure of moves.
     </p>
     </details>
  2. Why does that change when the transformation is to imperfect extensive games?  
  3. How are the set of behavioral strategies different from the set of mixed strategies?
     <details><summary>Solution</summary>
     <p>The set of mixed strategies are each distributions over pure strategies. 
     The set of behavioral strategies are each vectors of distributions over the
     actions and assign that distribution independently at each Information Set.
     </p>
     </details>
  4. Succinctly describe the technique demonstrated in the Accelerating Best Response paper.

<br>
# 4 Counterfactual Regret Minimization #1
  **Motivation**: Counterfactual Regret Minimization (CFR) is only a decade old 
  but has already achieved huge success as the foundation underlying DeepStack 
  and Libratus. In the first of two weeks dedicated to CFR, we learn how the
  algorithm works practically and get our hands dirty coding up our implementation.

  The optional readings are papers introducing CFR-D and CFR+, further
  iterations upon CFR. These are both used in DeepStack.
  
  **Required Reading**:
  1. ICRM: Sections 2.1-2.4.
  2. ICRM: Sections 3.1-3.4.
  2. LT: Section 2.2.
  3. [Regret Minimization in Games with Incomplete Information](http://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf).
  
  **Optional Reading**: These two papers are CFR extensions used in DeepStack.
  1. [Solving Imperfect Information Games Using Decomposition](https://pdfs.semanticscholar.org/8216/0cbdcbeb13d53db85da928d8c42a789fdd69.pdf): CFR-D.
  2. [Solving Large Imperfect Information Games Using CFR+](https://arxiv.org/pdf/1407.5042.pdf): CFR+.
  
  **Questions**:
  1. What is the difference between external regret, internal regret, swap regret,
  and counterfactual regret?
     <details><summary>Hint</summary>
     <p>The definitions of the three are the following:</p>
     <ul>
     <li><b>External Regret</b>: How much the algorithm regrets not taking the best
     single decision in hindsight. We compare to a policy that performs a single
     action in all timesteps.</li>
     <li><b>Internal Regret</b>: How much the algorithm regrets making one choice
     over another in all instances. An example is whenever you bought Amazon stock,
     you instead bought Microsoft stock.</li>
     <li><b>Swap Regret</b>: Similar to Internal Regret but instead of one categorical
     action being replaced wholesale with another categorical action, now we allow
     for any number of categorical swaps.</li>
     <li><b>Counterfactual Regret</b>: Assuming that your actions take you to a 
     node, this is the expectation of that node over your opponents' strategies.
     The counterfactual component is that we assume you get to that node with a
     probability of one.</li>
     </ul>
     </details>
  2. Why is Swap Regret important?
     <details><summary>Hint</summary>
     <p>Swap Regret is connected to Correlated Equilibrium. Can you see why?</p>
     </details>
  3. Implement CFR (or CFR+ / CFR-D) in your favorite programming language to play 
  Leduc Poker or Liar’s Dice. 
  4. How do you know if you've implemented CFR correctly?
     <details><summary>Solution</summary>
     <p>One way is to test it by implementing Local Best Response. It should 
     perform admirably against that algorithm, which is meant to best it.</p>
     </details>
    
<br>
# 5 Counterfactual Regret Minimization #2
  **Motivation**: In the last section, we saw the practical side of CFR and how effective it 
  can be. In this section, we’ll understand the theory underlying it. This will culminate 
  with Blackwell’s Approachability Theorem, a generalization of repeated two-player 
  zero-sum games. This is a challenging session but the payoff will be a much 
  keener understanding of CFR's strengths.
  
  **Required**:
  1. PLG: Sections 7.3 - 7.7, 7.9.
  
  **Optional**:
  1. [A Simple Adaptive Procedure Leading to Correlated Equilibrium](http://wwwf.imperial.ac.uk/~dturaev/Hart0.pdf).
  2. [Prof. Johari's 2007 Class - 11](http://web.stanford.edu/~rjohari/teaching/notes/336_lecture11_2007.pdf).
  3. [Prof. Johari's 2007 Class - 13](http://web.stanford.edu/~rjohari/teaching/notes/336_lecture13_2007.pdf).
  4. [Prof. Johari's 2007 Class - 14](http://web.stanford.edu/~rjohari/teaching/notes/336_lecture14_2007.pdf).
  5. [Prof. Johari's 2007 Class - 15](http://web.stanford.edu/~rjohari/teaching/notes/336_lecture15_2007.pdf).
  
  **Questions**:
  1. PLG: Prove Lemma 7.1. <br />
     $$\textit{Lemma}$$: A probability distribution $$P$$ over the set of all $$K$$-tuples
     $$i = (i_{1}, ..., i_{K})$$ of actions is a correlated equilibrium iff, for every
     player $$k \in {1, ..., K}$$ and actions $$j, j' \in {1, ..., N_{k}}$$, we have

     $$
     \sum_{i: i_{k} = j} P(i)\big(\mathcal{l}(i) - \mathcal{l}(i^{-}, j')\big) \leq 0
     $$

     where $$(i^{-}, j') = (i_{1}, ..., i_{k-1}, j', i_{k+1}, ..., i_{K})$$.
  2. It's brushed over in the proof of Theorem 7.5 in PLG, but prove that if set 
  $$S$$ is approachable, then every halfspace $$H$$ containing $$S$$ is approachable.
     <details><summary>Solution</summary>
     <p>Because \(S \in H\) is approachable, we can always find a strategy for player one s.t.
     the necessary approachability clauses hold (see Johari's Lecture 13). Namely, choose
     the strategy in \(S\) that asserts \(S\) as being approachable.</p>
     </details>

<br>
# 6 DeepStack
  **Motivation**: Let’s read the paper! A summary of what's going on to help with your
  understanding:

  DeepStack runs counterfactual regret minimization at every decision. However, it uses
  two separate neural networks, one for after the flop and one for after the turn, to
  estimate the counterfactual values without having to continue running CFR after those
  moments. This approach is trained beforehand and helps greatly with cutting short the
  search space at inference time. Each of the networks take as input the size of the pot
  and the current Bayesian ranges for each player across all hands. They output the
  counterfactual values for each hand for each player.

  In addition to DeepStack, we also include Libratus as required reading. This paper
  highlights Game Theory and CFR as the really important concepts in this curriculum; 
  deep learning is not necessary to build a champion Poker bot.
  
  **Required Reading**:
  1. [DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker](https://static1.squarespace.com/static/58a75073e6f2e1c1d5b36630/t/58b7a3dce3df28761dd25e54/1488430045412/DeepStack.pdf).
  2. [DeepStack Supplementary Materials](https://static1.squarespace.com/static/58a75073e6f2e1c1d5b36630/t/58bed28de3df287015e43277/1488900766618/DeepStackSupplement.pdf).
  3. [Libratus](https://arxiv.org/pdf/1705.02955.pdf).
  4. [Michael Bowling on DeepStack](https://vimeo.com/212288252).
  
  **Optional Reading**:
  1. [DeepStack Implementation for Leduc Hold’em](https://github.com/lifrordi/DeepStack-Leduc).
  2. [Noam Brown on Libratus](https://www.youtube.com/watch?v=2dX0lwaQRX0).
  3. [Depth-Limited Solving for Imperfect-Information Games](https://arxiv.org/abs/1805.08195): This paper is fascinating because it is achieves a poker-playing bot almost as good as Libratus but using a fraction of the necessary computation and disk space.
  
  **Questions**:
  1. What are the differences between the approaches taken in DeepStack and in Libratus?
     <details><summary>Solution</summary>
     <p>Here are some differences:</p>
     <ul>
     <li>A clear difference is that DeepStack uses a deep neural network to reduce the necessary search space, and Libratus does not.</li>
     <li>DeepStack does not use any action abstraction and instead melds those considerations into the pot size input. Libratus does use a dense action abstraction but adapts it each game and additionally constructs new sub-games on the fly for actions not in its abstraction.</li>
     <li>DeepStack uses card abstraction by first clustering the hands into 1000 buckets and then considering probabilities over that range. Libratus does not use any card abstraction preflop or on the flop, but does use it on later rounds such that the game's \(10^{61}\) decision points are reduced to \(10^{12}\).</li>
     <li>DeepStack does not have a way to learn from recent games without further neural network training. On the other hand, Libratus improves via a background process that adds novel opponent actions to its action abstraction.</li>
     </ul>
     </details>
  2. Can you succinctly explain "Continual Re-solving"?
  3. Can you succinctly explain AIVAT?
