---
layout: post
title:  "AlphaGoZero"
date:   2018-06-26 12:00:00 -0400
categories: games
author: cinjon
blurb: "In this curriculum, you will learn about two-person zero-sum perfect 
        information games and develop understanding to completely grok AlphaGoZero."
hidden: false
feedback: true
---

Thank you to Marc Lanctot, Hugo Larochelle and Tim Lillicrap for contributions to this guide.

Additionally, this would not have been possible without the generous support of
Prof. Joan Bruna and his class at NYU, [The Mathematics of Deep Learning](https://github.com/joanbruna/MathsDL-spring18).
Special thanks to him, as well as Martin Arjovsky, my colleague in leading this
recitation, and my fellow students Ojas Deshpande, Anant Gupta, Xintian Han,
Sanyam Kapoor, Chen Li, Yixiang Luo, Chirag Maheshwari, Zsolt Pajor-Gyulai,
Roberta Raileanu, Ryan Saxe, and Liang Zhuo.

<div class="deps-graph">
  <iframe class="deps" src="/assets/ag0-deps.svg" width="200"></iframe>
  <div>Concepts used in AlphaGoZero. Click to navigate.</div>
</div>

# Why

AlphaGoZero was a big splash when it debuted and for good reason. The grand effort
was led by David Silver at DeepMind and was an extension of work that he started
during his PhD. The main idea is to solve the game of Go and the approach taken
is to use an algorithm called Monte Carlo Tree Search (MCTS). This algorithm acts as an expert guide to teach
a deep neural network how to approximate the value of each state. The convergence
properties of MCTS provides the neural network with a founded way to reduce the
search space.

In this curriculum, you will focus on the study of two-person zero-sum perfect
information games and develop understanding so that you can completely grok
AlphaGoZero.

<br />
# Common Resources:
1. Knuth: [An Analysis of Alpha-Beta Pruning](https://pdfs.semanticscholar.org/dce2/6118156e5bc287bca2465a62e75af39c7e85.pdf)
2. SB: [Reinforcement Learning: An Introduction, Sutton & Barto](http://incompleteideas.net/book/bookdraft2017nov5.pdf).
3. Kun: [Jeremy Kun: Optimizing in the Face of Uncertainty](https://jeremykun.com/2013/10/28/optimism-in-the-face-of-uncertainty-the-ucb1-algorithm/).
4. Vodopivec: [On Monte Carlo Tree Search and Reinforcement Learning](https://pdfs.semanticscholar.org/3d78/317f8aaccaeb7851507f5256fdbc5d7a6b91.pdf).

<br />
# 1 Minimax & Alpha Beta Pruning
  **Motivation**: Minimax and Alpha-Beta Pruning are original ideas that blossomed
  from the study of games starting in the 50s. To this day, they are components in 
  strong game-playing computer engines like Stockfish. In this class, we will go
  over these foundations, learn from Prof. Knuth's work analyzing their properties,
  and prove that these algorithms are theoretically sound solutions to two-player
  games.

  **Topics**:
  1. Perfect Information Games.
  1. Minimax.
  2. Alpha-Beta Pruning.

  **Required Reading**:
  1. [Cornell Recitation on Minimax & AB Pruning](https://www.cs.cornell.edu/courses/cs312/2002sp/lectures/rec21.htm).
  2. Knuth: Section 6 (Theorems 1&2, Corollaries 1&3).

  **Optional Reading**:
  1. [CMU's Mathematical Foundations of AI Lecture 1](https://www.cs.cmu.edu/~arielpro/mfai_papers/lecture1.pdf).
  2. Knuth: Sections 1-3.
  3. [Chess Programming on Minimax](https://chessprogramming.wikispaces.com/Minimax).
  4. [Chess Programming on AB Pruning](https://chessprogramming.wikispaces.com/Alpha-Beta).

  **Questions**:
  1. Knuth: Show that AlphaBetaMin $$= G2(p, \alpha, \beta) = -F2(p, -\beta, -\alpha) = -$$AlphaBetaMax. (p. 300)
     <details><summary>Solution</summary>
     <p>If \(d = 0\), then \(F2(p, \alpha, \beta) = f(p)\) and \(G2(p, -\alpha, -\beta) = g(p) = -f(p)\)
     as desired, where the last step follows from equation 2 on p. 295.
     </p>
     <p>Otherwise, \(d > 0\) and we proceed by induction on the height \(h\). The
     base case of \(h = 0\) is trivial because then the tree is a single root and
     consequently is the \(d = 0\) case. Assume it is true for height \(< h\),
     then for \(p\) of height \(h\), we have that \(m = a\) at the start of
     \(F2(p, \alpha, \beta)\) and \(m\prime = -\alpha\) at the start of \(G2(p, -\beta, -\alpha)\). So
     \(m = -m\prime\).
     </p>
     <p>In the i-th iteration of the loop, let's label the resulting value of \(m\)
     as \(m_{n}\). We have that \(t = G2(p_{i}, m , \beta) = -F2(p_i, -\beta, -m) = -t\)
     by the inductive assumption. Then,
     \(t > m \iff -t < -m \iff t\prime < m\prime \iff m_{n} = t = -m_{n}\prime\),
     which means that every time there is an update to the value of \(m\), it will
     be preserved across both functions. Further, because
     \(m \geq \beta \iff -m \leq -\beta \iff m\prime \leq -\beta\), we have that \(G2\) and
     \(F2\) will have the same stopping criterion. Together, these imply that
     \(G2(p, \alpha, \beta) = -F2(p, -\beta, -\alpha)\) after each iteration of the
     loop as desired.
     </p>
     </details>
  2. Knuth: For Theorem 1.1, why are the successor positions of type 2? (p. 305)
     <details><summary>Solution</summary>
     <p>By the definition of being type 1, \(p = a_{1} a_{2} \ldots a_{l}\), where
     each \(a_{k} = 1\). Its successor positions \(p_{l+1} = p (l+1)\) all have length
     \(l + 1\) and their first term \(> 1\) is at position \(l+1\), the last entry.
     Consequently, \((l+1) - (l+1) = 0\) is even and they are type 2.
     </p>
     </details>
  3. Knuth: For Theorem 1.2, why is it that p’s successor position is of type 3
     if p is not terminal?
     <details><summary>Solution</summary>
     <p>If \(p\) is type 2 and size \(l\), then for \(j\) s.t. \(a_j\) is the first entry where
     \(a_j > 1\), we have that \(l - j\) is even. When it's not terminal, then its
     successor position \(p_1 = a_{1} \ldots a_{j} \dots a_{l} 1\) has a length of
     size \(l + 1\), which implies that \(l + 1 - j\) is odd and so \(p_1\) is
     type 3.
     </p>
     </details>
  4. Knuth: For Theorem 1.3, why is it that p’s successor positions are of type 2
     if p is not terminal?
     <details><summary>Hint</summary>
     <p>This is similar to the above two.</p>
     </details>
  5. Knuth: Show that the three inductive steps of Theorem 2 are correct.

<br />
# 2 Multi-Armed Bandits & Upper Confidence Bounds
  **Motivation**: The multi-armed bandits problem is a framework for understanding 
  the exploitation vs exploration tradeoff. Upper Confidence Bounds, or UCB, is 
  an algorithmically tight approach to addressing that tradeoff under certain
  constraints. Together, they are important components of how Monte Carlo Tree 
  Search (MCTS), a key aspect of AlphaGoZero, was originally formalized. For 
  example, in MCTS there is a notion of node selection where UCB is used extensively. 
  In this section, we will cover bandits and UCB.
  
  **Topics**:
  1. Basics of reinforcement learning.
  2. Multi-armed bandit algorithms and their bounds.

  **Required Reading**:
  1. SB: Sections 2.1 - 2.7.
  2. Kun.

  **Optional Reading**:
  1. [Original UCB1 Paper](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)
  2. [UW Lecture Notes](https://courses.cs.washington.edu/courses/cse599s/14sp/scribes/lecture15/lecture15_draft.pdf)

  **Questions**:
  1. SB: Exercises 2.3, 2.4, 2.6.
  2. SB: What are the pros and cons of the optimistic initial values method? (Section 2.6)
  3. Kun: In the proof for the expected cumulative regret of UCB1, why is $$\delta *T$$
  a trivial regret bound if the deltas are all the same?
     <details><summary>Solution</summary>
     <p>\(
     \begin{align}
     \mathbb{E}[R_{A}(T)] &= \mu^{*}T - \mathbb{E}[G_{A}(T)] \\
     &= \mu^{*}T - \sum_{i} \mu_{i}\mathbb{E}[P_{i}(T)] \\
     &= \sum_{i} (\mu^{*} - \mu_{i})\mathbb{E}[P_{i}(T)] \\
     &= \sum_{i} \delta_{i} \mathbb{E}[P_{i}(T)] \\
     &\leq \delta \sum_{i} \mathbb{E}[P_{i}(T)] \\
     &= \delta * T
     \end{align}
     \)
     </p>
     <p>The third line follows from \(sum_{i} \mathbb{E}[P_{i}(T)] = T\) and the
     fifth line from the definition of \(\delta\).
     </p>
     </details>
  4. Kun: Do you understand the argument for why the regret bound is $$O(\sqrt{KT\log(T)})$$?
     <details><summary>Hint</summary>
     <p>
     What happens if you break the arms into those with regret \(< \sqrt{K(\log{T})/T}\)
     and those with regret \(\geq \sqrt{K(\log{T})/T}\)? Can we use this to bound
     the total regret?
     </p>
     </details>
  5. Reproduce the UCB1 algorithm in code with minimal supervision.

<br />
# 3 Policy & Value Functions  
  **Motivation**: Policy and value functions are at the core of reinforcement 
  learning. The policy function is the representative probabilities that our
  policy assigns to each action. When we sample from these, we would like for 
  better actions to have higher probability. The value function is our estimate 
  of the strength of the current state. In AlphaGoZero, a single network 
  calculates both a value and a policy, then later updates its weights according
  to how well the agent performs in the game.

  **Topics**:
  1. Bellman equation.
  2. Policy gradient.
  3. On-policy / off-policy.
  4. Policy iteration.
  5. Value iteration.
  
  **Required Reading**: 
  1. Value Function:
     1. SB: Sections 3.5, 3.6, 3.7.
     2. SB: Sections 9.1, 9.2, 9.3*.
  2. Policy Function:
     1. SB: Sections 4.1, 4.2, 4.3.
     2. SB: Sections 13.1, 13.2*, 13.3, 13.4.

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
     On-policy algorithms learn from the current policy's action decisions.
     Off-policy algorithms learn from another arbitrary policy's actions. An
     example of an on-policy algorithm is SARSA or REINFORCE. An example of an
     off-policy algorithm is Q-Learning.
     </p>
     </details>
  3. SB: Exercises 3.13, 3.14, 3.15, 3.20, 4.3.
  4. SB: Exercise 4.6 - How would policy iteration be defined for action values?
  Give a complete algorithm for computing $$q^{*}$$, analogous to that on page 65
  for computing $$v^{*}$$.
       <details><summary>Solution</summary>
    <p> The solution follows the proof (page 65) for \(v^{*}\), with the following modifications:
       <ol>
        <li>Consider a randomly initialized Q(s, a) and a random policy \( \pi(s) \). </li>
        <li><b> Policy Evaluation </b> : Update Q(s, a) \( \leftarrow \sum_{s'} P_{ss'}^{a} R_{ss'}^{a} + \gamma \sum_{s'} \sum_{a'} P_{ss'}^{a} Q^{\pi}(s', a') \pi(a' | s') \) <br /> 
                                         Note that \( P_{ss'}^{a} \leftarrow P(s' |s, a) , R_{ss'}^{a} \leftarrow R(s, a, s').\)</li>
        <li><b> Policy Improvement </b> :   Update \( \pi(s) = {argmax}_{a} Q^{\pi}(s, a) \). If \(unstable\), go to step 2. Here, \( unstable \), implies \( \pi_{before\_update}(s) \neq \pi_{after\_update}(s) \)</li>
        <li> \( q^{*} \leftarrow Q(s, a) \) </li>
       </ol> </p>
       </details>
  5. SB: Exercise 13.2 - Prove that the eligibility vector 
  $$\nabla_{\theta} \ln \pi (a | s, \theta) = x(s, a) - \sum_{b} \pi (b | s, \theta)x(s, b)$$ 
  using the definitions and elementary calculus. Here, $$\pi (a | s, \theta)$$ = softmax( $$\theta^{T}x(s, a)$$ ).
       <details><summary>Solution</summary>
       <p align="center">
        By definition, we have \( \pi( a| s, \theta) = \frac{e^{ \theta^{T}
        \mathbf{x}( s, a) }}{ \sum_b e^{ \theta^{T}\mathbf{x}(s, b)) }} \), where
        \( \mathbf{x}(s, a) \) is the state-action feature representation. Consequently:
        <br />
        \(
        \begin{align}
         \nabla_{\theta} \ln \pi (a | s, \theta) &= \nabla_\theta \Big( \theta^{T}\mathbf{x}(s, a) - \ln \sum_b e^{ \theta^{T}\mathbf{x}(s, b) } \Big) \\
          &= \mathbf{x}(s, a) - \sum_b \mathbf{x}(s, b) \frac{ e^{ \theta^{T}\mathbf{x}(s, b) } }{ \sum_b e^{ \theta^{T}\mathbf{x}(s, b) } } \\
          &= \mathbf{x}(s, a) - \sum_{b} \pi (b | s, \theta)\mathbf{x}(s, b) \\
        \end{align}
        \)
       </p>
       </details>


<br />
# 4 MCTS & UCT
  **Motivation**: Monte Carlo Tree Search (MCTS) forms the backbone of AlphaGoZero.
  It is what lets the algorithm reliably explore and then hone in on the best policy.
  UCT (UCB for Trees) combines MCTS and UCB so that we get reliable convergence
  guarantees. In this section, we will explore how MCTS works and how to make
  it excel for our purposes in solving Go, a game with an enormous branching factor.

  **Topics**:
  1. Conceptual understanding of Monte Carlo Tree Search.
  2. Optimality of UCT.

  **Required Reading**:
  1. SB: Section 8.11
  2. [Browne](https://gnunet.org/sites/default/files/Browne%20et%20al%20-%20A%20survey%20of%20MCTS%20methods.pdf): Sections 2.2, 2.4, 3.1-3.5, 8.2-8.4.
  3. [Silver Thesis](http://papersdb.cs.ualberta.ca/~papersdb/uploaded_files/1029/paper_thesis.pdf): Sections 1.4.2 and 3.

  **Optional Reading**:
  1. [Jess Hamrick on Browne](http://jhamrick.github.io/quals/planning%20and%20decision%20making/2015/12/16/Browne2012.html).
  2. [Original MCTS Paper](https://hal.archives-ouvertes.fr/file/index/docid/116992/filename/CG2006.pdf).
  3. [Original UCT Paper](http://ggp.stanford.edu/readings/uct.pdf).
  4. Browne:
     1. Section 4.8: MCTS applied to Stochastic or Imperfect Information Games.
     2. Sections 7.2, 7.3, 7.5, 7.7: Applications of MCTS.

  **Questions**:
  1. Can you detail each of the four parts of the MCTS algorithm?
     <details><summary>Solution</summary>
     <ol>
     <li><b>Selection</b>: Select child node from the current node based on the tree policy.</li>
     <li><b>Expansion</b>: Expand the child node based on the exploration / exploitation trade-off.</li>
     <li><b>Simulation</b>: Simulate from the child node until termination or upon reaching a suitably small future reward (like from reward decay).</li>
     <li><b>Backup</b>: Backup the reward along the path taken according to the tree policy.</li>
     </ol>
     </details>
  2. What happens to the information gained from the Tree Search after each run?
     <details><summary>Solution</summary>
     <p>We can reuse the accumulated statistics in subsequent runs. We could also
     ignore those statistics and build fresh each subsequent root. Both are used
     in actual implementations.
     </p>
     </details>
  3. What characteristics of a domain would make MCTS a good algorithmic choice?
     <details><summary>Solution</summary>
     <p>
     A few such characteristics are:
     </p>
     <ul>
     <li>MCTS is aheuristic, meaning that it does not require any domain-specific
     knowledge. Consequently, if it is difficult to produce game heuristics for
     your target domain (e.g. Go), then it can perform much better than alternatives
     like Minimax. And on the flip side, if you did have domain-specific knowledge,
     MCTS can incorporate it and will improve dramatically.
     </li>
     <li>
     If the target domain needs actions online, then MCTS is a good choice as all
     values are always up to date. Go does not have this property but digital games
     like in the <a href="http://ggp.stanford.edu/">General Game Playing</a> suite
     may.
     </li>
     If the target domain's game tree is of a nontrivial size, then MCTS may be
     a much better choice than other algorithms as it tends to build unbalanced
     trees that explore the more promising routes rather than consider all routes.
     </li>
     <li>
     If there is noise or delayed rewards in the target domain, then MCTS is a
     good choice because it is robust to these effects which can gravely impact
     other algorithms such as modern Deep Reinforcement Learning.
     </li>
     </ul>
     </details>
  4. What are examples of domain knowledge default policies in Go?
     <details><summary>Solution</summary>
     <ul>
     <li>Crazy Stone, an early program that won the 2006 9x9 Computer Go Olympiad,
     used an
     <a href="https://www.researchgate.net/figure/Examples-of-move-urgency_fig2_220174551">urgency</a>
     heuristic value for each of the moves on the board.
     </li>
     <li>
     MoGo, the algorithm that introduced UCT, bases its default policies on this
     sequence:
       <ol>
       <li>Respond to ataris by playing a saving move at random.</li>
       <li>If one of the eight intersections surrounding the last move matches a
       simple pattern for cutting or <i>hane</i>, randomly play one.</li>
       <li>If there are capturing moves, play one at random.</li>
       <li>Play a random move.</li>
       </ol>
     </li>
     <li>The second version of Crazy Stone used an algorithm learned from actual
     game play to learn a library of strong patterns. It incorporated this into
     its default policy.
     </li>
     </ul>
     </details>
  5. Why is UCT optimal? For a finite-horizon MDP with rewards scaled to lie in
  $$[0, 1]$$, can you prove that the failure probability at the root converges
  to zero at a polynomial rate in the number of games simulated?
     <details><summary>Hint</summary>
     <p>
     Try using induction on \(D\), the horizon of the MDP. At \(D=1\), to what
     result does this correspond?
     </p>
     </details>
     <details><summary>Hint 2</summary>
     <p>
     Assume that the result holds for a horizon up to depth  \(D - 1\) and
     consider a tree of depth \(D\). We can keep the cumulative rewards bounded
     in the interval by dividing by \(D\). Now can you show that the UCT payoff
     sequences at the root satisfy the drift conditions, repeated below?
     </p>
     <ul>
     <li>The payoffs are bounded - \(0 \leq X_{it} \leq 1\), where \(i\) is the
     arm number and \(t\) is the time step.</li>
     <li>The expected values of the averages, \(\overline{X_{it}} =
     \frac{1}{n} \sum_{t=1}^{n} X_{it}\), converge.</li>
     <li>Define \(\mu_{in} = \mathbb{E}[\overline{X_{in}}]\) and \(\mu_{i} = \lim_{n\to\inf} \mu_{in}\).
     Then, for \(c_{t, s} = 2C_{p}\sqrt{\frac{\ln{t}}{s}}\), where \(C_p\) is a
     suitable constant, both
     \(\mathbb{P}(\overline{X_{is}} \geq \mu_{i} + c_{t, s}) \leq t^{-4}\) and
     \(\mathbb{P}(\overline{X_{is}} \leq \mu_{i} - c_{t, s}) \leq t^{-4}\) hold.
     </li>
     </ul>
     </details>
     <details><summary>Solution</summary>
     <p>
     For a complete detail of the proof, see the original
     <a href="http://ggp.stanford.edu/readings/uct.pdf">UCT</a> paper.
     </p>
     </details>


<br />
# 5 MCTS & RL
  **Motivation**: Up to this point, we have learned a lot about how games can be
  solved and how reinforcement learning works on a foundational level. Before we 
  jump into the paper, one last foray contrasting and unifying the games vs 
  learning perspective is worthwhile for understanding the domain more fully. In 
  particular, we will focus on a paper from Vodopivec et al. After completing 
  this section, you should have an understanding of what research directions in
  this field have been thoroughly explored and which still have open directions.

  **Topics**:
  1. Integrating MCTS and reinforcement learning.
  
  **Required Reading**:
  1. Vodopivec:
    1. Section 3.1-3.4: Connection between MCTS and RL.
    2. Section 4.1-4.3: Integrating MCTS and RL.
  2. [Why did TD-Gammon Work?](https://papers.nips.cc/paper/1292-why-did-td-gammon-work.pdf)

  **Optional Reading**:
  1. Vodopivec: Section 5: Survey of research inspired by both fields.

  **Questions**:
  1. What are key differences between MCTS and RL?
  2. UCT can be described in RL terms as the following "The original UCT searches
  identically as an offline on-policy every-visit MC control algorithm that uses
  UCB1 as the policy." What do each of these terms mean?
     <details><summary>Solution</summary>
     <ul>
     <li>
     UCT is trained on-policy, which means it improves the policy used to make the
     action decisions, i.e. UCB1.
     </li>
     <li>
     The offline means that we can't learn until after the episode is completed.
     An alternative online algorithm would learn while the episode was running.
     </li>
     <li>
     Every-visit versus first-visit decides if we are going to update a state for
     every time it's accessed in an episode or just the first time. The original
     UCT algorithm did every-visit. Subsequent versions relaxed this.
     </li>
     <li>
     MC control means that we are using Monte Carlo as the policy, i.e. we use
     the average value of the state as the true value.
     </li>
     </ul>
     </details>
  3. What is a Representation Policy? Give an example not described in the text.
     <details><summary>Solution</summary>
     <p>A Representation Policy defines the model of the state space (e.g. in
     the form of a value function) and the boundary between memorized and
     non-memorized parts of the space.
     </p>
     </details>
  4. What is a Control Policy? Give an example not described in the text.
     <details><summary>Solution</summary>
     <p>A Control Policy dictates what actions will be performed and (consequently)
     which states will be visited. In MCTS, it includes the tree and default policies.
     </p>
     </details>

<br />
# 6 The Paper
  **Motivation**: Let's read the paper! We have a deep understanding of the background,
  so let's delve into the apex result. Note that we don't just focus on the final
  AlphaGoZero paper but also explore a related paper written coincidentally by
  a team at UCL using Hex as the game of choice. Their algorithm is very similar
  to the AlphaGoZero algorithm and considering both in context is important to 
  gauging what was really the most important aspects of this research.

  **Topics**:
  1. MCTS learning and computational capacity.

  **Required Reading**:
  1. [Mastering the Game of Go Without Human Knowledge](https://deepmind.com/documents/119/agz_unformatted_nature.pdf)
  2. [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/pdf/1705.08439.pdf)

  **Optional Reading**:
  1. [Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning](http://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf)
  2. [Silver Thesis](http://papersdb.cs.ualberta.ca/~papersdb/uploaded_files/1029/paper_thesis.pdf): Section 4.6
  3. [Mastering the game of Go with deep neural networks and tree search](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
  4. [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)

  **Questions**:
  1. What were the differences between the two papers "Mastering the Game of Go
  Without Human Knowledge" and "Thinking Fast and Slow with Deep Learning and Tree Search"?
     <details><summary>Solution</summary>
     <p>Some differences between the former (AG0) and the latter (ExIt) are:</p>
     <ul>
     <li>AG0 uses MC value estimates from the expert for the value network
     where ExIt uses estimates from the apprentice. This requires more computation
     by AG0 but produces better estimates.</li>
     <li>The losses were different. For the value network, AG0 uses an MSE loss
     with L2 regularization and ExIt uses a cross entropy loss with early stopping.
     For the policy part, AG0 used cross entropy while ExIt uses a weighted
     cross-entropy that takes into account how confident MCTS is in the action
     based on the state count.</li>
     <li>AG0 uses the value network to evaluate moves; ExIt uses RAVE and rollouts,
     plus warm starts from the MCTS.</li>
     <li>AG0 adds in Dirichlet noise to the prior probability at the root node.</li>
     <li>AG0 elevates a new network as champion only when it's markedly better than
     the prior champion; ExIt replaces the old network without verification of if
     it is better.</li>
     </ul>
     </details>
  2. What was common to both of "Mastering the Game of Go Without Human Knowledge"
  and "Thinking Fast and Slow with Deep Learning and Tree Search"?
     <details><summary>Solution</summary>
     <p>The most important commonality is that they both use MCTS as an expert
     guide to help a neural network learn through self-play.</p>
     </details>
  3. Will the system get stuck if the current neural network can’t beat the previous ones?
     <details><summary>Solution</summary>
     <p>No. The algorithm won’t accept a policy that is worse than the current best
     and MCTS’s convergence properties imply that it will eventually tend towards
     the equilibrium solution in a zero-sum two player game
     </p>
     </details>
  4. Why include both a policy and a value head in these algorithms? Why not just use policy?
     <details><summary>Solution</summary>
     <p>Value networks reduce the required search depth. This helps tremendously
     because a rollout approach without the value network is inaccurate and spends
     too much time on sub-optimal directions.
     </p>
     </details>
