---
layout: post
title:  "Deep Stack"
date:   2018-05-24 18:50:23 -0400
categories: games
author: cinjon
blurb: "In this curriculum, you will explore game theory and counterfactual
        regret minimization in order to understand techniques for solving two 
        person zero-sum games of incomplete information."
---

# Introduction

Along with Libratus, DeepStack is one of two approaches to solving No-Limit 
Texas Hold-em that debuted coincidentally. This game was notoriously difficult
to solve as it has just as large a branching factor as Go, but additionally is a
game of imperfect information.

In this curriculum, you will wander down another branch of the study of games
with a tour through game theory and counterfactual regret minimization while
building up the requisite understanding to tackle DeepStack.

<br>
# Resources:
1. MAS: [Multi Agent Systems](http://www.masfoundations.org/mas.pdf)
2. LT: [Marc Lanctot’s Thesis](http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf)
3. ICRM: [Introduction to Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf**)
4. PLG: [Prediction, Learning, and Games](http://www.ii.uni.wroc.pl/~lukstafi/pmwiki/uploads/AGT/Prediction_Learning_and_Games.pdf)

<br>
# Class 1: Normal-Form Games & Poker
  **Motivation**: Normal-Form games are the backbone for many of the techniques that later were used in DeepStack and Libratus. Understanding them will be a necessary foundation to understanding the innovations they presented.
  
  **Required Reading**:
  1. MAS sections 3.1 & 3.2
  2. LT pages 5-7
  3. [The Game of Poker](https://arxiv.org/pdf/1701.01724.pdf) (Supplementary #1 on pages 16-17)
  
  **Optional Reading**:
  1. [The State of Solving Large Incomplete-Information Games, and Application to Poker](https://www.cs.cmu.edu/~sandholm/solving%20games.aimag11.pdf) (2010)
  2. [Why Poker is Difficult](https://www.youtube.com/watch?v=2dX0lwaQRX0) (very good video by Noam Brown, the main author on Libratus. The first 18 minutes are most relevant for now.)
  
  **Questions**:
  1. Prove that in a zero-sum game, the nash equilibrium strategies are interchangeable. (LT)
  2. Prove that in a zero-sum game, the expected payoff to each player is the same for every equilibrium. (LT)
  3. Can you prove Lemma 3.1.6?
  4. Can you prove Theorem 3.1.8 (which is a really cool result)? 

<br>
# Class 2: Optimality and Equilibrium 
  **Motivation**: How do you reason about games? The best strategy in multi-agent scenario depends on the choices of others. Game theory deals with this problem by identifying subsets of outcomes called solution concepts, of which fundamental ones are the Nash Equilibrium, Pareto Optimality, and Correlated Equilibrium.
  
  **Required Reading**:
  1. MAS Sections 3.3, 3.4.5, 3.4.7, 4.1, 4.2.4, 4.3, 4.6
  2. LT Section 2.1.1
  
  **Optional Reading**:
  1. The rest of section 3.4 in MAS.
  
  **Questions**:
  1. Why must every game have a Pareto Optimal strategy?
  2. Why must there always exist at least one Pareto Optimal Strategy in which all players adopt pure strategies?
  3. Why in common-payoff games do all Pareto optimal strategies have the same payoff?
  4. Why does definition 3.3.12 imply that the vertices of a simplex must all receive different labels?
  5. Why in definition 3.4.12 does it not matter that the mapping is to pure strategies rather than a mixed strategy?
  6. Take your favorite normal-form game, find a Nash Equilibrium, and then find a corresponding Correlated Equilibrium.

<br>
# Class 3: Extensive-Form Games
  **Motivation**: What happens when players don't act simultaneously? Extensive-Form Games are an answer to this question. While this representation of a game always has a comparable Normal-Form, it's much more natural to reason about in this format.
  
  **Required Reading**:
  1. MAS 5.1.{1,2,3}
  2. MAS 5.2.{1,2,3}
  3. [Accelerating Best Response Calculation in Large Extensive Games](http://martin.zinkevich.org/publications/ijcai2011_rgbr.pdf) --> Important for understanding how to evaluate Poker algorithms.
  
  **Optional Reading**: 
  1. LT Section 2.1.2

  **Questions**:
  1. What is the intuition for why not all normal form games can be transformed into perfect-form extensive games?
  2. How are the set of behavioral strategies different from the set of mixed strategies?
  3. Succinctly describe the technique demonstrated in the Accelerating Best Response paper.

<br>
# Class 4: Counterfactual Regret Minimization #1
  **Motivation**: Counterfactual Regret Minimization (CFR) is only a decade old 
  but has already achieved huge success as the foundation underlying DeepStack 
  and Libratus. In the first of two weeks dedicated to CFR, we learn how it 
  works algorithmically.
  
  **Required Reading**:
  1. ICRM: 2.1-2.4, 3.1-3.4
  2. LT: 2.2
  3. Original Paper --> [Regret Minimization in Games with Incomplete Information](http://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf)
  
  **Optional Reading**: The two below are CFR extensions used in DeepStack.
  1. CFR-D --> [Solving Imperfect Information Games Using Decomposition](https://pdfs.semanticscholar.org/8216/0cbdcbeb13d53db85da928d8c42a789fdd69.pdf)
  2. CFR+ --> [Solving Large Imperfect Information Games Using CFR+](https://arxiv.org/pdf/1407.5042.pdf)
  
  **Questions**:
  1. What is the difference between internal regret, external regret, and counterfactual regret?
  2. Implement CFR (or CFR+ / CFR-D) in your favorite programming language to play Leduc Poker or Liar’s Dice. 
  3. How do you know if you’ve implemented CFR correctly?

<br>
# Class 5: Counterfactual Regret Minimization #2
  **Motivation**: We saw last week the practical side of CFR and how effective it 
  can be. This week we’ll be diving more into the theory underlying it. This 
  will culminate with Blackwell’s Approachability Theorem, a generalization of 
  repeated two-player zero-sum games.
  
  **Required**:
  1. PLG: Section 7.3-7.7, 7.9
  
  **Optional**:
  1. [A Simple Adaptive Procedure Leading to Correlated Equilibrium](http://wwwf.imperial.ac.uk/~dturaev/Hart0.pdf) --> Important originating paper.
  2. [Prof. Johari's 2007 Class - 11](http://web.stanford.edu/~rjohari/teaching/notes/336_lecture11_2007.pdf)
  3. [Prof. Johari's 2007 Class - 13](http://web.stanford.edu/~rjohari/teaching/notes/336_lecture13_2007.pdf)
  4. [Prof. Johari's 2007 Class - 14](http://web.stanford.edu/~rjohari/teaching/notes/336_lecture14_2007.pdf)
  5. [Prof. Johari's 2007 Class - 15](http://web.stanford.edu/~rjohari/teaching/notes/336_lecture15_2007.pdf)
  
  **Questions**:
  1. Prove Lemma 7.1.
  2. It’s brushed over in the proof of Theorem 7.5 (PLG), but prove that if set S is approachable, then every halfspace H containing S is approachable.

<br>
# Class 6: DeepStack
  **Motivation**: Let’s read the paper!
  
  **Required Reading**:
  1. [DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker](https://static1.squarespace.com/static/58a75073e6f2e1c1d5b36630/t/58b7a3dce3df28761dd25e54/1488430045412/DeepStack.pdf)
  2. [DeepStack Supplementary Materials](https://static1.squarespace.com/static/58a75073e6f2e1c1d5b36630/t/58bed28de3df287015e43277/1488900766618/DeepStackSupplement.pdf)
  3. [Michael Bowling on DeepStack](https://vimeo.com/212288252)
  
  **Optional Reading**:
  1. [DeepStack Implementation for Leduc Hold’em](https://github.com/lifrordi/DeepStack-Leduc)
  2. [Libratus](http://www.cs.cmu.edu/~sandholm/safeAndNested.aaa17WS.pdf)
  3. [Noam Brown on Libratus](https://www.youtube.com/watch?v=2dX0lwaQRX0)
  
  **Questions**:
  1. What are the differences between the approaches taken in DeepStack and in Libratus?
  2. Do you understand "Continual Re-solving"?
  3. Do you understand AIVAT?
