---
title: NLP Overview
description: General reference for NLP topics. Some sections may only include key words.
layout: post
mermaid: true
date: 2024-05-23 21:56 -0500
tags:
- nlp
categories:
- Tech
- Natural Language Processing
---
## Part-of-Speech (POS) tagging
- Byte-pair encoding
- Morphological Parsing
  - Porter stemmer Algo (<https://tartarus.org/martin/PorterStemmer/>)
- Named Entity Recognition (NER)
- **precision**: fraction of retrieved documents that are relevant
- **recall**: fraction of relevant documents that are retrieved
- IO vs IOB (inside-outside-beginning) tagging

## Markov
- Future state depends on past state (t depends on t-1)
- Conditional Markov Decision Models (CMM)
- Maximum Entropy Markov Model (MEMM)
- Flavors:
  - t depends on t-1 and t+1
  - t depends on t-1, t-2 ...
- greedy vs beam search

## Parsing
- Constitency (<https://parser.kitaev.io/>)
  - <https://nlpprogress.com/english/constituency_parsing.html>
  - Starts right to left. Forms a tree.
- Dependency (<https://demos.explosion.ai/displacy>)
  - Which words depend on which other words.
  - Identify the "head" or "root" then go from there
  - **projectivity**: A dependency tree is projective if it can be drawn with no crossing edges.
- MaltParser (<https://www.maltparser.org/>)

## N-gram
- `Unigram` -- essentially, random words. Their tag only depends on the word.
- `Bi-gram` -- the markov model. Their tag depends on the previous word.
- `N-gram` -- you get the idea ... N=k for some value of k>1

## Naiive Bayes
- ...

## Neural Language Models
- ...

## Word Vectors
- ...

## LSTM
- ...

## Attention and Transformers
- ...

## Finetuning and Prompting
- ...

## Reinforcement Learning with Human Feedback (RLHF)
- ...

## Popular Libs
1. nltk (<https://www.nltk.org/>)
2. spaCy (<https://spacy.io/>)

## Popular Conferences
1. EMNLP (<https://2024.emnlp.org/>)
2. ACL (<https://www.aclweb.org/portal/>)
3. NAACL (<https://2024.naacl.org/>)

## Books and References
1. Jurafsky and Martin 's book (<https://web.stanford.edu/~jurafsky/slp3/>)
2. nltk book (<https://www.nltk.org/book/>)
3. Foundation of Statistical NLP by Manning (<https://icog-labs.com/wp-content/uploads/2014/07/Christopher_D._Manning_Hinrich_Sch%C3%BCtze_Foundations_Of_Statistical_Natural_Language_Processing.pdf>)

