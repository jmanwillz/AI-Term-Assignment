# COMS7044A Reproducibility Assignment

## Introduction

In the field of scientific research and knowledge discovery, reproducibility stands as a cornerstone
principle, ensuring the reliability and integrity of findings. The ability to reproduce experimental
results and validate conclusions is fundamental for advancing scientific understanding
and fostering trust within the scientific community and beyond. Reproducibility not only serves
as a measure of the robustness of scientific claims, but also plays an important role in promoting
transparency and accountability in research practices.

In recent years, concerns have been raised about the reproducibility crisis across various
disciplines. Studies have shown that a significant portion of published research findings cannot
be reproduced, casting doubt on the reliability of scientific knowledge. This crisis not only
undermines scientific progress but also erodes public trust in the scientific enterprise. While
this is true primarily in the “soft sciences” and humanities, this phenomenon has also shown to
be prevalent in artificial intelligence research.

## Aim

In this assignment, you will be tasked with reproducing an existing paper. You may work either
individually or in groups of up to three, although working alone will create significantly
more work for yourself, so I would definitely recommend partnering up. The paper you will be
required to replicate is the following:

Ofir Maron, and Benjamin Rosman. "Utilising uncertainty for efficient learning of
likely-admissible heuristics." Proceedings of the International Conference on Automated Planning
and Scheduling. Vol. 30. 2020.

The PDF and supplementary material for the paper can be found [here](https://www.raillab.org/publication/marom-2020-utilising/ "Utilising Uncertainty for Efficient Learning of Likely-Admissible Heuristics").

## Task Description

Your aim is to replicate the experiments described in the paper. The goal is to assess if the
experiments are reproducible, and to determine if the conclusions of the paper are supported by
your findings. Your results can be either positive (i.e. confirm reproducibility), or negative (i.e.
explain what you were unable to reproduce, and potentially explain why). Essentially, think of
your role as an inspector verifying the validity of the experimental results and conclusions of
the paper.

For the purpose of this assignment, all code should be written in Python. However, you are
free to use any libraries you require, and should not program things from scratch when there
is no need to. Additionally, the authors’ code is available online [here](https://github.com/OfirMarom/LearnHeuristicWithUncertaintly). Unfortunately, it is written in C#, but you can
and should make use of this as a reference where needed.

Participants should produce a reproducibility report, describing the target questions, experimental
methodology, implementation details, analysis and discussion of findings, conclusions
on reproducibility of the paper. Generally, a report should include any information future researchers
or practitioners would find useful for reproducing or building upon the chosen paper.
The results of any experiments should be included; a “negative result” which doesn’t support
the main claims of the original paper is still valuable.

## Installation

In order to create a conda environment that mirrors the development environment run the following command:

```bash
conda create --name 1352200 --file requirements.txt
```

In order to run the code, run the following commands:

```bash
cd code;
python Main.py
```
