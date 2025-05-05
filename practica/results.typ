#set page(flipped: true)
#set text(20pt)

#align(center + horizon)[
  #box(
    width: 100%,
    height: 100%,
    inset: 20pt,
    stroke: 1pt + black,
    [
      #text(size: 48pt, "Accuracy results")\
      Random Forest - Group 17
    ],
  )]
== Iris
Metaparameters:
- Number of trees: 84
- Max depth of trees: 20
- Minimum size of each split: 5
- Ratio of samples: 0.8
- Ratio of rows for training: 0.7
#figure(
  caption: "Values obtained from the average of running 5 tests",
  table(
    columns: (auto, auto),
    inset: 20pt,
    align: center,
    table.header(
      [*Purity Measure*],
      [*Accuracy*],
    ),

    "Gini", [93.76%],
    "Entropy", [92.46%],
  ),
)
== Sonar
Metaparameters:
- Number of trees: 84
- Max depth of trees: 20
- Minimum size of each split: 5
- Ratio of samples: 0.8
- Ratio of rows for training: 0.7
#figure(
  caption: "Values obtained from the average of running 5 tests",
  table(
    columns: (auto, auto),
    inset: 20pt,
    align: center,
    table.header(
      [*Purity Measure*],
      [*Accuracy*],
    ),

    "Gini", [77.14%],
    "Entropy", [78.42%],
  ),
)
== MNIST
Metaparameters:
- Number of trees: 42
- Max depth of trees: 20
- Minimum size of each split: 20
- Ratio of samples: 0.4
- Ratio of rows for training: 0.7
#figure(
  caption: "Values obtained from the average of running 5 tests",
  table(
    columns: (auto, auto),
    inset: 20pt,
    align: center,
    table.header(
      [*Purity Measure*],
      [*Accuracy*],
    ),

    "Gini", [92.8%],
    "Entropy", [92.5%],
  ),
)
