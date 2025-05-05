#set page(flipped: true)
#set text(19pt)
#let result(database, times, accuracies) = table(
  columns: (1fr, auto, auto, auto),
  inset: 10pt,
  align: center,
  table.header(
    [Purity Measure: _#{ database }_],
    [*Computing method*],
    [*Total training time*],
    [*Accuracy*],
  ),
  table.cell(
    align: horizon,
    rowspan: 2,
    [*Random Forest*],
  ), "Sequential", [#{ times.at(0) }s], [#{ accuracies.at(0) }%],
  "Parallel", [#{ times.at(1) }s], [#{ accuracies.at(1) }%],
  table.cell(
    align: horizon,
    rowspan: 2,
    [*Extra Trees*],
  ), "Sequential", [#{ times.at(2) }s], [#{ accuracies.at(2) }%],
  "Parallel", [#{ times.at(3) }s], [#{ accuracies.at(3) }%],
)
#align(center + horizon)[
  #box(
    width: 100%,
    height: 100%,
    inset: 20pt,
    stroke: 1pt + black,
    [
      #text(size: 48pt, "Optimization results")\
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
#box(
  width: 100%,
  inset: 20pt,
  stroke: 1pt + black,
  [Best purity measure for performance: *Gini* \
    Worst time: 0.656s on sequential random forest using entropy\
    Best time: 0.089s on sequential extra trees using Gini\
    Performance Improvement:\
    #h(20pt)$"Extra trees" => "7.3 times faster"$],
)
#pagebreak()
#result(
  "Gini",
  ("0.614", "0.189", "0.089", "0.094"),
  ("93.3", "95.6", "93.3", "93.3"),
)
#result(
  "Entropy",
  ("0.656", "0.201", "0.092", "0.091"),
  ("95.6", "93.3", "95.0", "95.6"),
)
#pagebreak()
== Sonar
Metaparameters:
- Number of trees: 84
- Max depth of trees: 20
- Minimum size of each split: 5
- Ratio of samples: 0.8
- Ratio of rows for training: 0.7
#box(
  width: 100%,
  inset: 20pt,
  stroke: 1pt + black,
  [Best purity measure for performance: *Entropy* \
    Worst time: 11.893s on sequential random forest using Gini\
    Best time: 0.168s on parallel extra trees using entropy\
    Performance Improvement:\
    #h(20pt)$"Extra trees " + " Parallel processing" => "70.79 times faster"$],
)
#pagebreak()
#result(
  "Gini",
  ("11.893", "2.006", "0.435", "0.177"),
  ("77.8", "74.6", "81.0", "73.0"),
)
#result(
  "Entropy",
  ("10.269", "1.999", "0.391", "0.168"),
  ("79.4", "76.2", "76.2", "79.4"),
)
#pagebreak()
== MNIST
Metaparameters:
- Number of trees: 42
- Max depth of trees: 20
- Minimum size of each split: 20
- Ratio of samples: 0.4
- Ratio of rows for training: 0.7
#box(
  width: 100%,
  inset: 20pt,
  stroke: 1pt + black,
  [Best purity measure for performance: *Entropy* \
    Worst estimated time: 27233s or 7.6h on sequential random forest using Gini\
    Best time: 240.652s on parallel extra trees using Entropy\
    Performance Improvement:\
    #h(20pt)$"Extra trees " + " Parallel processing" => "113 times faster"$],
)
#pagebreak()
#result(
  "Gini",
  (
    [27233#footnote([Time estimated by measuring the time taken to build a single tree and assuming that the performance improvement of using parallel computing is of aproximately x6 as seen in the datasets of Iris and Sonar.])<prox>],
    [4538 @prox],
    "479.015",
    "253.498",
  ),

  ([>73.4 #footnote("Accuraccy of a single tree.")<acc>], "NA", "92.8", "92.8"),
)
#result(
  "Entropy",
  ([21040@prox], [3506@prox], "439.417", "240.562"),
  ([>73.3 @acc], "NA", "92.5", "92.5"),
)
