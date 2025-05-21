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

== Min Temperatures
#figure(
  caption: "RMSE: 3.1",
  image("./figures/FeatureImportance_temperatures.png", height: 85%)
)

== Iris trees
#figure(
  caption: [Decision trees for Iris],
  grid(
    columns: (auto, auto),
    gutter: 1em,
    align(left)[
      #show raw: set text(size: 12pt)
      ```
      parent - feature index 3, threshold 1.0
      |   leaf, 0
      |   parent - feature index 0, threshold 6.4
      |   |   parent - feature index 2, threshold 4.9
      |   |   |   leaf, 1
      |   |   |   parent - feature index 3, threshold 1.8
      |   |   |   |   leaf, 1
      |   |   |   |   leaf, 2
      |   |   parent - feature index 2, threshold 5.2
      |   |   |   leaf, 1
      |   |   |   leaf, 2
      ```
    ],
    align(center)[
      #show raw: set text(size: 12pt)
      ```
      parent - feature index 0, threshold 5.5
      |   parent - feature index 2, threshold 3.3
      |   |   leaf, 0
      |   |   leaf, 1
      |   parent - feature index 0, threshold 6.4
      |   |   parent - feature index 3, threshold 1.8
      |   |   |   parent - feature index 3, threshold 1.0
      |   |   |   |   leaf, 0
      |   |   |   |   parent - feature index 2, threshold 5.6
      |   |   |   |   |   leaf, 1
      |   |   |   |   |   leaf, 2
      |   |   |   leaf, 2
      |   |   parent - feature index 3, threshold 1.8
      |   |   |   leaf, 1
      |   |   |   leaf, 2
      ```
    ]
  ),
)

== Feature importance Sonar and Iris
#grid(
  columns: (auto, auto),
  gutter: 3pt,
  align(center)[
    #figure(
      caption: "Feature importance for Iris",
      image("./figures/FeatureImportance_iris.png")
    )
  ],
  align(center)[
    #figure(
      caption: "Feature importance for Sonar",
      image("./figures/FeatureImportance_sonar.png")
    )
  ]
)

== Feature importance MNIST
#figure(
  caption: "Feature importance for MNIST",
  image("./figures/FeatureImportance_MNIST.png", height: 85%)
)
