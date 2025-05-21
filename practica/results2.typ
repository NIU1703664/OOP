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
  caption: "RMSE: ",
  image("images/die.png", height: 80%)
)

== Iris trees
#figure(
  caption: [Decision trees for Iris],
  grid(
    columns: (auto, auto),
    gutter: 3pt,
    align(center)[
      #image("images/die.png")
    ],
    align(center)[
      #image("images/die.png")
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
      image("images/die.png")
    )
  ],
  align(center)[
    #figure(
      caption: "Feature importance for Sonar",
      image("images/die.png")
    )
  ]
)

== Feature importance MNIST
#figure(
  caption: "Feature importance for MNIST",
  image("images/die.png", height: 80%)
)
