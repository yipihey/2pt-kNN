#set page(width: 16cm, height: 11cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#let steelblue = rgb("#4682B4")
#let crimson = rgb("#DC143C")
#let seagreen = rgb("#2E8B57")
#let darkorange = rgb("#FF8C00")
#let level-colors = (steelblue, seagreen, darkorange, crimson)
#let r = (3.90625000, 5.52427173, 6.76582347, 7.81250000, 11.04854346, 13.53164693, 15.62500000, 22.09708691, 27.06329387, 31.25000000, 44.19417382, 54.12658774, 62.50000000, 88.38834765, 108.25317547, 125.00000000, 176.77669530, 216.50635095, 250.00000000, 353.55339059, 433.01270189)
#let res = (-592.29993996, -473.95385754, -138.50144082, -67.78797378, -8.83748268, 20.61416393, 1.77669980, 0.95076485, 0.37434592, 2.16500179, 2.46601357, 3.83608574, 1.10517810, -0.03263176, -0.20972627, 0.37042433, -0.34162348, -0.38010031, -0.03888823, -0.92769288, -0.57876164)

#lq.diagram(
  title: [Morton Grid: Residuals $(hat(xi) - xi_"true") / sigma$],
  xlabel: [$r$ #h(0.3em) $[$h$""^(-1)$ Mpc$]$],
  ylabel: [Residual [$sigma$]],
  xscale: "log", yscale: "linear", xlim: auto, ylim: (-10.00000000, 10.00000000),
  lq.hlines(0, stroke: (paint: black, thickness: 0.5pt)),
  lq.hlines(2, stroke: (dash: "dashed", paint: gray, thickness: 0.5pt)),
  lq.hlines(-2, stroke: (dash: "dashed", paint: gray, thickness: 0.5pt)),
  lq.plot(r, res,
    stroke: none,
    mark: "o",
    mark-size: 5pt,
    color: steelblue),
)
