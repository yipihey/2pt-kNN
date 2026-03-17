#set page(width: 16cm, height: 11cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#let steelblue = rgb("#4682B4")
#let crimson = rgb("#DC143C")
#let seagreen = rgb("#2E8B57")
#let darkorange = rgb("#FF8C00")
#let level-colors = (steelblue, seagreen, darkorange, crimson)
#let r = (3.90625000, 5.52427173, 6.76582347, 7.81250000, 11.04854346, 13.53164693, 15.62500000, 22.09708691, 27.06329387, 31.25000000, 44.19417382, 54.12658774, 62.50000000, 88.38834765, 108.25317547, 125.00000000, 176.77669530, 216.50635095, 250.00000000, 353.55339059, 433.01270189)
#let res = (-626.51042803, -423.17063964, -127.18047058, -55.56410105, -7.12393564, 13.00928213, 1.83477680, 1.24284799, 0.52666801, 2.81468077, 1.88843567, 1.61974213, 0.61293740, 0.07443695, 0.11681047, 0.29543435, -0.41847043, -0.25066833, -0.01726014, -1.03260309, -0.57203753)

#lq.diagram(
  title: [Morton Grid: Residuals $(hat(xi) - xi_"true") / sigma$],
  xlabel: [$r$ #h(0.3em) $[$h$""^(-1)$ Mpc$]$],
  ylabel: [Residual [$sigma$]],
  xscale: "log", yscale: "linear", xlim: auto, ylim: auto,
  lq.hlines(0, stroke: (paint: black, thickness: 0.5pt)),
  lq.hlines(2, stroke: (dash: "dashed", paint: gray, thickness: 0.5pt)),
  lq.hlines(-2, stroke: (dash: "dashed", paint: gray, thickness: 0.5pt)),
  lq.plot(r, res,
    stroke: none,
    mark: "o",
    mark-size: 5pt,
    color: steelblue),
)
