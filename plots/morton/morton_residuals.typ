#set page(width: 16cm, height: 11cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#let steelblue = rgb("#4682B4")
#let crimson = rgb("#DC143C")
#let seagreen = rgb("#2E8B57")
#let darkorange = rgb("#FF8C00")
#let level-colors = (steelblue, seagreen, darkorange, crimson)
#let r = (62.50000000, 88.38834765, 108.25317547, 125.00000000, 176.77669530, 216.50635095, 250.00000000, 353.55339059, 433.01270189)
#let res = (1.11332109, -0.02824687, -0.27753552, 0.35795155, -0.42274954, -0.34840611, -0.02369201, -0.93254766, -0.55004880)

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
