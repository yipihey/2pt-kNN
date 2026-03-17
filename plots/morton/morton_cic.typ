#set page(width: 16cm, height: 11cm, margin: 5mm)
#set text(font: "New Computer Modern")
#import "@preview/lilaq:0.6.0" as lq
#let steelblue = rgb("#4682B4")
#let crimson = rgb("#DC143C")
#let seagreen = rgb("#2E8B57")
#let darkorange = rgb("#FF8C00")
#let level-colors = (steelblue, seagreen, darkorange, crimson)
#let r-eff = (155.08762272, 77.54381136, 38.77190568, 19.38595284, 9.69297642, 4.84648821, 2.42324411)
#let vom = (10.77240000, 5.39840000, 3.35015000, 2.28079375, 1.67125445, 1.34455195, 1.13681338)

#lq.diagram(
  title: [Counts-in-Cells: $"Var" slash N$ vs Scale],
  xlabel: [$r_"eff"$ #h(0.3em) $[$h$""^(-1)$ Mpc$]$],
  ylabel: [$"Var"(N) / angle.l N angle.r$],
  xscale: "log", yscale: "linear", xlim: auto, ylim: auto,
  lq.hlines(1, stroke: (dash: "dashed", paint: gray, thickness: 0.8pt), label: [Poisson]),
  lq.plot(r-eff, vom,
    stroke: (paint: crimson, thickness: 2pt),
    mark: "o",
    mark-size: 5pt,
    color: crimson,
    label: [Morton CIC]),
)
