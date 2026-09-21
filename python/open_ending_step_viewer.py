import sys

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_GRAY, Quantity_TOC_RGB
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Display.SimpleGui import init_display


reader = STEPControl_Reader()
if reader.ReadFile(sys.argv[1]) != 1:
    raise RuntimeError(f"Unable to read STEP file: {sys.argv[1]}")
reader.TransferRoots()

display, start_display, _, _ = init_display()
shape = reader.OneShape()
display.DisplayShape(
    shape,
    color=Quantity_NOC_GRAY,
    transparency=0.82,
    update=False,
)
edges = []
edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
while edge_explorer.More():
    edges.append(edge_explorer.Current())
    edge_explorer.Next()
black = Quantity_Color(0.0, 0.0, 0.0, Quantity_TOC_RGB)
display.DisplayShape(edges, color=black, update=False)
display.FitAll()
display.Repaint()
start_display()
