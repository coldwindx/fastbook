#hide_input
#caption  A traditional program
#is basic_program
#alt Pipeline inputs, program, results
import fastbook
from fastai.vision.all import *
import graphviz
s = "program[shape=box3d width=1 height=0.7] inputs->program->results"
graphviz.Source('digraph G{ rankdir="LR"' + s + '; }')