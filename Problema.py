from aima import Problem, Node, PriorityQueue, GraphProblem, memoize
from collections.abc import Callable
from collections import deque
from colorama import Fore, Back, Style
from collections.abc import Callable
import time, sys
import functools
import random
import math

# TODO: algoritmo A* con tutto il resto 

def bfss (problema: Problem, f: Callable) -> Node:
    node: Node = Node(problema.initial)
    if problema.goal_test(node.state):
        return node
    f = memoize(f, 'f')






# TODO: Classe del problema
class Board(dict):
    vuoto = '0'
    fuori = '#'
    #funzioni sborranti






#TODO: Risoluzione del problema 
class Blocconi(Game):
    sborra = ""