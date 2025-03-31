from aima import Problem, Node, PriorityQueue, GraphProblem, memoize, romania_map
from collections.abc import Callable
from collections import deque
from colorama import Fore, Back, Style
import time, sys
from collections.abc import Callable
import functools
import random
import math


BLUE = "\033[34;1m"
RED = "\033[31;1m"
GREEN = "\033[32;1m"
RESET = "\033[0m"

def execute(name: str, algorithm: Callable, problem: Problem, *args) -> None:
    print(f"{BLUE}{name}{RESET}\n")
    start = time.time()
    sol = algorithm(problem, *args)
    end = time.time()
    print(f"\n{GREEN}PROBLEM:{RESET} {problem.initial} -> {problem.goal}")
    print(f"{GREEN}Result:{RESET} {sol.solution() if sol is not None else '---'}")
    if isinstance(sol, Node):
        print(f"{GREEN}Path Cost:{RESET} {sol.path_cost}")
        print(f"{GREEN}Path Length:{RESET} {sol.depth}")
    print(f"{GREEN}Time:{RESET} {end - start} s")

# TODO: algoritmo A* con tutto il resto asdsadsaasd

def bfss(problem: Problem, f: Callable) -> Node:
    node: Node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    f = memoize(f, 'f')

    frontiera = PriorityQueue('min', f)
    frontiera.append(node)
    esplorati = set()
    print(f"{BLUE}BFSSSSSSSSSSSSS:{RESET}")
    while frontiera:
        node = frontiera.pop()
        if problem.goal_test(node.state):
            print("YEEEEEE")
            return node
        esplorati.add(node.state)
        for child in node.expand(problem):
            if child.state not in esplorati and child not in frontiera:
                frontiera.append(child)
            elif child in frontiera:
                inc = frontiera.get_item(child)
                if f(child) < f(inc):
                    del frontiera[inc]
                    frontiera.append(child)
                    print(f"({inc}, {inc.f}) -- replaced by ({child}, {child.f})")
    return None

def aStar(problema: Problem, h : Callable | None = None) -> Node:
    h = memoize(h or problema.h, 'h')
    return bfss(problema, lambda node : h(node) + node.path_cost)

# TEST
romania_problem = GraphProblem('Arad','Bucharest', romania_map)
execute("BFS su grafo", aStar, romania_problem)






# TODO: Classe del problema
class Board():
    
    #funzioni sborranti
    def __init__(self, matrix):
        self.matrix = matrix
    
#[[0,0,0,1,4,5]]
#[[0,0,0,0,0,0]]
#[[0,0,0,0,0,6]]
#[[0,0,0,0,0,0]]
#[[0,0,0,0,3,2]]
#[[0,0,0,0,0,0]]

#TODO: Risoluzione del problema 
class Blocconi(Problem):
    
    def __init__(self, initial: Board, goal: Board):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        actions = []
        for x in len(state.matrix):
            for y in x:
                if state.matrix[x,y] != 0:
                    print
                    actions.append(tuple(x,y))
                    print(f"{x},{y}")
                    break
        return actions

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
    
tavola = Board([[0,0,0,1,4,5],[0,0,0,0,0,0],[0,0,0,0,0,6],[0,0,0,0,0,0],[0,0,0,0,3,2],[0,0,0,0,0,0]])
problema = Blocconi(tavola, Board([]))
problema.actions