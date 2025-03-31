from aima import Problem, Node, PriorityQueue, GraphProblem, memoize, romania_map
from collections.abc import Callable
from collections import deque
from colorama import Fore, Back, Style
import time, sys

BLUE = "\033[34;1m"
RED = "\033[31;1m"
GREEN = "\033[32;1m"
RESET = "\033[0m"

def execute(name: str, algorithm: Callable, problem: Problem, *args, **kwargs) -> None:
    print(f"{RED}{name}{RESET}\n")
    start = time.time()
    sol = algorithm(problem, *args, **kwargs)
    end = time.time()
    if problem.goal is not None:
        print(f"\n{GREEN}PROBLEM:{RESET} {problem.initial} -> {problem.goal}")
    if isinstance(sol, Node):
        print(f"{GREEN}Total nodes generated:{RESET} {sol.nodes_generated}")
        print(f"{GREEN}Paths explored:{RESET} {sol.paths_explored}")
        print(f"{GREEN}Nodes left in frontier:{RESET} {sol.nodes_left_in_frontier}")
        sol = sol.result
    print(f"{GREEN}Result:{RESET} {sol.solution() if sol is not None else '---'}")
    if isinstance(sol, Node):
        print(f"{GREEN}Path Cost:{RESET} {sol.path_cost}")
        print(f"{GREEN}Path Length:{RESET} {sol.depth}")
    print(f"{GREEN}Time:{RESET} {end - start} s")

# TODO: algoritmo A* con tutto il resto asdsadsaasd

def bfss (problema: Problem, f: Callable) -> Node:
    node: Node = Node(problema.initial)
    if problema.goal_test(node.state):
        return node
    f = memoize(f, 'f')
    node = node(problema.initial)
    frontiera = PriorityQueue('min', f)
    frontiera.append(node)
    esplorato = set()
    contatore = 1
    while frontiera:
        node = frontiera.pop()
        if problema.goal_test(node.state):
            return node
        esplorato.add(node.state)
        for figlio in node.expand(problema):
            contatore += 1
            if figlio.state not in esplorato and figlio not in frontiera:
                frontiera.append(figlio)
            elif figlio in frontiera:
                if f(figlio) < frontiera[figlio]:
                    del frontiera[figlio]
                    frontiera.append(figlio)
    return None

def aStar(problema: Problem, h : Callable | None = None) -> Node:
    h = memoize(h or problema.h, 'h')
    return bfss(problema, lambda node : h(node) + node.path_cost)

# TEST
romania_problem = GraphProblem('Arad','Bucharest', romania_map)
execute("BFS su grafo", aStar, romania_problem)


# TODO: Classe del problema
class Board(dict):
    vuoto = '0'
    fuori = '#'
    #funzioni sborranti






#TODO: Risoluzione del problema 
class Blocconi(Game):
    sborra = ""