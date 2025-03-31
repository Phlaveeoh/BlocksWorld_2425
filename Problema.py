from aima import Problem, Node, PriorityQueue, GraphProblem, memoize, breadth_first_graph_search
from collections.abc import Callable
from collections import deque
from colorama import Fore, Back, Style
import time, sys
from collections.abc import Callable
import functools
import random
import math
from copy import deepcopy

BLUE = "\033[34;1m"
RED = "\033[31;1m"
GREEN = "\033[32;1m"
RESET = "\033[0m"

#Metodo per risolvere un problema con un dato algoritmo
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

#Algoritmo di ricerca
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

#Definizione A* g(n) + h(n)
def aStar(problema: Problem, h : Callable | None = None) -> Node:
    h = memoize(h or problema.h, 'h')
    return bfss(problema, lambda node : h(node) + node.path_cost)

#Definizione UCS
def ucs(problem: Problem) -> Node:
  return bfss(problem, lambda node : node.path_cost)

class Board():
    '''La classe Board rappresenta il dominio del problema, ovvero una griglia 6*6
    nella quale 6 blocchi numerati possono essere disposti uno sopra l'altro'''
    
    #Costruttore della classe
    def __init__(self, matrix):
        self.matrix = matrix
    
    #Metodo che restituisce gli spostamenti possibili che può fare un dato blocco
    def get_legal_positions(self, x, y):
        positions = []
        #Scorro tutta la board
        for nx in range(len(self.matrix)):
            for ny in range(6):
                #Appena trovo una posizione legale dove il blocco può spostarsi la aggiungo alle posizioni
                # FIXME: Condizione posizione legale
                if(nx != x and self.matrix[nx][ny] == 0 and (ny == 5 or self.matrix[nx][ny+1] != 0)):
                    #l'azione effettuabile è rappresentata da una tupla(x, y, nx, ny)
                    # dove la prima coppia di coordinate è la posizione attuale del blocco, la seconda coppia è la nuova posizione dove verrà spostato
                    positions.append((x, y, nx, ny))
                    print(f"{self.matrix[x][y]} da {x},{y} a {nx},{ny}")
        return positions

#Esempio di stato del problema
#[[0,0,0,1,4,5]]
#[[0,0,0,0,0,0]]
#[[0,0,0,0,0,6]]
#[[0,0,0,0,0,0]]
#[[0,0,0,0,3,2]]
#[[0,0,0,0,0,0]]


class BlocksWorldProblem(Problem):
    '''Il seguente problema consiste nello spostare i blocchi in uno spazio da una certa
    configurazione iniziale ad una configurazione finale'''
    
    #Costruttore della classe
    def __init__(self, initial: Board, goal: Board):
        self.initial = initial
        self.goal = goal

    #Metodo che restituisce tutte le azioni possibili da un dato stato
    def actions(self, state):
        actions = []
        #Scorro tutta la Board
        for x in range(len(state.matrix)):
            for y in range(6):
                #Appena trovo un blocco che può essere spostato prendo la sua posizione e controllo con get_legal_positions dove può essere spostato
                if state.matrix[x][y] != 0:
                    actions = actions + state.get_legal_positions(x, y)
                    break
        #Ritorno l'insieme delle azioni effettuabili
        return actions

    #metodo che restituisce un nuovo stato che è il risultato di una determinata azione
    def result(self, state, action):
        x, y, nx, ny = action
        next_board = Board(deepcopy(state.matrix))
        temp = next_board.matrix[x][y]
        next_board.matrix[x][y] = 0
        next_board.matrix[nx][ny] = temp
        print(next_board.matrix)
        return next_board

    #Metodo che controlla se lo stato che riceve come parametro è lo stato goal
    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        print("testiamo se abbiamo finito")
        for x in range(len(state.matrix)):
            for y in range(6):
                if state.matrix[x][y] != self.goal.matrix[x][y]:
                    return False
        return True

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

#Piccoli test 
tavola = Board([[0,0,0,1,4,5],[0,0,0,0,0,0],[0,0,0,0,0,6],[0,0,0,0,0,0],[0,0,0,0,3,2],[0,0,0,0,0,0]])
problema = BlocksWorldProblem(tavola, Board([[0,0,0,1,4,5],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,6,3,2],[0,0,0,0,0,0]]))
problema2 = BlocksWorldProblem(Board([[0,0,0,1,4,5],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,6,3,2],[0,0,0,0,0,0]]), Board([[0,0,0,1,4,5],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,6,3,2],[0,0,0,0,0,0]]))
    
execute("BFS", breadth_first_graph_search, problema)