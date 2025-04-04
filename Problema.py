from aima import Problem, Node, PriorityQueue, memoize
from collections.abc import Callable
import time
from collections.abc import Callable
from copy import deepcopy

BLUE = "\033[34;1m"
RED = "\033[31;1m"
GREEN = "\033[32;1m"
RESET = "\033[0m"

#Metodo per risolvere un problema con un dato algoritmo
def execute(name: str, algorithm: Callable, problem: Problem, *args) -> None:
    print(f"{BLUE}{name}{RESET}")
    start = time.time()
    sol = algorithm(problem, *args)
    end = time.time()
    print(f"{GREEN}Problem:{RESET}\n{problem.initial}\n{GREEN}Goal:{RESET}\n{problem.goal}")
    print(f"{GREEN}Result:{RESET} {sol.solution() if sol is not None else '---'}")
    if isinstance(sol, Node):
        print(f"{GREEN}Path Cost:{RESET} {sol.path_cost}")
        print(f"{GREEN}Path Length:{RESET} {sol.depth}")
    print(f"{GREEN}Time:{RESET} {end - start} s")

#Algoritmo di ricerca best_first_search
def bfss(problem: Problem, f: Callable) -> Node:
    node: Node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    f = memoize(f, 'f')
    frontiera = PriorityQueue('min', f)
    frontiera.append(node)
    esplorati = set()
    while frontiera:
        node = frontiera.pop()
        if problem.goal_test(node.state):
            print(f"Numero di nodi espansi:{len(esplorati)}")
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
    return None

#Definizione A* g(n) + h(n) che utilizza la prima euristica
def aStar(problema: Problem, h : Callable | None = None) -> Node:
    h = memoize(h or problema.h, 'h')
    return bfss(problema, lambda node : h(node) + node.path_cost)

#Definizione A* g(n) + h(n) che utilizza la seconda euristica
def aStar2(problema: Problem, h : Callable | None = None) -> Node:
    h = memoize(h or problema.h2, 'h')
    return bfss(problema, lambda node : h(node) + node.path_cost)

#Definizione UCS
def ucs(problem: Problem) -> Node:
  return bfss(problem, lambda node : node.path_cost)

class Board():
    '''La classe Board rappresenta il dominio del problema, ovvero una griglia 6*6
    nella quale 6 blocchi numerati possono essere disposti uno sopra l'altro'''

    def __init__(self, matrix):
        self.matrix = matrix

    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        return self.matrix == other.matrix
    
    def __lt__(self, other):
        return hash(self) < hash(other)

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.matrix))

    def __str__(self):
        ruotata = [list(row) for row in zip(*self.matrix[::-1])]
        specchiata = [row[::-1] for row in ruotata]
        return "\n".join([" ".join(f"{cell:2}" for cell in row) for row in specchiata])

    def __repr__(self):
        return self.__str__()
    
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
        return positions

class BlocksWorldProblem(Problem):
    '''Il seguente problema consiste nello spostare i blocchi in uno spazio da una certa
    configurazione iniziale ad una configurazione finale'''
    
    #Costruttore della classe
    def __init__(self, initial: Board, goal: Board):
        self.initial = initial
        self.goal = goal
        #Coordinate finali di tutti i blocchi, utilizzate nelle euristiche
        self.goal_positions = {self.goal.matrix[x][y]: (x, y) for x in range(len(self.goal.matrix)) for y in range(6) if self.goal.matrix[x][y] != 0}

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
        x, y, nx, ny = action #Raccolgo l'azione da effettuare
        next_board = Board(deepcopy(state.matrix)) #Creo una copia dello stato della Board
        temp = next_board.matrix[x][y]
        next_board.matrix[x][y] = 0 #Scambio gli elementi alle due posizioni
        next_board.matrix[nx][ny] = temp
        return next_board

    #Metodo che controlla se lo stato che riceve come parametro è lo stato goal
    def goal_test(self, state):
        for x in range(len(state.matrix)):
            for y in range(6):
                if state.matrix[x][y] != self.goal.matrix[x][y]:
                    return False
        return True

    def path_cost(self, c, state1, action, state2):
        #Le azioni non hanno costi particolari
        return c + 1
    
    #Prima euristica, soluzioni subottimali ma veloce
    def h(self, node):
        state = node.state  # Prendiamo lo stato dal nodo
        euristica = 0
        #print(state,"\n")
        valori = list(self.goal_positions.keys())
        coordinateX = []
        coordinateY = []
        for valore, (x, y) in self.goal_positions.items():
            coordinateX.append((x))
            coordinateY.append((y))
        for x in range(len(state.matrix)):
            for y in range(6):
                block = state.matrix[x][y]
                #controllare se sta in un goal state E se sotto di lui non ha blocchi che non sono in goal
                if block != 0:
                    for sus in valori:
                        if valori[sus-1] == block:
                            if(x == coordinateX[sus-1] and y == coordinateY[sus-1]): 
                                #siamo contenti SOLO SE SOTTO NON CI SONO SBURATORI
                                euristica -= y
                            #se un goal e bloccato da unaltro blocco non ci paice
                            elif x in coordinateX and y in coordinateY:
                                euristica += y + 1000
                            else:
                                # FIXME: il 5 e hard codato deve essere y.maxlength()
                                euristica += 5 -1 - y
        return euristica
    
    #Seconda euristica, soluzioni ottimali ma lenta
    def h2(self, node):
        state = node.state  # Prendiamo lo stato dal nodo
        distance = 0
        for x in range(len(state.matrix)):
            for y in range(6):
                block = state.matrix[x][y]
                if block != 0 and block in self.goal_positions:
                    gx, gy = self.goal_positions[block]
                    base_distance = abs(x - gx) + abs(y - gy)
                    # Penalizza blocchi bloccati sotto altri
                    penalty = sum(1 for yy in range(y+1, 6) if state.matrix[x][yy] != 0)
                    distance += base_distance + penalty  # Penalizza blocchi se devono essere liberati prima
        return distance
    
#Piccoli test    
tavola = Board([[0,0,0,1,4,5],[0,0,0,0,0,0],[0,0,0,0,0,6],[0,0,0,0,0,0],[0,0,0,0,3,2],[0,0,0,0,0,0]])
problema1 = BlocksWorldProblem(tavola, Board([[0,0,0,5,1,4],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,2,6,3]]))
problema2 = BlocksWorldProblem(Board([[1,2,3,4,5,6],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]), Board([[6,5,4,3,2,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]))
problema3 = BlocksWorldProblem(Board([[0,0,0,0,0,1],[0,0,0,0,0,2],[0,0,0,0,0,3],[0,0,0,0,0,4],[0,0,0,0,0,5],[0,0,0,0,0,6]]), Board([[0,0,0,0,0,6],[0,0,0,0,0,5],[0,0,0,0,0,4],[0,0,0,0,0,3],[0,0,0,0,0,2],[0,0,0,0,0,1]]))
problema4 = BlocksWorldProblem(Board([[0,0,0,0,0,0],[0,0,0,0,5,6],[0,0,0,0,3,4],[0,0,0,0,1,2],[0,0,0,0,0,0],[0,0,0,0,0,0]]), Board([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,2],[0,0,0,0,3,4],[0,0,0,0,5,6],[0,0,0,0,0,0]]))
problema5 = BlocksWorldProblem(Board([[0,0,0,0,0,0],[0,0,0,0,0,6],[0,0,0,0,3,4],[0,0,0,5,1,2],[0,0,0,0,0,0],[0,0,0,0,0,0]]), Board([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,2],[0,0,0,0,3,4],[0,0,0,0,5,6],[0,0,0,0,0,0]]))
problema6 = BlocksWorldProblem(Board([[0,0,0,0,0,0],[0,0,0,0,0,2],[0,0,0,0,4,6],[0,0,0,0,1,3],[0,0,0,0,0,5],[0,0,0,0,0,0]]), Board([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,2],[0,0,0,0,3,4],[0,0,0,0,5,6]]))
problema7 = BlocksWorldProblem(Board([[6,5,4,3,2,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]), Board([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,2],[0,0,0,0,3,4],[0,0,0,0,5,6]]))
problema8 = BlocksWorldProblem(Board([[1,2,3,4,5,6],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]), Board([[5,4,3,2,1,6],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]))
problema9 = BlocksWorldProblem(Board([[1,2,3,4,5,6],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]), Board([[0,0,0,0,0,0],[5,4,3,2,1,6],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]))
problema10 = BlocksWorldProblem(Board([[0,0,0,1,2,3],[0,0,0,4,5,6],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]), Board([[0,0,0,0,0,0],[0,0,0,5,4,3],[0,0,0,0,0,0],[0,0,0,2,1,6],[0,0,0,0,0,0],[0,0,0,0,0,0]]))

execute("A*", aStar, problema1)
execute("A*", aStar, problema2)
execute("A*", aStar, problema3)
execute("A*", aStar, problema4)
execute("A*", aStar, problema5)
execute("A*", aStar, problema6)
execute("A*", aStar, problema7)
execute("A*", aStar, problema8)
execute("A*", aStar, problema9)
execute("A*", aStar, problema10)