import numpy as np

x_tolerance = 250

def digitalizza(nums):
    # Crea una matrice 6x6 piena di zeri.
    mat = np.zeros((6, 6), dtype=int)
    
    # Se la lista è vuota, ritorna subito la matrice.
    if not nums:
        return mat
    
    # Trova il valore minimo e massimo di X fra tutti i numeri.
    min_x = min(nums, key=lambda tup: tup[1])[1]
    max_x = max(nums, key=lambda tup: tup[1])[1]

    # Calcola la larghezza totale (distanza tra max_x e min_x).
    width = max_x - min_x
    min_x = min(nums, key=lambda tup: tup[1])[1]
    max_x = max(nums, key=lambda tup: tup[1])[1]
    
    # Calcola la larghezza totale (distanza tra max_x e min_x).
    width = max_x - min_x
    # Divide la larghezza in 6 step. Se width è 0, imposta step_size a 1 per evitare divisioni per zero.
    step_size = width / 6 if width != 0 else 1

    # Crea 6 colonne (liste) vuote.
    columns = [[] for _ in range(6)]
    
    # Distribuisce ogni tupla nella colonna (sarebbero gli step) corretta usando round() per arrotondare le piccole variazioni.
    for tup in nums:
        digit, x, y = tup

        # Arrotonda x per tolleranza prima di calcolare col_index
        adjusted_x = round((x - min_x) / x_tolerance) * x_tolerance + min_x

        col_index = int(round((adjusted_x - min_x) / step_size)) if step_size > 0 else 0
        # Garanzia di non superare il limite di 6 colonne.
        if col_index >= 6:
            col_index = 5
        columns[col_index].append(tup)
    
    # Per ogni colonna ordina le tuple in base alla coordinata Y.
    for group in columns:
        group.sort(key=lambda tup: tup[2])
    
    # Carica i numeri ordinati nella matrice.
    # In ogni colonna riordina in ordine decrescente di Y: il numero con Y maggiore verrà in basso (riga 5).
    for col_index, group in enumerate(columns):
        if not group:
            continue
        group.sort(key=lambda tup: tup[2], reverse=True)
        # Inserisce i numeri a partire dal basso (riga 5) fino alla riga 0.
        for j, (digit, x, y) in enumerate(group):
            # Limite 6 numeri per colonna
            if j < 6:
                row_index = 5 - j
                mat[col_index, row_index] = digit
    return mat.tolist()
