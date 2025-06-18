import numpy as np

def digitalizza(nums):
    print(nums)
    NUM_COLS = 6
    NUM_ROWS = 6

    # Crea una matrice 6x6 piena di zeri
    mat = np.zeros((NUM_COLS, NUM_ROWS), dtype=int)

    if not nums:
        return mat.tolist()

    # Estrai solo le X
    x_vals = [x for _, x, _ in nums]
    min_x = min(x_vals)
    max_x = max(x_vals)
    width = max_x - min_x

    # Tolleranza dinamica: se i numeri sono stretti su X, attiva la tolleranza
    if width < 300:
        x_tolerance = 250
    else:
        x_tolerance = 0

    # Calcolo dello step_size (divisione in 6 colonne)
    step_size = width / NUM_COLS if width != 0 else 1

    # Inizializza colonne
    columns = [[] for _ in range(NUM_COLS)]

    for tup in nums:
        digit, x, y = tup

        # Applica tolleranza se > 0
        if x_tolerance > 0:
            adjusted_x = round((x - min_x) / x_tolerance) * x_tolerance + min_x
        else:
            adjusted_x = x

        col_index = int(round((adjusted_x - min_x) / step_size)) if step_size > 0 else 0
        if col_index >= NUM_COLS:
            col_index = NUM_COLS - 1
        columns[col_index].append(tup)

    # Ordina ogni colonna per Y decrescente (dal basso verso l’alto)
    for group in columns:
        group.sort(key=lambda tup: tup[2], reverse=True)

    # Riempi la matrice
    for col_index, group in enumerate(columns):
        for j, (digit, x, y) in enumerate(group):
            if j < NUM_ROWS:
                row_index = NUM_ROWS - 1 - j
                mat[col_index, row_index] = digit

    return mat.tolist()
