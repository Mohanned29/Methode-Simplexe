import numpy as np

"""                              MADE BY JINX                                   """

def get_user_input():
    while True:
        try:
            num_vars = int(input("nombre de variables (max 3) : "))
            if 1 <= num_vars <= 3:
                break
            else:
                print("veuillez entrer un nombre entre 1 et 3.")
        except ValueError:
            print("entree invalide , entrez un nombre entier")

    #collecte les coefficients de la fonction objectif
    print("\nentrez les coefficients de la fonction objectif (max z):")
    obj_coeffs = [] 
    for i in range(num_vars): 
        while True: 
            try: 
                coeff = float(input(f"Coefficient x{i+1} : ")) 
                obj_coeffs.append(coeff) 
                break 
            except ValueError: 
                print("Entree invalide , entrez un nombre.") 
     
    #collecte les contraintes 
    print("\nEntrez les contraintes (forme ≤):")
    print("Une contrainte représente une limite sur la combinaison des variables.")
    print("Par exemple, 2x1 + 3x2 <= 10 signifie que 2 fois la valeur de x1 plus 3 fois la valeur de x2 doit être inférieure ou égale à 10.")
    constraints = [] 
    for i in range(num_vars + 1): 
        constraint = [] 
        print(f"\nContrainte {i+1}:") 
        for j in range(num_vars): 
            while True: 
                try: 
                    coeff = float(input(f"Coefficient x{j+1} (nombre réel) : ")) 
                    constraint.append(coeff) 
                    break 
                except ValueError: 
                    print("Entree invalide, veuillez entrer un nombre.") 
         
        while True: 
            try: 
                rhs = float(input("Terme constant (membre de droite) : ")) 
                constraint.append(rhs) 
                break 
            except ValueError: 
                print("Entree invalide , Entrez un nombre.") 
         
        constraints.append(constraint) 
     
    # Afficher un résumé des entrées
    print("\nRésumé des entrées:")
    print("Fonction objectif : max z =", *obj_coeffs)
    print("Contraintes :")
    for i, constraint in enumerate(constraints):
        print(f"  {i+1}: {constraint[0]}x1 + {constraint[1]}x2 <= {constraint[2]}")

    return obj_coeffs, constraints 
 
def convert_to_canonical_form(obj_coeffs, constraints): 
    """Conversion a la forme canonique""" 
    #ajouter les variables d'ecart 
    num_vars = len(obj_coeffs) 
    slack_vars = np.eye(len(constraints)) 
 
    canonical_matrix = np.column_stack([ 
        constraints[:, :num_vars],  
        slack_vars,  
        constraints[:, -1].reshape(-1, 1) 
    ]) 
     
    #mise a jour des coefficients de la fonction objectif 
    full_obj_coeffs = np.concatenate([ 
        obj_coeffs,  
        np.zeros(len(constraints)) 
    ]) 
     
    return canonical_matrix, full_obj_coeffs 
 
def simplex_solve(obj_coeffs, constraints): 
    """Resolution par la methode du simplexe""" 
    #conversion à la forme canonique 
    tableau, obj_coeffs = convert_to_canonical_form(obj_coeffs, constraints) 
     
    #initialisation du tableau du simplexe 
    num_vars = len(obj_coeffs) - len(constraints) 
    base_vars = list(range(num_vars, num_vars + len(constraints))) 
     
    iteration = 0 
    while True: 
        #number of iteration 
        print(f"\n--- Iteration {iteration} ---") 
        print_tableau(tableau, obj_coeffs, base_vars) 
         
        #determination de la variable entrante 
        enter_col = determine_entering_variable(obj_coeffs, tableau) 
        if enter_col is None: 
            break 
         
        #determination de la variable sortante 
        leave_row = determine_leaving_variable(tableau, enter_col) 
        if leave_row is None: 
            print("Problème non borné.") 
            return None 
         
        #pivotage 
        pivot = tableau[leave_row, enter_col] 
        tableau[leave_row, :] /= pivot 
         
        for i in range(tableau.shape[0]): 
            if i != leave_row: 
                factor = tableau[i, enter_col] 
                tableau[i, :] -= factor * tableau[leave_row, :] 
         
        base_vars[leave_row] = enter_col 
         
        iteration += 1 
     
 
    solution = np.zeros(len(obj_coeffs)) 
    for i, var in enumerate(base_vars): 
        if var < len(solution): 
            solution[var] = tableau[i, -1] 
     
    return solution, -tableau[-1, -1] 
 
def determine_entering_variable(obj_coeffs, tableau): 
    reduced_costs = obj_coeffs[:-1] - np.dot(tableau[:, :-1].T, tableau[-1, :-1]) 
    if np.all(reduced_costs <= 0): 
        return None 
    return np.argmax(reduced_costs) 
 
def determine_leaving_variable(tableau, enter_col): 
    ratios = [] 
    for i in range(tableau.shape[0] - 1): 
        if tableau[i, enter_col] > 0: 
            ratios.append(tableau[i, -1] / tableau[i, enter_col]) 
        else: 
            ratios.append(np.inf) 
     
    if all(r == np.inf for r in ratios): 
        return None 
     
    return np.argmin(ratios) 
 
def print_tableau(tableau, obj_coeffs, base_vars): 
    print("\nTableau du Simplexe:") 
    headers = [f"x{i+1}" for i in range(tableau.shape[1]-1)] + ["b"] 
    print("Base".rjust(10), end=" ") 
    for header in headers: 
        print(header.rjust(10), end=" ") 
    print("\n" + "-" * (10 * (len(headers) + 1))) 
     
    for i, row in enumerate(tableau): 
        if i == tableau.shape[0] - 1: 
            print("Z-row".rjust(10), end=" ") 
        else: 
            print(f"x{base_vars[i]+1}".rjust(10), end=" ") 
         
        for val in row: 
            print(f"{val:10.2f}", end=" ") 
        print() 
 
def main(): 
    print("=== Solveur de Programmation Lineaire - Methode du Simplexe , Jinx  ===") 
 
    obj_coeffs, constraints = get_user_input() 
 
    obj_coeffs = np.array(obj_coeffs) 
    constraints = np.array(constraints) 
    solution = simplex_solve(obj_coeffs, constraints) 
     
    if solution: 
        print("\n=== Resultat Optimal ===") 
        for i, val in enumerate(solution[0]): 
            print(f"x{i+1} = {val:.2f}") 
        print(f"Valeur optimale = {solution[1]:.2f}") 

if __name__ == "__main__":
    main()