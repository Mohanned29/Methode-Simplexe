import numpy as np
from constraint import Problem, AllDifferentConstraint

"""                       MADE BY JINX                           """


def get_user_input():
    """Collects user input for the objective function and constraints."""
    while True:
        try:
            num_vars = int(input("Nombre de variables (max 3) : "))
            if 1 <= num_vars <= 3:
                break
            else:
                print("Veuillez entrer un nombre entre 1 et 3.")
        except ValueError:
            print("Entrée invalide, entrez un nombre entier.")

    print("\nEntrez les coefficients de la fonction objectif (max z):")
    obj_coeffs = [] 
    for i in range(num_vars): 
        while True: 
            try: 
                coeff = float(input(f"Coefficient x{i+1} : ")) 
                obj_coeffs.append(coeff) 
                break 
            except ValueError: 
                print("Entrée invalide, entrez un nombre.") 

    print("\nEntrez les contraintes (forme ≤):")
    num_constraints = int(input("Combien de contraintes souhaitez-vous ajouter? : "))
    constraints = [] 
    for i in range(num_constraints): 
        constraint = [] 
        print(f"\nContrainte {i+1}:") 
        for j in range(num_vars): 
            while True: 
                try: 
                    coeff = float(input(f"Coefficient x{j+1} : ")) 
                    constraint.append(coeff) 
                    break 
                except ValueError: 
                    print("Entrée invalide, veuillez entrer un nombre.")
        while True: 
            try: 
                rhs = float(input("Terme constant (membre de droite) : ")) 
                constraint.append(rhs) 
                break 
            except ValueError: 
                print("Entrée invalide, entrez un nombre.") 
         
        constraints.append(constraint) 

    print("\nRésumé des entrées:")
    print("Fonction objectif : max z =", *obj_coeffs)
    print("Contraintes :")
    for i, constraint in enumerate(constraints):
        print(f"  {i+1}: {constraint[0]}x1 + {constraint[1]}x2 <= {constraint[2]}")

    return obj_coeffs, constraints, num_constraints 


def convert_to_canonical_form(obj_coeffs, constraints): 
    """Convertit la fonction et les contraintes en forme canonique."""
    num_vars = len(obj_coeffs) 
    slack_vars = np.eye(len(constraints))
    canonical_matrix = np.column_stack([ 
        [row[:num_vars] for row in constraints],  
        slack_vars,  
        [row[-1] for row in constraints]
    ]) 
    
    full_obj_coeffs = np.concatenate([ 
        obj_coeffs,  
        np.zeros(len(constraints))
    ]) 
    
    return canonical_matrix, full_obj_coeffs 


def constraint_programming_solve(obj_coeffs, constraints, num_vars):
    """Résolution par Programmation par Contraintes."""
    prob = Problem()

    variables = [f"x{i+1}" for i in range(num_vars)]
    for var in variables:
        prob.addVariable(var, range(101))  # Limiter les variables à des valeurs entre 0 et 100 (ou autre borne)

    # Ajouter les contraintes
    for i, constraint in enumerate(constraints):
        expr = []
        for j in range(num_vars):
            expr.append((f"x{j+1}", constraint[j]))  # Coefficients des variables
        prob.addConstraint(lambda *args: sum([args[i] * expr[i][1] for i in range(num_vars)]) <= expr[-1][-1], variables)

    def objective(*args):
        return sum(obj_coeffs[i] * args[i] for i in range(len(obj_coeffs)))

    prob.addConstraint(objective, variables)

    #trouver la meilleure solution
    solution = prob.getSolution()
    return solution 


def simplex_solve_with_cp(obj_coeffs, constraints, num_vars): 
    """Résolution par la méthode Simplexe et Programmation par Contraintes."""  
    tableau, obj_coeffs = convert_to_canonical_form(obj_coeffs, constraints) 
    
    num_vars = len(obj_coeffs) - len(constraints) 
    base_vars = list(range(num_vars, num_vars + len(constraints)))  #variables de base (slack variables)

    iteration = 0 
    while True: 
        print(f"\n--- Itération {iteration} ---")
        print_tableau(tableau, obj_coeffs, base_vars)
        
        #step 1: Calcul de C_i - Z_i
        Ci_Zi = obj_coeffs - tableau[-1, :-1]  
        print(f"\nC_i - Z_i : {Ci_Zi}")
        
        #if tous les C_i - Z_i sont <= 0, on a terminé
        if np.all(Ci_Zi <= 0): 
            break
        
        #step 2: Trouver la variable entrante (celle avec le plus grand C_i - Z_i)
        enter_col = np.argmax(Ci_Zi) 
        print(f"Variable entrante : x{enter_col + 1}")
        
        #step 3: Calcul de la variable sortante (test du rapport minimum)
        leave_row = determine_leaving_variable(tableau, enter_col)
        if leave_row is None: 
            print("Problème non borné.") 
            return None 
        
        # Pivotage
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


def determine_leaving_variable(tableau, enter_col): 
    """Détermine la variable sortante."""
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
    """Affiche le tableau du Simplexe"""
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
    print("=== Solveur de Programmation Linéaire - Méthode du Simplexe ===") 
    
    obj_coeffs, constraints, num_constraints = get_user_input() 
    
    obj_coeffs = np.array(obj_coeffs) 
    constraints = np.array(constraints) 
  
    solution = constraint_programming_solve(obj_coeffs, constraints, len(obj_coeffs)) 
    print("\n=== Résultat avec Programmation par Contraintes ===")
    for var, value in solution.items():
        print(f"{var} = {value:.2f}")
    print(f"Valeur optimale = {sum(obj_coeffs[i] * solution[f'x{i+1}'] for i in range(len(obj_coeffs))):.2f}")

if __name__ == "__main__":
    main()
