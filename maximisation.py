import numpy as np
from constraint import Problem


"""                           DONE BY JINX                            """


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


def constraint_programming_solve(obj_coeffs, constraints, num_vars):
    """Résolution par Programmation par Contraintes."""
    prob = Problem()

    variables = [f"x{i+1}" for i in range(num_vars)]
    for var in variables:
        prob.addVariable(var, range(0, 101))

    for i, constraint in enumerate(constraints):
        def constraint_func(*args):
            return sum(args[j] * constraint[j] for j in range(num_vars)) <= constraint[-1]
        
        prob.addConstraint(constraint_func, variables)

    solution = prob.getSolution()

    return solution


def main():
    print("=== Solveur de Programmation Linéaire - Méthode du Simplexe ===")
    
    obj_coeffs, constraints, num_constraints = get_user_input()
    
    obj_coeffs = np.array(obj_coeffs)
    constraints = np.array(constraints)

    solution = constraint_programming_solve(obj_coeffs, constraints, len(obj_coeffs))

    if solution:
        print("\n=== Résultat avec Programmation par Contraintes ===")
        for var, value in solution.items():
            print(f"{var} = {value:.2f}")
        optimal_value = sum(obj_coeffs[i] * solution[f"x{i+1}"] for i in range(len(obj_coeffs)))
        print(f"Valeur optimale = {optimal_value:.2f}")
    else:
        print("Aucune solution trouvée.")


if __name__ == "__main__":
    main()
