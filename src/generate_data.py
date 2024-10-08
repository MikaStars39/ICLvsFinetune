import random

def generate_sum_expressions(
    min_terms: int = 1,
    max_terms: int = 7,
    num_range: int = 10,
    need_total_sum: int = 0,
    minus: bool = True,
    ):
    terms = []
    sum_so_far = 0
    num_terms = random.randint(min_terms, max_terms)
    if need_total_sum is None:
        need_total_sum=random.randint(-num_range, num_range) if minus else random.randint(0, num_range)
    if num_terms == 1:
        expression = str(need_total_sum)
        sum_so_far = need_total_sum
    else:
        for i in range(num_terms - 1):
            number = random.randint(-num_range, num_range) if minus else random.randint(0, num_range)
            op = '+' if number >= 0 else "-"
            sum_so_far += number
            if i == 0 and op == '+':
                op = ''
            terms.append(f'{op}{abs(number)}')
        final_number = need_total_sum-sum_so_far
        final_op = '+' if final_number >= 0 else '-'
        terms.append(f'{final_op}{abs(final_number)}')
        expression = ''.join(terms)
    return expression, need_total_sum

def generate_boolean_expression(num_terms=3):
    operators = ['and', 'or']
    values = ['True', 'False']
    expression = []

    # Start with a random boolean value
    expression.append(random.choice(values))

    # Add operators and boolean values
    for _ in range(num_terms - 1):
        operator = random.choice(operators)
        value = random.choice(values)
        expression.append(operator)
        expression.append(value)

    # Join all parts to form the final expression
    expression_str = ' '.join(expression)
    return expression_str, eval(expression_str)

def generate_bool_expression(
    num_groups: int = 3,
    num_terms: int = 4,
    and_false: bool = False,
    or_true: bool = False,
    randoms: bool = False,
    need_false: bool = False,
):
    if and_false == False and or_true == False and randoms == False:
        choice = random.choice(['False', 'True'])
        if choice == "False":
            and_false = True
        else:
            or_true = True
    
    expression = []

    for _ in range(num_groups):
        # Determine the number of terms in this group
        num_terms = random.randint(2, num_terms)
        sub_expr, _ = generate_boolean_expression(num_terms)

        # Add parentheses around the sub-expression
        if len(expression) > 0:
            operator = random.choice(['and', 'or'])
            expression.append(operator)
        expression.append(f"({sub_expr})")

    # Join all parts to form the final expression
    expression_str = ' '.join(expression)

    if and_false:
        expression_str = "(" + expression_str + ")" + " and False"
    elif or_true:
        expression_str = expression_str + ' or True'
    
    if need_false:
        choice = random.choice(['False', 'True'])
        if choice == "False":
            expression_str = "(" + expression_str + ")" + " or False"
        else:
            expression_str = "(" + expression_str + ")" + " and True"

    return expression_str, eval(expression_str)

def generate_relation_problem(
    n: int = 10,
    need_false: bool = False,
):
    # Generate random cities excluding A and Z
    cities = [chr(i) for i in range(65, 65 + n) if chr(i) not in ['A', 'Z']]
    random.shuffle(cities)
    
    # Initialize connections and make sure A and Z are included
    connections = []
    a_connection = random.choice(cities)
    connections.append(('A', a_connection))
    
    # Randomly connect other cities
    for i in range(len(cities) - 1):
        connections.append((cities[i], cities[i + 1]))
    if need_false:
        connections.pop()
    # Randomly decide if A should be connected to Z through the point A is connected to
    if need_false:
        answer = False
    elif random.choice([True, False]):
        connections.append((a_connection, 'Z'))
        answer = True
    else:
        answer = False

    text = ""
    
    for each in connections:
        text += each[0] + " is connected with " + each[1] + "\n"
    
    return text, answer

import numpy as np

def generate_linear_equations(
    variables: int = 2, 
    equations: int = 3, 
    dependent: int = 1,
):
    # Ensure we have more equations than variables to avoid singular matrix errors in creation
    if equations < variables:
        raise ValueError("Number of equations must be at least equal to the number of variables")
    
    # Check if the number of dependent equations is less than the total number of equations
    if dependent >= equations:
        raise ValueError("Number of dependent equations must be less than the total number of equations")
    
    # Generate the coefficients and solutions for variables
    solutions = {chr(97 + i): random.randint(1, 5) for i in range(variables)}
    independent_equations = equations - dependent

    # Create a matrix of coefficients for independent equations
    coefficients = np.random.randint(1, 5, size=(independent_equations, variables))
    
    # Calculate the constants using the generated solutions
    constants = coefficients @ np.array([solutions[chr(97 + i)] for i in range(variables)])

    # Add dependent equations
    for _ in range(dependent):
        # Randomly select two existing equations to create a linear combination
        indices = random.sample(range(independent_equations), 2)
        alpha, beta = random.randint(1, 3), random.randint(1, 3)
        
        # Create a new dependent row as a linear combination of two selected rows
        new_row = alpha * coefficients[indices[0]] + beta * coefficients[indices[1]]
        new_constant = alpha * constants[indices[0]] + beta * constants[indices[1]]
        
        # Append the new row and new constant
        coefficients = np.vstack([coefficients, new_row])
        constants = np.append(constants, new_constant)

    # Format the equations as a string
    equation_strings = []
    for i in range(coefficients.shape[0]):
        terms = [f"{int(coefficients[i, j])}{chr(97 + j)}" for j in range(variables)]
        equation = " + ".join(terms) + f" = {int(constants[i])}"
        equation_strings.append(equation)
    
    equations_str = "\n".join(equation_strings)
    
    return equations_str, solutions

def build_expression(
    min_terms: int = 1,
    max_terms: int = 3,
    num_range: int = 10,
    need_false: bool = False,
):
    zero, __ = generate_sum_expressions(
        min_terms=min_terms,
        max_terms=max_terms,
        num_range=num_range,
        need_total_sum=0,
    )

    none_zero, result_zero = generate_sum_expressions(
        min_terms=min_terms,
        max_terms=max_terms,
        num_range=num_range,
        need_total_sum=None,
    )
    
    expression_cpl, result_cpl = generate_sum_expressions(
        min_terms=4,
        max_terms=4,
        num_range=10,
        need_total_sum=None,
    )

    expression_simple, result_simple = generate_sum_expressions(
        min_terms=1,
        max_terms=2,
        num_range=10,
        need_total_sum=None,
        minus=False,
    )
    
    if need_false:
        expression = "(" + expression_simple +")+"  + "(" + none_zero + ")*(" + expression_cpl + ")" + "="
        result = result_cpl * result_zero + result_simple
    else:
        expression = "(" + expression_simple +")+" + "(" + zero + ")*(" + expression_cpl + ")" + "="
        result = result_simple

    return expression, result

def build_code(inputs, range: int = 10):
    input_value = random.randint(0, range)

    if inputs["type"] == "+":
        result = input_value + int(inputs["number"])
    elif inputs["type"] == "-":
        result = input_value - int(inputs["number"])
    elif inputs["type"] == "*":
        result = input_value * int(inputs["number"])
    elif inputs["type"] == "/":
        result = input_value / int(inputs["number"])
    else:
        print(inputs["type"])
        raise ValueError("Not a valid type")
    
    return inputs["code"], input_value, result