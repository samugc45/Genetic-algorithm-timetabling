import numpy as np

# Ejemplo de dataset de entrada para el problema de asignación de horarios
dataset = {"n_courses" : 3,
           "n_days" : 3,
           "n_hours_day" : 3,
           "courses" : [("IA", 1), ("ALG", 2), ("BD", 3)]}


def generate_random_array_int(alphabet, length):
    # Genera un array de enteros aleatorios de tamaño length
    indices = np.random.randint(0, len(alphabet), length)
    return np.array(alphabet)[indices]

def generate_initial_population_timetabling(pop_size, *args, **kwargs):
    dataset = kwargs['dataset'] # Dataset con la misma estructura que el ejemplo

    courses = dataset['courses']
    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']

    alphabet = list(range(n_days * n_hours_day))  # Rango de valores posibles
    length = sum(hours for _, hours in courses)  # Total de horas a planificar


    # Población Inicial
    population = [generate_random_array_int(alphabet, length) for _ in range(pop_size)]

    return population

################################# NO TOCAR #################################
#                                                                          #
def print_timetabling_solution(solution, dataset):
    # Imprime una solución de timetabling
    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']
    courses = dataset['courses']

    # Crea una matriz de n_days x n_hours_day
    timetable = [[[] for _ in range(n_hours_day)] for _ in range(n_days)]

    # Llena la matriz con las asignaturas
    i = 0
    max_len = 6 # Longitud del título Día XX
    for course in courses:
        for _ in range(course[1]):
            day = solution[i] // n_hours_day
            hour = solution[i] % n_hours_day
            timetable[day][hour].append(course[0])
            i += 1
            # Calcula la longitud máxima del nombre de las asignaturas
            # en una misma franja horaria
            max_len = max(max_len, len('/'.join(timetable[day][hour])))

    # Imprime la matriz con formato de tabla markdown
    print('|         |', end='')
    for i in range(n_days):
        print(f' Día {i+1:<2}{" "*(max_len-6)} |', end='')
    print()
    print('|---------|', end='')
    for i in range(n_days):
        print(f'-{"-"*max_len}-|', end='')
    print()
    for j in range(n_hours_day):
        print(f'| Hora {j+1:<2} |', end='')
        for i in range(n_days):
            s = '/'.join(timetable[i][j])
            print(f' {s}{" "*(max_len-len(s))}', end=' |')
        print()
#                                                                          #
################################# NO TOCAR #################################

# Ejemplo de uso de la función generar individuo con el dataset de ejemplo
candidate = generate_random_array_int(list(range(9)), 6)
print_timetabling_solution(candidate, dataset = dataset)

def get_timetabling_table(solution, dataset):
    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']
    courses = dataset['courses']

    # Creamos una tabla vacía
    timetable = [[[] for _ in range(n_hours_day)] for _ in range(n_days)]

    # Asignamos las asignaturas a los horarios
    i = 0
    for course in courses:
        for _ in range(course[1]):
            day = solution[i] // n_hours_day
            hour = solution[i] % n_hours_day
            timetable[day][hour].append(course[0])  # Nombre
            i += 1

    return timetable

def get_course_hours_per_day(solution, dataset):
    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']
    courses = dataset['courses']

    # Inicializamos un diccionario para almacenar las horas por día para cada asignatura
    course_hours = {course[0]: [0] * n_days for course in courses}  # Inicializamos con 0 horas para cada día

    # Se itera sobre la solución
    i = 0
    for course in courses:
        course_name = course[0]
        for _ in range(course[1]):  # Para cada vez que esta asignatura se debe asignar
            day = solution[i] // n_hours_day
            course_hours[course_name][day] += 1  # Aumentamos la hora en el día correspondiente
            i += 1

    return course_hours



def calculate_c1(solution, *args, **kwargs):
    dataset = kwargs['dataset']

    # Obtenemos la tabla de horarios
    timetable = get_timetabling_table(solution, dataset)

    penalty = 0
    # Recorremos cada día y hora
    for day in range(dataset['n_days']):
        for hour in range(dataset['n_hours_day']):
            # Si hay más de una asignatura en una misma franja horaria, sumamos la penalización
            if len(timetable[day][hour]) > 1:
                penalty += len(timetable[day][hour]) - 1  # Penaliza las repeticiones

    return penalty

print(print_timetabling_solution(candidate, dataset = dataset))



def calculate_c2(solution, *args, **kwargs):
    dataset = kwargs['dataset']

    # Obtenemos las horas por día de cada asignatura
    course_hours = get_course_hours_per_day(solution, dataset)

    penalty = 0
    for course_name, hours in course_hours.items():
        for day_hours in hours:
            if day_hours > 2:
                penalty += day_hours - 2  # Penaliza las horas adicionales

    return penalty


def calculate_p1(solution, *args, **kwargs):
    dataset = kwargs['dataset']

    # Obtenemos la tabla de horarios a partir de la solución
    timetable = get_timetabling_table(solution, dataset)

    huecos = 0

    # Iteramos sobre cada día del horario
    for day in timetable:
        # Lista de índices de horas donde hay clases en el día actual
        class_hours = [i for i, hour in enumerate(day) if len(hour) > 0]

        # Si hay al menos dos clases, podemos tener gaps
        if len(class_hours) > 1:
            for j in range(len(class_hours) - 1):
                # Calculamos los huecos entre clases consecutivas
                huecos += (class_hours[j + 1] - class_hours[j] - 1)

    return huecos



def calculate_p2(solution, *args, **kwargs):
    dataset = kwargs['dataset']

    # Obtenemos la tabla de horarios
    timetable = get_timetabling_table(solution, dataset)

    days_used = 0
    for day in range(dataset['n_days']):
        # Si hay al menos una asignatura en el día, contamos ese día como usado
        if any(timetable[day]):
            days_used += 1

    return days_used


def calculate_p3(solution, *args, **kwargs):
    dataset = kwargs['dataset']

    # Obtenemos la tabla de horarios
    timetable = get_timetabling_table(solution, dataset)

    non_consecutive = 0
    for day in range(dataset['n_days']):
        # Creamos un diccionario para contar las horas de cada asignatura en ese día
        course_slots = {course[0]: [] for course in dataset['courses']}

        # Recorremos la tabla y llenamos el diccionario
        for hour in range(dataset['n_hours_day']):
            for course in timetable[day][hour]:
                course_slots[course].append(hour)

        # Verificamos si alguna asignatura tiene horas no consecutivas
        for slots in course_slots.values():
            if len(slots) > 1:  # Si tiene más de una hora asignada
                slots.sort()
                # Si las horas no son consecutivas, agregamos penalización
                for i in range(1, len(slots)):
                    if slots[i] != slots[i-1] + 1:
                        non_consecutive += 1

    return non_consecutive

print("c1: ", calculate_c1(candidate, dataset=dataset))
print("c2: ", calculate_c2(candidate, dataset=dataset))
print("p1: ", calculate_p1(candidate, dataset=dataset))
print("p2: ", calculate_p2(candidate, dataset=dataset))
print("p3: ", calculate_p3(candidate, dataset=dataset))


def fitness_timetabling(solution, *args, **kwargs):

    dataset = kwargs['dataset']

    # Obtenemos las tablas y las horas por día
    schedule_table = get_timetabling_table(solution, dataset)
    hours_per_day = get_course_hours_per_day(solution, dataset)

    c1 = calculate_c1(solution, dataset = dataset)
    c2 = calculate_c2(solution, dataset = dataset)
    p1 = calculate_p1(solution, dataset = dataset)
    p2 = calculate_p2(solution, dataset = dataset)
    p3 = calculate_p3(solution, dataset = dataset)

    # Fitness según la fórmula del enunciado
    if c1 > 0 or c2 > 0:
        return 0
    return 1 / (1 + p1 + p2 + p3)

fitness_timetabling(candidate, dataset=dataset) # Devuelve la fitness del candidato de ejemplo

def tournament_selection(population, fitness, number_parents, *args, **kwargs):

    t = kwargs['tournament_size']

    seleccion_padres = []

    # Selección de padres
    for _ in range(number_parents):
        # Selección por torneo para un padre
        parent = seleccion_padre_torneo(population, fitness, t)
        seleccion_padres.append(parent)

    return seleccion_padres

def seleccion_padre_torneo(population, fitness, t):

    tournament_indices = np.random.choice(len(population), t, replace=False)

    # Encuentra el índice del mejor individuo (máximo fitness)
    mejor_indice = tournament_indices[np.argmax([fitness[i] for i in tournament_indices])]

    # Retorna el individuo correspondiente al índice del mejor fitness
    return population[mejor_indice]

def one_point_crossover(parent1, parent2, p_cross, *args, **kwargs):

    # Verifica si el cruce debe realizarse
    if np.random.rand() > p_cross:
        # print("No se realizó cruce")
        return parent1.copy(), parent2.copy()

    # Se elige el punto de cruce
    crossover_point = np.random.randint(1, len(parent1))  # El punto de cruce está entre 1 y el tamaño del padre

    # Generar los hijos a partir del cruce
    hijo1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    hijo2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

    # print("Se realizó el cruce en el punto:", crossover_point)
    return hijo1, hijo2

def uniform_mutation(chromosome, p_mut, *args, **kwargs):

    dataset = kwargs['dataset']
    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']

    # Tamaño del espacio de solución (total de franjas disponibles)
    max_slots = n_days * n_hours_day

    # Convertir el cromosoma en un array NumPy para aprovechar sus funciones
    mutated_chromosome = np.array(chromosome)

    # Generar una máscara de mutaciones con probabilidad p_mut
    mutation_mask = np.random.rand(len(chromosome)) < p_mut

    # Generar nuevos valores aleatorios para las posiciones seleccionadas
    new_values = np.random.randint(0, max_slots, size=len(chromosome))

    # Aplicar las mutaciones según la máscara
    mutated_chromosome[mutation_mask] = new_values[mutation_mask]

    return mutated_chromosome.tolist()

def generational_replacement(population, fitness, offspring, fitness_offspring, *args, **kwargs):

    auxpopulation = population.copy()
    auxfitness = fitness.copy()
    for i in range(len(offspring)):
        auxpopulation.pop(0)
        auxpopulation.append(offspring.pop(0))
        auxfitness.pop(0)
        auxfitness.append(fitness_offspring.pop(0))
    return auxpopulation, auxfitness

def generation_stop(generation, fitness, *args, **kwargs):
    max_gen = kwargs['max_gen']

    # Si la generación actual es mayor o igual que el número máximo de generaciones
    return generation >= max_gen

def genetic_algorithm(generate_population, pop_size, fitness_function, stopping_criteria, offspring_size,
                      selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs):
    # Generar la población inicial utilizando la función `generate_population` con el tamaño `pop_size`
    poblacion = generate_population(pop_size, *args, **kwargs)

    # Evaluar el fitness de cada individuo en la población inicial
    fitness = [fitness_function(ind, *args, **kwargs) for ind in poblacion]

    # Inicializar listas para guardar el mejor fitness y el fitness promedio por generación
    best_fitness = [max(fitness)]
    mean_fitness = [sum(fitness) / len(fitness)]

    # Contador para las generaciones
    generation = 1

    # Comenzar el ciclo evolutivo mientras no se cumpla el criterio de parada
    while not stopping_criteria(generation, fitness, *args, **kwargs):

        AUX1, AUX2 = poblacion.copy(), fitness.copy()

        # Hacemos la creación de descendientes
        for _ in range(offspring_size // 2):  # Generar pares de descendientes
            # Seleccionar dos padres utilizando el método `selection`
            parents = selection(poblacion, fitness, 2, *args, **kwargs)

            # Aplicar cruce (crossover) a los padres con probabilidad `p_cross`
            offspring1, offspring2 = crossover(parents[0], parents[1], p_cross, *args, **kwargs)

            # Aplicar mutación (mutation) a los descendientes con probabilidad `p_mut`
            offspring1 = mutation(offspring1, p_mut, *args, **kwargs)
            offspring2 = mutation(offspring2, p_mut, *args, **kwargs)

            # Evaluar el fitness de los descendientes
            offspring_fitness = [
                fitness_function(offspring1, *args, **kwargs),
                fitness_function(offspring2, *args, **kwargs)
            ]

            AUX1, AUX2 = environmental_selection(AUX1, AUX2, [offspring1, offspring2], offspring_fitness, *args, **kwargs)


        # Reemplazar los valores por las variables AUX
        population, fitness = AUX1.copy(), AUX2.copy()

        # Registrar el mejor fitness y el fitness promedio de la generación actual
        best_fitness.append(max(fitness))
        mean_fitness.append(sum(fitness) / len(fitness))

        generation += 1

    return population, fitness, generation, best_fitness, mean_fitness

### Coloca aquí tus funciones de fitness propuestas ###

def fitness_timetabling2(solution, *args, **kwargs):

    dataset = kwargs['dataset']

    # Obtenemos las tablas y las horas por día
    schedule_table = get_timetabling_table(solution, dataset)
    hours_per_day = get_course_hours_per_day(solution, dataset)

    c1 = calculate_c1(solution, dataset = dataset)
    c2 = calculate_c2(solution, dataset = dataset)
    p1 = calculate_p1(solution, dataset = dataset)
    p2 = calculate_p2(solution, dataset = dataset)
    p3 = calculate_p3(solution, dataset = dataset)

    k = 2 #Aquí k lo podemos ajustar según el tamaño y complejidad del problema

    fitness = 1 / (1 + k * (c1 + c2) + p1 + p2 + p3)

    return fitness

def rank_selection(population, fitness, number_parents, **kwargs):

    sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i]) # Fitness de menor a mayor

    sorted_population = [population[i] for i in sorted_indices]

    ranks = np.arange(1, len(fitness) + 1) # Rangos de 1 al tamaño de la población
    total = np.sum(ranks)
    probabilities = ranks / total # Calcula las probabilidades

    indices = np.random.choice(len(sorted_population), size=number_parents, p=probabilities, replace=True)
    selected_parents = [sorted_population[i] for i in indices]

    return selected_parents

#Hemos dejado la misma
def one_point_crossover2(parent1, parent2, p_cross, *args, **kwargs):

    if np.random.rand() > p_cross:
        return parent1.copy(), parent2.copy()

    # Generar dos puntos de cruce aleatorios
    point1 = np.random.randint(1, len(parent1) - 1)
    point2 = np.random.randint(point1, len(parent1))

    # Intercambiar segmentos entre los padres
    hijo1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
    hijo2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))

    return hijo1, hijo2

def uniform_mutation2(chromosome, p_mut, *args, **kwargs):
    dataset = kwargs['dataset']
    sol = chromosome.copy()
    m = dataset['n_days']
    k = dataset['n_hours_day']
    for i in range(0,len(sol)):
        if(np.random.random() < p_mut):
            sol[i] = np.random.randint(0,m*k)
    return sol

def generational_replacement2(population, fitness, offspring, fitness_offspring, *args, **kwargs):
    auxpopulation = population.copy()
    auxfitness = fitness.copy()
    for i in range(len(offspring)):
        auxpopulation.pop(0)
        auxpopulation.append(offspring.pop(0))
        auxfitness.pop(0)
        auxfitness.append(fitness_offspring.pop(0))
    return auxpopulation, auxfitness

## En esta no no hemos realizado cambios para la aproximación propuesta ##

def generation_stop2(generation, fitness, *args, **kwargs):
    max_gen = kwargs['max_gen']

    # Si la generación actual es mayor o igual que el número máximo de generaciones
    return generation >= max_gen

################################# NO TOCAR #################################
#                                                                          #
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        return *res, end - start
    return wrapper
#                                                                          #
################################# NO TOCAR #################################

# Este codigo temporiza la ejecución de una función cualquiera

################################# NO TOCAR #################################
#                                                                          #
@timer
def run_ga(generate_population, pop_size, fitness_function, stopping_criteria, offspring_size,
           selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs):
    # Además del retorno de la función, se devuelve el tiempo de ejecución en segundos
    return genetic_algorithm(generate_population, pop_size, fitness_function, stopping_criteria, offspring_size,
                             selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs)
#                                                                          #
################################# NO TOCAR #################################

# Se deben probar los 6 datasets
dataset1 = {"n_courses" : 3,
            "n_days" : 3,
            "n_hours_day" : 3,
            "courses" : [("IA", 1), ("ALG", 2), ("BD", 3)]}

dataset2 = {"n_courses" : 4,
            "n_days" : 3,
            "n_hours_day" : 4,
            "courses" : [("IA", 1), ("ALG", 2), ("BD", 3), ("POO", 2)]}

dataset3 = {"n_courses" : 4,
            "n_days" : 4,
            "n_hours_day" : 4,
            "courses" : [("IA", 2), ("ALG", 4), ("BD", 6), ("POO", 4)]}

dataset4 = {"n_courses" : 5,
            "n_days" : 4,
            "n_hours_day" : 6,
            "courses" : [("IA", 2), ("ALG", 4), ("BD", 6), ("POO", 4), ("AC", 4)]}

dataset5 = {"n_courses" : 7,
            "n_days" : 4,
            "n_hours_day" : 8,
            "courses" : [("IA", 2), ("ALG", 4), ("BD", 6), ("POO", 4), ("AC", 4), ("FP", 4), ("TP", 2)]}

dataset6 = {"n_courses" : 11,
            "n_days" : 5,
            "n_hours_day" : 12,
            "courses" : [("IA", 2), ("ALG", 4), ("BD", 6), ("POO", 4), ("AC", 4), ("FP", 4), ("TP", 2), ("FC", 4), ("TSO", 2), ("AM", 4), ("LMD", 4)]}

import numpy as np
import random

def set_seed(seed):
    # Se debe fijar la semilla usada para generar números aleatorios
    # Con la librería random
    random.seed(seed)
    # Con la librería numpy
    np.random.seed(seed)

################################# NO TOCAR #################################
#                                                                          #
def best_solution(population, fitness):
    # Devuelve la mejor solución de la población
    return population[fitness.index(max(fitness))]

import matplotlib.pyplot as plt
def plot_fitness_evolution(best_fitness, mean_fitness):
    plt.plot(best_fitness, label='Best fitness')
    plt.plot(mean_fitness, label='Mean fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()
#                                                                          #
################################# NO TOCAR #################################

from statistics import mean, median, stdev

def launch_experiment(seeds, dataset, generate_population, pop_size, fitness_function, c1, c2, p1, p2, p3, stopping_criteria,
                      offspring_size, selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs):
    best_individuals = []
    best_inds_c1 = []
    best_inds_c2 = []
    best_inds_p1 = []
    best_inds_p2 = []
    best_inds_p3 = []
    best_inds_fitness = []
    best_fitnesses = []
    mean_fitnesses = []
    last_generations = []
    execution_times = []
    # Ejecutamos el algoritmo con cada semilla
    for seed in seeds:
        print(f"Running Genetic Algorithm with seed {seed}")
        set_seed(seed)
        population, fitness, generation, best_fitness, mean_fitness, execution_time = run_ga(generate_population, pop_size, fitness_function,stopping_criteria,
                                                                                             offspring_size, selection, crossover, p_cross, mutation, p_mut,
                                                                                             environmental_selection, dataset=dataset, *args, **kwargs)
        best_individual = best_solution(population, fitness)
        best_ind_c1 = c1(best_individual, dataset=dataset)
        best_ind_c2 = c2(best_individual, dataset=dataset)
        best_ind_p1 = p1(best_individual, dataset=dataset)
        best_ind_p2 = p2(best_individual, dataset=dataset)
        best_ind_p3 = p3(best_individual, dataset=dataset)
        best_ind_fitness = fitness_function(best_individual, dataset=dataset)
        best_individuals.append(best_individual)
        best_inds_c1.append(best_ind_c1)
        best_inds_c2.append(best_ind_c2)
        best_inds_p1.append(best_ind_p1)
        best_inds_p2.append(best_ind_p2)
        best_inds_p3.append(best_ind_p3)
        best_inds_fitness.append(best_ind_fitness)
        best_fitnesses.append(best_fitness)
        mean_fitnesses.append(mean_fitness)
        last_generations.append(generation)
        execution_times.append(execution_time)
    # Imprimimos la media y desviación típica de los resultados obtenidos
    print("Mean Best Fitness: " + str(mean(best_inds_fitness)) + " " + u"\u00B1" + " " + str(stdev(best_inds_fitness)))
    print("Mean C1: " + str(mean(best_inds_c1)) + " " + u"\u00B1" + " " + str(stdev(best_inds_c1)))
    print("Mean C2: " + str(mean(best_inds_c2)) + " " + u"\u00B1" + " " + str(stdev(best_inds_c2)))
    print("Mean P1: " + str(mean(best_inds_p1)) + " " + u"\u00B1" + " " + str(stdev(best_inds_p1)))
    print("Mean P2: " + str(mean(best_inds_p2)) + " " + u"\u00B1" + " " + str(stdev(best_inds_p2)))
    print("Mean P3: " + str(mean(best_inds_p3)) + " " + u"\u00B1" + " " + str(stdev(best_inds_p3)))
    print("Mean Execution Time: " + str(mean(execution_times)) + " " + u"\u00B1" + " " + str(stdev(execution_times)))
    print("Mean Number of Generations: " + str(mean(last_generations)) + " " + u"\u00B1" + " " + str(stdev(last_generations)))
    # Mostramos la evolución de la fitness para la mejor ejecución
    print("Best execution fitness evolution:")
    best_execution = best_inds_fitness.index(max(best_inds_fitness))
    plot_fitness_evolution(best_fitnesses[best_execution], mean_fitnesses[best_execution])
    # Mostramos la evolución de la fitness para la ejecución mediana
    print("Median execution fitness evolution:")
    median_execution = best_inds_fitness.index(median(best_inds_fitness))
    plot_fitness_evolution(best_fitnesses[median_execution], mean_fitnesses[median_execution])
    # Mostramos la evolución de la fitness para la peor ejecución
    print("Worst execution fitness evolution:")
    worst_execution = best_inds_fitness.index(min(best_inds_fitness))
    plot_fitness_evolution(best_fitnesses[worst_execution], mean_fitnesses[worst_execution])

    return best_individuals, best_inds_fitness, best_fitnesses, mean_fitnesses, last_generations, execution_times

# Crear un conjunto de 31 semillas para los experimentos
seeds = [1234567890 + i*13 for i in range(31)] # Semillas de ejemplo, cambiar por las semillas que se quieran
bestIndividualsAux,bestFitnessIndAux,_,_,_,_ = launch_experiment(seeds, dataset1, generate_initial_population_timetabling, 50, fitness_timetabling2, calculate_c1, calculate_c2,
                  calculate_p1, calculate_p2, calculate_p3, generation_stop2, 50, rank_selection, one_point_crossover2, 0.8,
                  uniform_mutation2, 0.1, generational_replacement2, max_gen=120);

print("Mejor individuo del mejor: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(max(bestFitnessIndAux))], dataset1))
print("\nMejor individuo del mediano: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(median(bestFitnessIndAux))], dataset1))
print("\nMejor individuo del peor: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(min(bestFitnessIndAux))], dataset1))

bestIndividualsAux,bestFitnessIndAux,_,_,_,_ = launch_experiment(seeds, dataset2, generate_initial_population_timetabling, 50, fitness_timetabling2, calculate_c1, calculate_c2,
                  calculate_p1, calculate_p2, calculate_p3, generation_stop2, 50, rank_selection, one_point_crossover2, 0.8,
                  uniform_mutation2, 0.1, generational_replacement2, max_gen=80)

print("Mejor individuo del mejor: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(max(bestFitnessIndAux))], dataset2))
print("\nMejor individuo del mediano: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(median(bestFitnessIndAux))], dataset2))
print("\nMejor individuo del peor: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(min(bestFitnessIndAux))], dataset2))

bestIndividualsAux,bestFitnessIndAux,_,_,_,_ = launch_experiment(seeds, dataset3, generate_initial_population_timetabling, 50, fitness_timetabling2, calculate_c1, calculate_c2,
                  calculate_p1, calculate_p2, calculate_p3, generation_stop2, 50, rank_selection, one_point_crossover2, 0.8,
                  uniform_mutation2, 0.1, generational_replacement2, max_gen=140)

print("Mejor individuo del mejor: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(max(bestFitnessIndAux))], dataset3))
print("\nMejor individuo del mediano: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(median(bestFitnessIndAux))], dataset3))
print("\nMejor individuo del peor: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(min(bestFitnessIndAux))], dataset3))

bestIndividualsAux,bestFitnessIndAux,_,_,_,_ = launch_experiment(seeds, dataset4, generate_initial_population_timetabling, 50, fitness_timetabling2, calculate_c1, calculate_c2,
                  calculate_p1, calculate_p2, calculate_p3, generation_stop2, 50, rank_selection, one_point_crossover2, 0.8,
                  uniform_mutation2, 0.1, generational_replacement2, max_gen=25)

print("Mejor individuo del mejor: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(max(bestFitnessIndAux))], dataset4))
print("\nMejor individuo del mediano: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(median(bestFitnessIndAux))], dataset4))
print("\nMejor individuo del peor: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(min(bestFitnessIndAux))], dataset4))

bestIndividualsAux,bestFitnessIndAux,_,_,_,_ = launch_experiment(seeds, dataset5, generate_initial_population_timetabling, 50, fitness_timetabling2, calculate_c1, calculate_c2,
                  calculate_p1, calculate_p2, calculate_p3, generation_stop2, 50, rank_selection, one_point_crossover2, 0.8,
                  uniform_mutation2, 0.1, generational_replacement2, max_gen=10)

print("Mejor individuo del mejor: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(max(bestFitnessIndAux))], dataset5))
print("\nMejor individuo del mediano: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(median(bestFitnessIndAux))], dataset5))
print("\nMejor individuo del peor: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(min(bestFitnessIndAux))], dataset5))

bestIndividualsAux,bestFitnessIndAux,_,_,_,_ = launch_experiment(seeds, dataset6, generate_initial_population_timetabling, 50, fitness_timetabling2, calculate_c1, calculate_c2,
                  calculate_p1, calculate_p2, calculate_p3, generation_stop2, 50, rank_selection, one_point_crossover2, 0.8,
                  uniform_mutation2, 0.1, generational_replacement2, max_gen=50)

print("Mejor individuo del mejor: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(max(bestFitnessIndAux))], dataset6))
print("\nMejor individuo del mediano: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(median(bestFitnessIndAux))], dataset6))
print("\nMejor individuo del peor: \n")
print(print_timetabling_solution(bestIndividualsAux[bestFitnessIndAux.index(min(bestFitnessIndAux))], dataset6))


# Recuerda también mostrar el horario de la mejor solución obtenida en los casos peor, mejor y mediano