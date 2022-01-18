import sys
from statistics import stdev
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog

from genetic_algorithm import geneticAlgorithm, City


class Experiment:
    def __init__(self, best_route, best_fitness, average_fitness, best_routes):
        self.best_route = best_route
        self.best_fitness = best_fitness
        self.average_fitness = average_fitness
        self.best_routes = best_routes


class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('tsp.ui', self)
        self.show()
        self.city_list = []
        self.file_label.setText("")

        self.elitist_checkbox.clicked.connect(lambda: self.enable_on_check(self.elitist_checkbox, self.elite_spinbox))
        self.hillclimbing_checkbox.clicked.connect(
            lambda: self.enable_on_check(self.hillclimbing_checkbox, self.opt_2_radio))
        self.hillclimbing_checkbox.clicked.connect(
            lambda: self.enable_on_check(self.hillclimbing_checkbox, self.opt_3_radio))
        self.hillclimbing_checkbox.clicked.connect(
            lambda: self.enable_on_check(self.hillclimbing_checkbox, self.generation_start_hc))
        self.clock_seed_check.clicked.connect(lambda: self.enable_on_check(self.clock_seed_check, self.seed_value))

        self.load_button.clicked.connect(self.open_file)
        self.run_button.clicked.connect(self.run_tsp)

    def enable_on_check(self, check, to_be_enabled):
        if check.isChecked():
            to_be_enabled.setEnabled(True)
        else:
            to_be_enabled.setEnabled(False)

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Wybierz plik z miastami", "",
                                                       "Miasta (*.tsp)",
                                                       options=options)
        if self.fileName:
            self.tsp_name = self.fileName[self.fileName.rfind("/") + 1:]
            self.file_label.setText(self.tsp_name)
            self.read_file()
            self.read_cities()
            self.city_graph()

    def city_graph(self):
        x = []
        y = []
        for city in self.city_list:
            x.append(city.x)
            y.append(city.y)

        self.tsp_graph_widget.canvas.axes.clear()
        self.tsp_graph_widget.canvas.axes.scatter(x, y)
        self.tsp_graph_widget.canvas.draw()

    def fitness_graph(self, best_fitness, average_fitness):
        self.data_graph_widget.canvas.axes.clear()
        self.data_graph_widget.canvas.axes.set_ylabel('Tour Length')
        self.data_graph_widget.canvas.axes.set_xlabel('Generation')
        self.data_graph_widget.canvas.axes.plot(best_fitness, label='Best in generation', marker='o')
        self.data_graph_widget.canvas.axes.plot(average_fitness, label='Average in generation', marker='s')
        self.data_graph_widget.canvas.axes.legend(loc='upper right')
        self.data_graph_widget.canvas.draw()

    def route_graph(self, city_route):
        x = []
        y = []
        for city in city_route:
            x.append(city.x)
            y.append(city.y)

        x.append(x[0])
        y.append(y[0])

        self.tsp_graph_widget.canvas.axes.clear()
        self.tsp_graph_widget.canvas.axes.scatter(x, y)
        self.tsp_graph_widget.canvas.axes.plot(x, y)
        self.tsp_graph_widget.canvas.draw()

    def read_values(self):
        self.popSize = self.population_size.value()
        self.generations = self.number_of_generations.value()
        self.mutationRate = self.mutation_probability.value()

        if self.elitist_checkbox.isChecked():
            self.elite = self.elite_spinbox.value()
            if self.elite > self.popSize:
                return "Elite value is bigger than population"
        else:
            self.elite = 0

        if self.hillclimbing_checkbox.isChecked():
            if self.opt_2_radio.isChecked():
                self.hillclimb_type = "2-opt"
            elif self.opt_3_radio.isChecked():
                self.hillclimb_type = "3-opt"
            self.hillclimb_gen_start = self.generation_start_hc.value()
            if self.hillclimb_gen_start > self.generations:
                return "Hillclimb starting generation is bigger then number of generations"
        else:
            self.hillclimb_gen_start = 0
            self.hillclimb_type = "OFF"

        if self.tournament_radio.isChecked():
            self.selection_type = "Tournament"
            self.selection_size = self.tournament_size.value()
            if self.selection_size > self.popSize:
                return "Tournament size is bigger than population"
        elif self.my_selection_radio.isChecked():
            self.selection_type = "Rank"
            self.selection_size = 0

        if self.ox_radio.isChecked():
            self.crossover_type = "OX"
        elif self.cx_radio.isChecked():
            self.crossover_type = "CX"
        elif self.my_crossover_radio.isChecked():
            self.crossover_type = "SCX"
        self.crossover_prob = self.crossover_probability.value()

        if self.swap_2_radio.isChecked():
            self.mutation_type = "2-swap"
        elif self.inversion_radio.isChecked():
            self.mutation_type = "Inversion"
        elif self.my_mutation_radio.isChecked():
            self.mutation_type = "last_but_not_least"
        self.mutation_prob = self.mutation_probability.value()

        if self.clock_seed_check.isChecked():
            self.seed = self.seed_value.value()
        else:
            self.seed = "none"

        self.num_of_exp = self.number_of_experiments.value()

        return True

    def read_file(self):
        file = open(self.fileName, "r")
        self.file_cities = file.readlines()
        self.run_button.setEnabled(True)

    def read_cities(self):
        dimension = 0
        for line in self.file_cities:
            if 'DIMENSION' in line:
                dimension = int(line[line.rfind(" "):])

        if self.file_cities[-1] == "EOF\n" or self.file_cities[-1] == "\n" or self.file_cities[-1] == "EOF":
            self.file_cities.pop()
            if self.file_cities[-1] == "EOF\n" or self.file_cities[-1] == "EOF":
                self.file_cities.pop()

        self.file_cities = self.file_cities[-dimension:]

        self.city_list = []

        for city in self.file_cities:
            city = city.split(" ")
            city = list(filter("".__ne__, city))
            x = int(float(city[1]))
            y = int(float(city[2]))
            self.city_list.append(City(x, y))

    def run_tsp(self):
        self.error_label.setText("")
        answer = self.read_values()

        if answer == True:
            if self.num_of_exp == 1:
                city_route, best_fitness, average_fitness, routes = geneticAlgorithm(population=self.city_list,
                                                                                     popSize=self.popSize,
                                                                                     generations=self.generations,
                                                                                     hillclimb_type=self.hillclimb_type,
                                                                                     hillclimb_generation=self.hillclimb_gen_start,
                                                                                     eliteSize=self.elite,
                                                                                     selection_type=self.selection_type,
                                                                                     selection_size=self.selection_size,
                                                                                     crossover_type=self.crossover_type,
                                                                                     crossover_prob=self.crossover_prob,
                                                                                     mutation_type=self.mutation_type,
                                                                                     mutation_prob=self.mutation_prob,
                                                                                     seed=self.seed)

                self.route_graph(city_route)

                self.fitness_graph(best_fitness, average_fitness)

                self.generation_number_graph.setText(f"{self.generations}")
                self.best_tour_graph.setText(f"{best_fitness[self.generations - 1]}")

                self.results(city_route, best_fitness, average_fitness, routes)
            else:
                experiments = []
                for i in range(self.num_of_exp):
                    city_route, best_fitness, average_fitness, routes = geneticAlgorithm(population=self.city_list,
                                                                                         popSize=self.popSize,
                                                                                         generations=self.generations,
                                                                                         hillclimb_type=self.hillclimb_type,
                                                                                         hillclimb_generation=self.hillclimb_gen_start,
                                                                                         eliteSize=self.elite,
                                                                                         selection_type=self.selection_type,
                                                                                         selection_size=self.selection_size,
                                                                                         crossover_type=self.crossover_type,
                                                                                         crossover_prob=self.crossover_prob,
                                                                                         mutation_type=self.mutation_type,
                                                                                         mutation_prob=self.mutation_prob,
                                                                                         seed=self.seed)
                    experiments.append(Experiment(city_route, best_fitness, average_fitness, routes))

                self.m_results(experiments)
                self.m_std_results(experiments)
                self.m_found_solutions(experiments)
        else:
            self.error_label.setStyleSheet('color: red')
            self.error_label.setText(answer)

    def tsp_city_sequence(self, city_route):
        city_sequence = ""
        for city in city_route:
            index = self.city_list.index(city)
            city_sequence += f"{index}-"
        return city_sequence[:-1]

    def tsp_data(self, filename):
        f = open(f"{filename}.txt", "w")

        f.write(f"# TSP INSTANCE: {self.tsp_name} \n")
        f.write(f"# Population size: {self.popSize} \n")
        f.write(f"# Number of generations: {self.generations} \n")
        if self.elite_spinbox.value() == 0:
            f.write("# Elitist strategy: OFF \n")
        else:
            elite = self.elite_spinbox.value()
            f.write("# Elitist strategy: ON \n")
            f.write(f"# Number of elites: {elite} \n")
        f.write(f"# Hillclimbing: {self.hillclimb_type} \n")
        if self.hillclimb_type != "OFF":
            f.write(f"# Hillclimbing generation start: {self.hillclimb_gen_start} \n")
        f.write(f"# Selection type: {self.selection_type} \n")
        f.write(f"# Tournament size: {self.selection_size} \n")
        f.write(f"# Crossover type: {self.crossover_type} \n")
        f.write(f"# Crossover probability: {self.crossover_prob} \n")
        f.write(f"# Mutation type: {self.mutation_type} \n")
        f.write(f"# Mutation probability: {self.mutation_prob} \n")
        if self.seed != "none":
            f.write(f"# Seed: {self.seed} \n")
        if self.num_of_exp > 1:
            f.write(f"# Number of experiments: {self.num_of_exp} \n")

        return f

    def results(self, city_route, best_fitness, average_fitness, routes):
        f = self.tsp_data("results")

        f.write("# GEN \t best_tour \t av_tour \t best_tour_sequence \n")
        for i in range(self.generations):
            best_tour = best_fitness[i]
            av_tour = average_fitness[i]
            best_tour_sequence = self.tsp_city_sequence(routes[i])
            f.write(f"{i} \t {best_tour} \t {av_tour} \t {best_tour_sequence}\n")

        f.close()

    def m_results(self, experiments):
        f = self.tsp_data("m_results")
        x = 1
        for exp in experiments:
            f.write(f"# Experiment {x} \n")
            x += 1
            f.write("# GEN \t best_tour \t av_tour \t best_tour_sequence \n")
            for i in range(self.generations):
                best_tour = exp.best_fitness[i]
                av_tour = exp.average_fitness[i]
                best_tour_sequence = self.tsp_city_sequence(exp.best_routes[i])
                f.write(f"{i} \t {best_tour} \t {av_tour} \t {best_tour_sequence}\n")
            f.write("\n")

        f.close()

    def m_std_results(self, experiments):
        f = self.tsp_data("m_std_results")
        f.write("# GEN \t av_best_tour \t std_av_best_tour \t AV_av_tour \t std_AV_av_tour\n")
        for i in range(self.generations):
            av_best_tour = 0
            std_av_best_tour = []
            AV_av_tour = 0
            std_AV_av_tour = []

            for exp in experiments:
                av_best_tour += exp.best_fitness[i]
                std_av_best_tour.append(exp.best_fitness[i])
                AV_av_tour += exp.average_fitness[i]
                std_AV_av_tour.append(exp.average_fitness[i])

            av_best_tour = av_best_tour / len(experiments)
            std_av_best_tour = stdev(std_av_best_tour)

            AV_av_tour = AV_av_tour / len(experiments)
            std_AV_av_tour = stdev(std_AV_av_tour)

            f.write(f"{i} \t {av_best_tour} \t {std_av_best_tour} \t {AV_av_tour} \t {std_AV_av_tour}\n")
        f.write("\n")

        f.close()

    def m_found_solutions(self, experiments):
        f = self.tsp_data("m_found_solutions")
        # TODO: for every experiment, print their best tour and route sequence
        f.write("# Experiment \t best_tour \t best_tour_sequence  \n")

        i = 0
        for exp in experiments:
            index = exp.best_routes.index(exp.best_route)
            f.write(f"{i} \t {exp.best_fitness[index]} \t {self.tsp_city_sequence(exp.best_route)} \n")
            i += 1

        f.close()


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
