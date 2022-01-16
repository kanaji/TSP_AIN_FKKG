import sys

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog

from genetic_algorithm import geneticAlgorithm, City


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
        self.selection_checkbox.clicked.connect(
            lambda: self.enable_on_check(self.selection_checkbox, self.tournament_radio))
        self.selection_checkbox.clicked.connect(
            lambda: self.enable_on_check(self.selection_checkbox, self.my_selection_radio))
        self.selection_checkbox.clicked.connect(
            lambda: self.enable_on_check(self.selection_checkbox, self.tournament_size))
        self.crossover_checkbox.clicked.connect(lambda: self.enable_on_check(self.crossover_checkbox, self.ox_radio))
        self.crossover_checkbox.clicked.connect(lambda: self.enable_on_check(self.crossover_checkbox, self.cx_radio))
        self.crossover_checkbox.clicked.connect(
            lambda: self.enable_on_check(self.crossover_checkbox, self.my_crossover_radio))
        self.crossover_checkbox.clicked.connect(
            lambda: self.enable_on_check(self.crossover_checkbox, self.crossover_probability))
        self.mutation_checkbox.clicked.connect(lambda: self.enable_on_check(self.mutation_checkbox, self.swap_2_radio))
        self.mutation_checkbox.clicked.connect(
            lambda: self.enable_on_check(self.mutation_checkbox, self.inversion_radio))
        self.mutation_checkbox.clicked.connect(
            lambda: self.enable_on_check(self.mutation_checkbox, self.my_mutation_radio))
        self.mutation_checkbox.clicked.connect(
            lambda: self.enable_on_check(self.mutation_checkbox, self.mutation_probability))

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
            self.file_label.setText(self.fileName[self.fileName.rfind("/") + 1:])
            self.read_file()

    def read_values(self):
        self.popSize = self.population_size.value()
        self.generations = self.number_of_generations.value()
        self.mutationRate = self.mutation_probability.value()

        if self.elitist_checkbox.isChecked():
            self.elite = self.elite_spinbox.value()
            if self.elite > self.popSize:
                return "Elite value is too big"
        else:
            self.elite = 0

        if self.hillclimbing_checkbox.isChecked():
            if self.opt_2_radio.isChecked():
                self.hillclimb_type = "2opt"
            elif self.opt_3_radio.isChecked():
                self.hillclimb_type = "3opt"
            self.hillclimb_gen_start = self.generation_start_hc.value()
            if self.hillclimb_gen_start > self.generations:
                return "Hillclimb starting generation is too big"
        else:
            self.hillclimb_gen_start = 0
            self.hillclimb_type = "none"

        if self.selection_checkbox.isChecked():
            if self.tournament_radio.isChecked():
                self.selection_type = "tournament"
            elif self.my_selection_radio.isChecked():
                self.selection_type = "rank"
            # TODO: Ask about rank, do we use size in it? and what size limitation is?
            self.selection_size = self.tournament_size.value()
        else:
            self.selection_size = 0
            self.selection_type = "none"

        if self.crossover_checkbox.isChecked():
            if self.ox_radio.isChecked():
                self.crossover_type = "ox"
            elif self.cx_radio.isChecked():
                self.crossover_type = "cx"
            elif self.my_crossover_radio.isChecked():
                self.crossover_type = "scx"
            self.crossover_prob = self.crossover_probability.value()
        else:
            self.crossover_prob = 0
            self.crossover_type = "none"

        if self.mutation_checkbox.isChecked():
            if self.swap_2_radio.isChecked():
                self.mutation_type = "swap_2"
            elif self.inversion_radio.isChecked():
                self.mutation_type = "inversion"
            elif self.my_mutation_radio.isChecked():
                self.mutation_type = "last_but_not_least"
            self.mutation_prob = self.mutation_probability.value()
        else:
            self.mutation_prob = 0
            self.mutation_type = "none"

        if self.clock_seed_check.isChecked():
            self.seed = self.seed_value.value()
        else:
            self.seed = "none"

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

        if self.file_cities[-1] == "EOF\n" or self.file_cities[-1] == "\n":
            self.file_cities.pop()
            if self.file_cities[-1] == "EOF\n":
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
        self.read_cities()
        answer = self.read_values()

        if answer == True:
            city_route, best_fitness, average_fitness = geneticAlgorithm(population=self.city_list,
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

            x = []
            y = []
            for city in city_route:
                x.append(city.x)
                y.append(city.y)

            self.tsp_graph_widget.canvas.axes.clear()
            self.tsp_graph_widget.canvas.axes.scatter(x, y)
            self.tsp_graph_widget.canvas.axes.plot(x, y)
            self.tsp_graph_widget.canvas.draw()

            self.data_graph_widget.canvas.axes.clear()
            self.data_graph_widget.canvas.axes.set_ylabel('Tour Length')
            self.data_graph_widget.canvas.axes.set_xlabel('Generation')
            self.data_graph_widget.canvas.axes.plot(best_fitness, label='Best in generation', marker='o')
            self.data_graph_widget.canvas.axes.plot(average_fitness, label='Average in generation', marker='s')
            self.data_graph_widget.canvas.axes.legend(loc='upper right')
            self.data_graph_widget.canvas.draw()

            self.generation_number_graph.setText(f"{self.generations}")
            self.best_tour_graph.setText(f"{best_fitness[self.generations - 1]}")
        else:
            self.error_label.setStyleSheet('color: red')
            self.error_label.setText(answer)

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
