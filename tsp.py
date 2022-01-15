import random
from PyQt5 import QtWidgets, uic
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvasQTAgg
from genetic_algorithm import geneticAlgorithm, geneticAlgorithmPlot, City

from PyQt5.QtWidgets import QFileDialog


class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('tsp.ui', self)
        self.show()
        self.city_list = []
        self.file_label.setText("")
        self.load_button.clicked.connect(self.open_file)
        self.run_button.clicked.connect(self.run_tsp)

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Wybierz plik z miastami", "",
                                                       "All Files (*);;Python Files (*.py);;Miasta (*.tsp)",
                                                       options=options)
        if self.fileName:
            self.file_label.setText(self.fileName[self.fileName.rfind("/") + 1:])
            self.read_file()

    def read_values(self):
        self.popSize = self.population_size.value()
        self.generations = self.number_of_generations.value()
        # eliteSize =
        self.mutationRate = self.mutation_probability.value()

    def read_file(self):
        file = open(self.fileName, "r")
        self.file_cities = file.readlines()

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

        for city in self.file_cities:
            city = city.split(" ")
            city = list(filter("".__ne__, city))
            x = int(float(city[1]))
            y = int(float(city[2]))
            self.city_list.append(City(x, y))

    def run_tsp(self):
        self.read_cities()
        self.read_values()

        city_route = geneticAlgorithm(population=self.city_list, popSize=self.popSize, eliteSize=20,
                                      mutationRate=self.mutationRate, generations=self.generations)

        best_fitness, average_fitness = geneticAlgorithmPlot(population=self.city_list, popSize=self.popSize, eliteSize=20,
                                                mutationRate=self.mutationRate,
                                                generations=self.generations)

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


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
