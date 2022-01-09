import random
from PyQt5 import QtWidgets, uic
import sys
from genetic_algorithm import geneticAlgorithm, geneticAlgorithmPlot, City

from PyQt5.QtWidgets import QFileDialog


class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('tsp.ui', self)
        self.show()
        self.file_label.setText("")
        self.load_button.clicked.connect(self.open_file)
        self.run_button.clicked.connect(self.run_tsp)

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Wybierz plik z miastami", "",
                                                  "All Files (*);;Python Files (*.py);;Miasta (*.tsp)", options=options)
        if fileName:
            self.file_label.setText(fileName[fileName.rfind("/") + 1:])

    def run_tsp(self):
        print("I have runned a test")


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()

cityList = []

for i in range(0, 25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
