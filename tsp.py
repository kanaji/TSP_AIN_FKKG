from PyQt5 import QtWidgets, uic
import sys


class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('tsp.ui', self)
        self.show()


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
