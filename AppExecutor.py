import sys
import warnings
import socket
from PyQt5.QtWidgets import QMessageBox, QApplication, QMainWindow
from BoxPanel import Ui_MainPanel


close_flag = False
warnings.filterwarnings('ignore')


# send a message to command module
def send_message(cmd, sock):
    try:
        sock.send(cmd)
        if cmd != b'5':
            print("Output command:", cmd)
        return True
    except OSError:
        print("Error, module is not responding")
        return False


# Application Board
class MyWindow(QMainWindow):
    def __init__(self, sock):
        super(MyWindow, self).__init__()
        self.ui = Ui_MainPanel()
        self.ui.setupUi(self)
        self.ui.barrierStatus.setText("Шлагбаум: - ")
        self.ui.barrierStatus.adjustSize()
        self.ui.carNumber.setText('Номер авто: - ')
        self.ui.carNumber.adjustSize()
        self.ui.outCar.setText('Авто на выезде: - ')
        self.ui.outCar.adjustSize()
        self.ui.openButton.clicked.connect(self.open_barrier)
        self.ui.closeButton.clicked.connect(self.close_barrier)
        self.port = sock
        self.OPEN = b'2'
        self.CLOSE = b'1'

    def open_barrier(self):
        if send_message(self.OPEN, sock=self.port):
            self.ui.barrierStatus.setText('Шлагбаум: открыт')
            self.ui.barrierStatus.adjustSize()
        else:
            self.ui.barrierStatus.setText('Команда \nне принята')
            self.ui.barrierStatus.adjustSize()

    def close_barrier(self):
        if send_message(self.CLOSE, sock=self.port):
            self.ui.barrierStatus.setText('Шлагбаум: закрыт')
            self.ui.barrierStatus.adjustSize()
        else:
            self.ui.barrierStatus.setText('Команда \nне принята')
            self.ui.barrierStatus.adjustSize()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "Вы точно хотите закрыть приложение?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            global close_flag
            close_flag = True
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        app = QApplication(sys.argv)
        application = MyWindow(sock=s)
        application.show()
        sys.exit(app.exec())
