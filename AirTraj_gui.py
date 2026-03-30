import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QActionGroup, QLabel, QComboBox
# Importa o seu script original
import AirTrajSimPy_main

class AirTrajSimGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Configuração básica da janela
        self.setWindowTitle("Simulador de Trajetória - Airsoft")
        self.resize(300, 150)

        # Criação do widget central e do layout vertical
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Criação do botão "Simular"
        self.btn_simular = QPushButton("Simular")
        self.btn_simular.setMinimumHeight(60)
        self.btn_simular.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.btn_config = QPushButton("Configurações")
        self.btn_config.setMinimumHeight(60)
        self.btn_config.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.btn_cilindro = QPushButton("Tipo de Cilindro")
        self.btn_cilindro.setMinimumHeight(60)
        self.btn_cilindro.setVisible(False)
        self.btn_cilindro.setStyleSheet("font-size: 14px; font-weight: bold;")

        # Combo de seleção escondido inicialmente
        self.combo_cilindro = QComboBox()
        self.combo_cilindro.addItem("Selecione o tipo de cilindro", None)
        self.combo_cilindro.addItem("Type 2 - Ported (52mm)", 52.0)
        self.combo_cilindro.addItem("Type 0 - Full (72mm)", 72.0)
        self.combo_cilindro.setVisible(False)
        self.combo_cilindro.currentIndexChanged.connect(self.on_cilindro_changed)

        # Conecta o clique do botão à função main do seu script
        self.btn_simular.clicked.connect(AirTrajSimPy_main.main)
        self.btn_config.clicked.connect(self.toogle_config_menu)
        self.btn_cilindro.clicked.connect(self.toggle_cilindro_menu)

        layout.addWidget(self.btn_simular)
        layout.addWidget(self.btn_config)
        layout.addWidget(self.btn_cilindro)
        layout.addWidget(self.combo_cilindro)

    def toggle_cilindro_menu(self):
        self.combo_cilindro.setVisible(not self.combo_cilindro.isVisible())
        
    def toogle_config_menu(self):
        self.btn_cilindro.setVisible(not self.btn_cilindro.isVisible())
        self.combo_cilindro.setVisible(False)  # Esconde o combo quando o menu de cilindro é ocultado   
            

    def on_cilindro_changed(self, index):
        valor = self.combo_cilindro.itemData(index)
        if valor is None:
            AirTrajSimPy_main.tipoCilindro = None
            print("TipoCilindro resetado (nenhum selecionado)")
        else:
            AirTrajSimPy_main.tipoCilindro = valor
            print(f"TipoCilindro definido para {valor}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AirTrajSimGUI()
    window.show()
    sys.exit(app.exec_())
