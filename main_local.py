# pip install ultralytics opencv-python
from ultralytics import YOLO

# Carregar o modelo YOLO
# Será necessário baixar o arquivo yolov10m.pt disponível no site YOLOv10.
model = YOLO('yolov10m.pt') # Altere o nome caso for utilizar outra versão.

# Iniciar o treinamento
model.train(data='config_local.yaml', epochs=100)

# Detectar objetos da imagem
results = model.predict('PS_Teste.jpg')

# Exibir resultado
for result in results:
    result.show()  # Exibe a imagem com as detecções
