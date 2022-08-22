import random # # pragma pylint: disable=C0114,E0401
import warnings
import imageio
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

CLASSES = [0, 1, 2]
cuatro_cercanos = [(0, 1), (0, -1), (1, 0), (-1, 0)]
ocho_cercanos= [
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]

# cargar imagen
PATH = "./data/test5.jpg"

#image = imageio.imread(PATH, as_gray=False)
#plt.imshow(image, label="gray image", cmap='gray')

# def obtener_etiquetas(data: [], mode="gray"):
#     etiquetas = np.zeros((len(data), len(data[0])))
#     for row in range(len(data)):
#         for col in range(len(data[0])):
#             value = data[row][col]
#             if mode == "rgb":
#                 value = value[0] * 0.2125 + value[1] * 0.7154 + value[2] * 0.0721
#             if value < 85:
#                 label = 0
#             elif value < 170:
#                 label = 1
#             else:
#                 label = 2
#             etiquetas[row][col] = label

#     return etiquetas


def obtner_informacion(data: [], etiquetas: []):
    data_info = {}
    class_info = {}

    for c in CLASSES:
        data_info[c] = {"data": [], "mean": 0, "var": 0, "probability": 0}

        class_info[c] = {"mean": 0, "var": 0, "probability": 0}

    for c in CLASSES:
        for row in range(len(data)):
            for col in range(len(data[0])):
                if etiquetas[row][col] == c:
                    data_info.get(c).get("data").append(data[row][col])

    data_count = len(data) * len(data[0])
    for c in CLASSES:
        class_info[c] = {
            "mean": np.mean(data_info.get(c).get("data")),
            "var": np.var(data_info.get(c).get("data")),
            "probability": len(data_info.get(c).get("data")) / data_count,
        }
    return class_info

# algorito SimulatedAnnealing
class analisis_simulado:

    def __init__(
            self,
            etiquetas,
            image,
            class_info,
            vecinos_cercanos,
            iteraciones=1000,
            temperatura=10000,
            betha=1,
            coeficiente=1,
            pasos=1000,
            mode="gray",
    ):
        self.etiquetas = etiquetas
        self.image = image
        self.class_info = class_info
        self.iteraciones = iteraciones
        self.temperatura = temperatura
        self.vecinos_cercanos= vecinos_cercanos
        self.betha = betha
        self.coeficiente = coeficiente
        self.pasos= pasos
        self.mode = mode

    def move(self, etiquetas):
        new_classes = np.copy(CLASSES).tolist()

        random_index_x, random_index_y = random.randint(
            0, len(etiquetas) - 1
        ), random.randint(0, len(etiquetas[0]) - 1)
        random_label = random.randint(0, len(new_classes) - 1)

        while etiquetas[random_index_x][random_index_y] == random_label:
            random_label = random.randint(0, len(new_classes) - 1)
        return random_label, random_index_x, random_index_y

    # calcular engergia
    def energia(self, new_etiquetas, label, row, col):
        energia = 0.0
        class_mean = self.class_info[int(label)].get("mean")
        class_var = self.class_info[int(label)].get("var")

        # ecuacion para calcular energia
        energia += np.log(np.sqrt(2 * np.pi * class_var)) + (
                (label * 127 - class_mean) ** 2
        ) / (2 * (class_var ** 2))

        neighbors_indexes = self.get_neighbours_indexes(row, col)
        for neighbor in neighbors_indexes:
            if neighbor[0] < len(new_etiquetas) and neighbor[1] < len(new_etiquetas[0]):
                energia+= self.betha * self.are_different(
                    label, new_etiquetas[neighbor[0]][neighbor[1]]
                )
        return energia
            
    def anneal(self, temperatura_function):
        new_etiquetas = np.copy(self.etiquetas)
        temperatura = self.temperatura
        for i in range(self.iteraciones):
            random_label, random_row, random_col = self.move(new_etiquetas)
            energia_actual= self.energia(
                new_etiquetas, new_etiquetas[random_row][random_col], random_row, random_col
            )
            new_energia = self.energia(new_etiquetas, random_label, random_row, random_col)
            delta_U = new_energia - energia_actual
            random_uniform = random.uniform(0, 1)
            if self.is_eligible_to_update(delta_U, temperatura, random_uniform):
                new_etiquetas[random_row][random_col] = random_label
            temperatura = temperatura_function(temperatura)
        return new_etiquetas

    def get_neighbours_indexes(self, row, col):
        indexes = []
        for index in self.vecinos_cercanos:
            indexes.append((row + index[0], col + index[1]))

        return indexes

    # comprueba si se puede usar el delta_u
    @staticmethod
    def is_eligible_to_update(delta_U, temperatura, random_uniform):
        return delta_U <= 0 or (
                delta_U > 0 and random_uniform < np.exp(-delta_U / temperatura)
        )

    @staticmethod
    def are_different(x, y):
        if x == y:
            return -1
        return 1

    def exponencial(self, temperatura):
        return temperatura * 0.99

def plot_trama_antes_y_despues(before, after, name):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))

    ax1.set_title("Imagen Real")
    ax1.imshow(before, cmap="gray")

    ax2.set_title("Analisis_Simulado")
    ax2.imshow(after, cmap="viridis")

    #plt.show()
    plt.savefig(name+".png")

# rgb_image = imageio.imread(PATH)
# rgb_etiquetas = obtener_etiquetas(data=rgb_image, mode="rgb")
# class_info = obtner_informacion(rgb_image, rgb_etiquetas)

# obtener parametros para pasarlos a al algorimo Simulated Anneling
def obtener_rgb_etiquetas(data: []):
    etiquetas = np.zeros((len(data), len(data[0])))
    for row in range(len(data)):
        for col in range(len(data[0])):
            value = data[row][col]
            #print(value)
            if value[2] > 200: # color oscuro
                label = 1
            elif value[2] < 70:
                label = 2
            else:
                label = 0
            etiquetas[row][col] = label

    return etiquetas

# cargar imagen a procesar
rgb_image = imageio.imread(PATH)
# obtener etiquetas de la imagen de acuerdo al rgb
rgb_etiquetas = obtener_rgb_etiquetas(data=rgb_image)
#  obtener media, varianza, probabilidad de las etiqutas del rgb
class_info = obtner_informacion(rgb_image, rgb_etiquetas)

# Segmentar imagen por clases
simulacion_analizada = analisis_simulado(
    etiquetas=rgb_etiquetas,
    image=rgb_image,
    class_info=class_info,
    vecinos_cercanos= ocho_cercanos,
    iteraciones=100000,
    betha=0.1,
    mode="rgb",
)

etiquetas_optimizadas = simulacion_analizada.anneal(simulacion_analizada.exponencial)
#plot_trama_antes_y_despues(rgb_image, etiquetas_optimizadas )

plot_trama_antes_y_despues(rgb_image, etiquetas_optimizadas, "antes y despues")

def plot_segmentacion_clases(image, name):

    fig = plt.figure(figsize=(20,10))
    ax1 = fig.subplots(1)
    #ax1.imshow(image, cmap='inferno')
    ax1.imshow(image, cmap='inferno')

    #ax1.legend(loc=4)
    #plt.show()
    plt.savefig(name+".png")

plot_segmentacion_clases(etiquetas_optimizadas, "imagen segmentacion clases")

print(etiquetas_optimizadas.shape)

# for i in etiquetas_optimizadas:
#     print('\t'.join(map(str, i)))

img1 = np.zeros_like(etiquetas_optimizadas)
img2 = np.zeros_like(etiquetas_optimizadas)
img3 = np.zeros_like(etiquetas_optimizadas)

for i in range(len(etiquetas_optimizadas)):
  for j in range(len(etiquetas_optimizadas[i])):
    if(etiquetas_optimizadas[i][j]!=0):
      img1[i][j] = 0.0
    else:
      img1[i][j] = 1.0


for i in range(len(etiquetas_optimizadas)):
  for j in range(len(etiquetas_optimizadas[i])):
    if(etiquetas_optimizadas[i][j]!=1):
      img2[i][j] = 0.0
    else:
      img2[i][j] = etiquetas_optimizadas[i][j]


for i in range(len(etiquetas_optimizadas)):
  for j in range(len(etiquetas_optimizadas[i])):
    if(etiquetas_optimizadas[i][j]!=2):
      img3[i][j] = 0.0
    else:
      img3[i][j] = etiquetas_optimizadas[i][j]

plot_segmentacion_clases(img1, "img1")
plot_segmentacion_clases(img2, "img2")
plot_segmentacion_clases(img3, "img3")

# test = [[2.0, 2.0, 0.0, 0.0 ],
#         [0.0, 0.0, 1.0, 1.0 ]
#         ]

# fig, (ax1) = plt.subplots(1)

# ax1.set_title('Simulated Annealing')
# ax1.imshow(test, cmap='viridis')

# plt.show()

# b = np.zeros_like(test)

# for i in range(len(test)):
#   for j in range(len(test[i])):
#     if(test[i][j]!=2):
#       b[i][j] = 0.0
#     else:
#       b[i][j] = test[i][j]

# print(b)


# fig, (ax1) = plt.subplots(1)

# ax1.set_title('Simulated Annealing')
# ax1.imshow(b, cmap='viridis')

# plt.show()