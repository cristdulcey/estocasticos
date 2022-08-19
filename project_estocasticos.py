import random # # pragma pylint: disable=C0114,E0401
import warnings
import imageio
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

CLASSES = [0, 1, 2]
four_neighbors_related_positions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
eight_neighbors_related_positions = [
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]

PATH = "./data/test9.jpg"


# image = imageio.imread(path, as_gray=True)
# plt.imshow(image, label="gray image", cmap='gray')


def get_class_labels(data: [], mode="gray"):
    labels = np.zeros((len(data), len(data[0])))
    for row in range(len(data)):
        for col in range(len(data[0])):
            value = data[row][col]
            if mode == "rgb":
                value = value[0] * 0.2125 + value[1] * 0.7154 + value[2] * 0.0721
            if value < 85:
                label = 0
            elif value < 170:
                label = 1
            else:
                label = 2
            labels[row][col] = label

    return labels


def get_class_information(data: [], labels: []):
    data_info = {}
    class_info = {}

    for c in CLASSES:
        data_info[c] = {"data": [], "mean": 0, "var": 0, "probability": 0}

        class_info[c] = {"mean": 0, "var": 0, "probability": 0}

    for c in CLASSES:
        for row in range(len(data)):
            for col in range(len(data[0])):
                if labels[row][col] == c:
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
class SimulatedAnnealing:

    def __init__(
            self,
            labels,
            image,
            class_info,
            neighbors_related_positions,
            iterations=1000,
            temperature=10000,
            betha=1,
            schedule_coefficient=1,
            schedule_steps=1000,
            mode="gray",
    ):
        self.labels = labels
        self.image = image
        self.class_info = class_info
        self.iterations = iterations
        self.temperature = temperature
        self.neighbors_related_positions = neighbors_related_positions
        self.betha = betha
        self.schedule_coefficient = schedule_coefficient
        self.schedule_steps = schedule_steps
        self.mode = mode

    def move(self, labels):
        new_classes = np.copy(CLASSES).tolist()

        random_index_x, random_index_y = random.randint(
            0, len(labels) - 1
        ), random.randint(0, len(labels[0]) - 1)
        random_label = random.randint(0, len(new_classes) - 1)

        while labels[random_index_x][random_index_y] == random_label:
            random_label = random.randint(0, len(new_classes) - 1)
        return random_label, random_index_x, random_index_y

    def energy(self, new_labels, label, row, col):
        energy = 0.0
        class_mean = self.class_info[int(label)].get("mean")
        class_var = self.class_info[int(label)].get("var")
        energy += np.log(np.sqrt(2 * np.pi * class_var)) + (
                (label * 127 - class_mean) ** 2
        ) / (2 * (class_var ** 2))
        neighbors_indexes = self.get_neighbours_indexes(row, col)
        for neighbor in neighbors_indexes:
            if neighbor[0] < len(new_labels) and neighbor[1] < len(new_labels[0]):
                energy += self.betha * self.are_different(
                    label, new_labels[neighbor[0]][neighbor[1]]
                )
        return energy

    def anneal(self, temperature_function):
        new_labels = np.copy(self.labels)
        temperature = self.temperature
        for i in range(self.iterations):
            random_label, random_row, random_col = self.move(new_labels)
            current_energy = self.energy(
                new_labels, new_labels[random_row][random_col], random_row, random_col
            )
            new_energy = self.energy(new_labels, random_label, random_row, random_col)
            delta_U = new_energy - current_energy
            random_uniform = random.uniform(0, 1)
            if self.is_eligible_to_update(delta_U, temperature, random_uniform):
                new_labels[random_row][random_col] = random_label
            temperature = temperature_function(temperature)
        return new_labels

    def get_neighbours_indexes(self, row, col):
        indexes = []
        for index in self.neighbors_related_positions:
            indexes.append((row + index[0], col + index[1]))

        return indexes

    @staticmethod
    def is_eligible_to_update(delta_U, temperature, random_uniform):
        return delta_U <= 0 or (
                delta_U > 0 and random_uniform < np.exp(-delta_U / temperature)
        )

    @staticmethod
    def are_different(x, y):
        if x == y:
            return -1
        return 1

    def exponential_schedule(self, temperature):
        return temperature * 0.99


def plot_before_after(before, after):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))

    ax1.set_title("Real image")
    ax1.imshow(before, cmap="gray")

    ax2.set_title("Simulated Annealing")
    ax2.imshow(after, cmap="viridis")

    plt.show()


rgb_image = imageio.imread(PATH)
rgb_labels = get_class_labels(data=rgb_image, mode="rgb")
class_info = get_class_information(rgb_image, rgb_labels)


def get_rgb_class_labels(data: []):
    labels = np.zeros((len(data), len(data[0])))
    for row in range(len(data)):
        for col in range(len(data[0])):
            value = data[row][col]
            if value[2] > 200:
                label = 1
            elif value[2] < 70:
                label = 2
            else:
                label = 0
            labels[row][col] = label

    return labels


rgb_image = imageio.imread(PATH)
rgb_labels = get_rgb_class_labels(data=rgb_image)
class_info = get_class_information(rgb_image, rgb_labels)

simulated_annealer = SimulatedAnnealing(
    labels=rgb_labels,
    image=rgb_image,
    class_info=class_info,
    neighbors_related_positions=eight_neighbors_related_positions,
    iterations=100000,
    betha=0.1,
    mode="rgb",
)

optimized_labels = simulated_annealer.anneal(simulated_annealer.exponential_schedule)
plot_before_after(rgb_image, optimized_labels)
