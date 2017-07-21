import random as rnd


def shuffle(training_data, training_labels):
    shuffled_data = []
    shuffled_labels = []

    size = len(training_data)
    random_indices = rnd.sample(range(0, size), size)

    for i in range(0, len(random_indices)):
        shuffled_data.append(training_data[random_indices[i]])
        shuffled_labels.append(training_labels[random_indices[i]])

    return shuffled_data, shuffled_labels


def to_onehot(labels):
    setosa = [1, 0, 0]
    versicolor = [0, 1, 0]
    verginicia = [0, 0, 1]

    new_labels = []

    for i in range(0, len(labels)):
        if labels[i] == "Iris-versicolor":
            new_labels.append(versicolor)
        elif labels[i] == "Iris-virginica":
            new_labels.append(verginicia)
        else:
            new_labels.append(setosa)

    return new_labels


def standardize(features_matrix, means=None, stds=None):
    if means is not None and stds is not None:
        for i in range(0, 2):
            features_matrix[:, i] = (features_matrix[:, i] - means[i]) / stds[i]

        return features_matrix

    means_vec, stds_vec = [], []

    for i in range(0, 2):
        current_mean = features_matrix[:, i].mean()
        current_std = features_matrix[:, i].std()

        means_vec.append(current_mean)
        stds_vec.append(current_std)

        features_matrix[:, i] = (features_matrix[:, i] - current_mean) / current_std

    return features_matrix, means_vec, stds_vec






