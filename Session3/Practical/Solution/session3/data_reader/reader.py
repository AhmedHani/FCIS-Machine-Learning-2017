class CsvReader(object):
    def __init__(self, file_path):
        self.__file_path = file_path

    def get_iris_data(self):
        iris_features, iris_labels = [], []

        with open(self.__file_path, 'r') as reader:
            all_lines = reader.readlines()
            for line in all_lines[1:]:
                line_tokens = line.strip().split(',')

                #iris_features.append(list(map(lambda v: float(v), line_tokens[1:-1])))
                iris_features.append(list(map(lambda v: float(v), [line_tokens[1], line_tokens[2], line_tokens[4]])))
                iris_labels.append(str(line_tokens[-1]))

        return iris_features, iris_labels