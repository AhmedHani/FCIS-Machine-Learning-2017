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

    def get_titanic_data(self):
        titanic_features, titanic_labels = [], []
        temp_features = []

        with open(self.__file_path, 'r') as reader:
            all_lines = reader.readlines()
            for line in all_lines[1:]:
                line_tokens = line.strip().split(',')

                # iris_features.append(list(map(lambda v: float(v), line_tokens[1:-1])))
                temp_features.append(list(map(lambda v: str(v), [line_tokens[2], line_tokens[5], line_tokens[6], line_tokens[7], line_tokens[8]])))
                titanic_labels.append(str(line_tokens[1]))

        for row in temp_features:
            temp = []
            # pclass - sex - age - sibsp - parch
            temp.append(float(row[0]))
            if row[1] == "male":
                temp.append(0.0)
            else:
                temp.append(1.0)
            if row[2] != '':
                temp.append(float(row[2]))
            else :
                temp.append(0.0)
            temp.append(float(row[3]))
            temp.append(float(row[4]))
            titanic_features.append(temp)
            print(row)

        return titanic_features, titanic_labels
