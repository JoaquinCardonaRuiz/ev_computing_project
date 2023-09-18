import os
import json
import statistics


class Utils:
    def __init__(self, start_dir, prefix):
        self.start_dir = start_dir
        self.prefix = prefix

    def process_directories(self):
        results = {}
        for root, dirs, files in os.walk(self.start_dir):
            for name in dirs:
                if name.startswith(self.prefix):
                    data = self.process_directory(root, name)
                    if data is not None:
                        results[name] = data
        return results

    def process_directory(self, root, name):
        fitness_file = os.path.join(root, name, "fitnesses.json")
        if os.path.isfile(fitness_file):
            with open(fitness_file, "r") as f:
                last_line = f.readlines()[-1]
                data = json.loads(last_line)
                data.update(self.calculate_statistics(data, name))
                kwargs_dict = self.extract_kwargs(name)
                if kwargs_dict is not None:
                    data["kwargs"] = kwargs_dict
                return data
        return None

    def calculate_statistics(self, data, name):
        fitnesses = data["fitnesses"]
        return {
            "name": data.get("name", name),
            "avg_fitness": statistics.mean(fitnesses),
            "max_fitness": max(fitnesses),
            "n_individuals": len(fitnesses),
        }

    def extract_kwargs(self, name):
        parts = name.split("_")
        cx_index = next(
            (i for i, part in enumerate(parts) if part.startswith("cx")), None
        )
        if cx_index is not None and (len(parts) - cx_index - 1) % 2 == 0:
            return {
                parts[i]: float(parts[i + 1])
                for i in range(cx_index + 1, len(parts), 2)
            }
        return None

    def save_results(self, results, filename):
        with open(filename, "w") as f:
            json.dump(results, f)


# Usage:
utils = Utils(".", "crossover_exp_")
results = utils.process_directories()
utils.save_results(results, "results.json")
