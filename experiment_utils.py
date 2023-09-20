import os
import json
import statistics
from typing import Dict, Optional


def process_prefixed_dirs(start_dir: str, prefix: str) -> Dict:
    """
    Walks through the directory structure starting from `start_dir`,
    processes directories that start with `prefix` and returns a dictionary
    of results.
    """
    results = {}
    for root, dirs, _ in os.walk(start_dir):
        for name in dirs:
            if name.startswith(prefix):
                data = process_fit_file(root, name)
                if data is not None:
                    results[name] = data
    return results


def process_fit_file(root: str, name: str) -> Optional[Dict]:
    """
    Processes a directory with name `name` under the root directory `root`.
    Returns a dictionary of data if the directory contains a file named
    "fitnesses.json", otherwise returns None.
    """
    fitness_file = os.path.join(root, name, "fitnesses.json")
    if os.path.isfile(fitness_file):
        with open(fitness_file, "r") as f:
            last_line = f.readlines()[-1]
            data = json.loads(last_line)
            data.update(calc_fit_stats(data, name))
            kwargs_dict = extract_kwargs_from_name(name)
            if kwargs_dict is not None:
                data["kwargs"] = kwargs_dict
            return data
    return None


def calc_fit_stats(data: Dict, name: str) -> Dict:
    """
    Calculates statistics from the fitnesses in `data` and returns a dictionary
    of results. The `name` parameter is used as a fallback if "name"
    is not present in `data`.
    """
    fitnesses = data["fitnesses"]
    return {
        "name": data.get("name", name),
        "avg_fitness": statistics.mean(fitnesses),
        "max_fitness": max(fitnesses),
        "n_individuals": len(fitnesses),
    }


def extract_kwargs_from_name(name: str) -> Optional[Dict]:
    """
    Extracts keyword arguments from the directory `name` and returns a
    dictionary of results.
    Returns None if no keyword arguments can be extracted.
    """
    parts = name.split("_")
    cx_index = next((i for i, part in enumerate(parts) if part.startswith("cx")), None)
    if cx_index is not None and (len(parts) - cx_index - 1) % 2 == 0:
        return {
            parts[i]: float(parts[i + 1]) for i in range(cx_index + 1, len(parts), 2)
        }
    return None


def save_sorted_results(results: Dict, filename: str) -> None:
    """
    Saves the `results` dictionary to a file with name `filename`.
    The results are sorted by average fitness in descending order by default.
    """
    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1]["avg_fitness"], reverse=True)
    )
    with open(filename, "w") as f:
        json.dump(sorted_results, f)


def print_top_n_results_from_file(filename: str, n: int) -> None:
    """
    Reads the results from a file with name `filename` and prints the top `n` results.
    Assumes that the results in the file are already sorted.
    """
    with open(filename, "r") as f:
        results = json.load(f)

    for i, (_, data) in enumerate(list(results.items())[:n]):
        print(f"Result {i+1}:")
        for key in data:
            if key != "fitnesses":
                print(f"{key}: {data[key]}")
        print()
