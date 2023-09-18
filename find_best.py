import os
import json
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Utils:
    def __init__(self, start_dir, prefix):
        self.start_dir = start_dir
        self.prefix = prefix

    def process_directories(self):
        # Initialize an empty dictionary to store the results
        results = {}

        for root, dirs, files in os.walk(self.start_dir):
            for name in dirs:
                if name.startswith(self.prefix):
                    # Construct the full path to the fitnesses.json file
                    fitness_file = os.path.join(root, name, "fitnesses.json")

                    # Check if the file exists
                    if os.path.isfile(fitness_file):
                        with open(fitness_file, "r") as f:
                            last_line = f.readlines()[-1]
                            data = json.loads(last_line)

                            # Add the directory name to the dictionary
                            data["name"] = name

                            # Calculate the average fitness
                            avg_fitness = statistics.mean(data["fitnesses"])
                            data["avg_fitness"] = avg_fitness

                            # Calculate the maximum fitness
                            max_fitness = max(data["fitnesses"])
                            data["max_fitness"] = max_fitness

                            # Calculate the number of individuals
                            n_individuals = len(data["fitnesses"])
                            data["n_individuals"] = n_individuals

                            # Store the results in the dictionary
                            results[name] = data

        return results

    def print_results(self, results):
        for name, data in results.items():
            print(
                f"Name: {data['name']}, Generation: {data['generation']}, Average Fitness: {data['avg_fitness']}, Max Fitness: {data['max_fitness']}, Number of Individuals: {data['n_individuals']}"
            )

    def create_bar_chart(self, results):
        # Sort the results in descending order by average fitness and limit to top 10
        sorted_results = sorted(
            results.items(), key=lambda x: x[1]["avg_fitness"], reverse=True
        )[:10]

        # Prepare data for the bar chart
        names = [data["name"] for name, data in sorted_results]
        avg_fitnesses = [data["avg_fitness"] for name, data in sorted_results]
        max_fitnesses = [data["max_fitness"] for name, data in sorted_results]

        # Create a DataFrame from the results
        df = pd.DataFrame(
            list(zip(names, avg_fitnesses, max_fitnesses)),
            columns=["Name", "Average Fitness", "Max Fitness"],
        )

        # Melt the DataFrame to make it suitable for seaborn
        df_melted = pd.melt(
            df, id_vars="Name", var_name="Fitness Type", value_name="Fitness"
        )

        # Create a bar chart using seaborn
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(20, 10))  # Adjust size here

        chart = sns.barplot(x="Name", y="Fitness", hue="Fitness Type", data=df_melted)

        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

        plt.title("Average and Max Fitness by Name")

        plt.tight_layout()

        # Save the figure as a PNG file
        plt.savefig("average_max_fitness.png")


# Usage:
utils = Utils(".", "crossover_exp_")
results = utils.process_directories()
utils.print_results(results)
utils.create_bar_chart(results)
