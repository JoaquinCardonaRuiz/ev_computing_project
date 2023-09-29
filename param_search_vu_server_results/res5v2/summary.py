def count_config_occurrences():
    # Open the file
    with open('results.json', 'r') as f:
        # Read the entire file as a string
        data_str = f.read()

    # Count the occurrences of "config"
    config_count = data_str.count("config")

    print(f"The word 'config' appears {config_count} times in the file.")

count_config_occurrences()
