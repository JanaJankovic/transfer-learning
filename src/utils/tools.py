import json
import numpy as np
import utils.constants as c


def numpy_converter(obj):
    """Convert NumPy types to Python-native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to list
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def write_results(json_file_path, new_data):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    if "target_country" in new_data:
        data = [
            entry
            for entry in data
            if not (
                entry["target_country"] == new_data["target_country"]
                and entry["country"] == new_data["country"]
                and entry["commodity"] == new_data["commodity"]
            )
        ]
    else:
        data = [
            entry
            for entry in data
            if not (
                entry["country"] == new_data["country"]
                and entry["commodity"] == new_data["commodity"]
            )
        ]

    data.append(new_data)

    with open(json_file_path, "w") as f:
        json.dump(data, f, default=numpy_converter, indent=4)


def get_result(json_file_path, country, commodity):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    for entry in data:
        if entry["country"] == country and entry["commodity"] == commodity:
            return entry

    return None


def get_tl_result(json_file_path, target_country, base, commodity, path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    for entry in data:
        if (
            entry["country"] == target_country
            and entry["base"] == base
            and entry["commodity"] == commodity
        ):
            return entry

    return None


def get_models_eval_metric(metric, json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    return [entry["evaluation"][metric] for entry in data]


def get_tl_metrics(metric, json_file_path, target_country):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Filter and return 'best_mae' for the target country
    return [
        entry["evaluation"][metric]
        for entry in data
        if entry.get("target_country") == target_country
    ]


def get_all_metrics(target_country, country, commodity):
    target_country_mae = get_result(
        c.get_small_model_results(), target_country, commodity
    )["best_mae"]
    country_mae = get_result(c.get_large_model_results(), country, commodity)[
        "best_mae"
    ]
    tl_mae = get_tl_result(
        c.get_tl_model_results(),
        target_country,
        country,
        commodity,
        c.get_tl_model_filename(country, target_country, commodity, "new-layers"),
    )["best_mae"]

    return [country_mae, target_country_mae, tl_mae]


def get_parameters(param_grid):
    network_type = np.random.choice(param_grid["network_type"])
    learning_rate = np.random.uniform(*param_grid["learning_rate"])
    num_layers = np.random.choice(param_grid["num_layers"])
    neurons_per_layer = np.random.choice(param_grid["neurons_per_layer"])
    batch_size = np.random.choice(param_grid["batch_size"])

    return {
        "network_type": network_type,
        "learning_rate": learning_rate,
        "num_layers": num_layers,
        "neurons_per_layer": neurons_per_layer,
        "batch_size": batch_size,
    }
