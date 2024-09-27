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

def write_results(json_file_path, updated_data):
    with open(json_file_path, 'r') as f:
        data = json.load(f)


    for entry in data:
        if (entry['country'] == updated_data['country'] and
            entry['commodity'] == updated_data['commodity'] and
            entry['path'] == updated_data['path']):
            
            entry['best_params'] = updated_data['best_params']
            entry['best_mae'] = updated_data['best_mae']
            break

    with open(json_file_path, 'w') as f:
        json.dump(data, f, default=numpy_converter, indent=4)
      
      
def write_tl_results(json_file_path, updated_data):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Use next with a generator expression to find the matching entry
    entry = next((entry for entry in data if 
                   entry['base'] == updated_data['base'] and
                   entry['country'] == updated_data['country'] and
                   entry['commodity'] == updated_data['commodity'] and
                   entry['path'] == updated_data['path']), None)

    if entry:
        entry['best_params'] = updated_data['best_params']
        entry['best_mae'] = updated_data['best_mae']
    else:
        data.append(updated_data)

    with open(json_file_path, 'w') as f:
        json.dump(data, f, default=numpy_converter, indent=4)  
        

def get_result(json_file_path, country, commodity):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        if entry['country'] == country and entry['commodity'] == commodity:
            return entry

    return None


def get_tl_result(json_file_path, target_country, base, commodity, path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        if entry['country'] == target_country and entry['base'] == base and entry['commodity'] == commodity and entry['path'] == path:
            return entry

    return None


def get_all_metrics(target_country, country, commodity):
    target_country_mae = get_result(c.get_small_model_results(), target_country, commodity)['best_mae']
    country_mae = get_result(c.get_large_model_results(), country, commodity)['best_mae']
    tl_mae = get_tl_result(c.get_tl_model_results(), target_country, country, commodity, c.get_tl_model_filename(country, target_country, commodity, 'new-layers'))['best_mae']
    
    return [country_mae, target_country_mae, tl_mae]
    

def get_parameters(param_grid):
    network_type = np.random.choice(param_grid['network_type'])
    window_size = np.random.choice(param_grid['window_size'])
    learning_rate = np.random.uniform(*param_grid['learning_rate'])
    num_layers = np.random.choice(param_grid['num_layers'])
    neurons_per_layer = np.random.choice(param_grid['neurons_per_layer'])
    batch_size = np.random.choice(param_grid['batch_size'])
    
    
    return {
        'network_type': network_type, 
        'window_size': window_size,
        'learning_rate': learning_rate,
        'num_layers': num_layers,
        'neurons_per_layer': neurons_per_layer,
        'batch_size': batch_size
    }
