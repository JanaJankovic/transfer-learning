import json

def write_results(json_file_path, updated_data):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        if (entry['country'] == updated_data['country'] and
            entry['commodity'] == updated_data['commodity'] and
            entry['path'] == updated_data['path']):
            
            entry['best_params'] = updated_data['best_params']
            entry['base_mae'] = updated_data['base_mae']
            break

    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)
        

def get_result(json_file_path, country, commodity):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        if entry['country'] == country and entry['commodity'] == commodity:
            return entry

    return None

