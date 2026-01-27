import json
import os

def remove_task_type(data_path, output_path):
    for filename in os.listdir(data_path):
        if filename.endswith(".json"):
            file_path = os.path.join(data_path, filename)
            output_file_path = os.path.join(output_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if 'task_type' in item:
                        del item['task_type']
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

if __name__ == "__main__":
    data_path = "data/example_data"
    output_path = "data/processed_data"
    os.makedirs(output_path, exist_ok=True)
    remove_task_type(data_path, output_path)