import yaml


def parse_params(params_dict):
    """Parse parameters from a dictionary

    Args:
        params_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    experiment_params = params_dict.get("experiment", {})
    train_params = params_dict.get("parameters", {}).get("train_params", {})
    dataset_params = params_dict.get("parameters", {}).get("dataset", {})
    model_params = params_dict.get("parameters", {}).get("model", {})

    return experiment_params, train_params, dataset_params, model_params


def load_yaml(file_path):
    try:
        with open(file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file.read())
            return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
