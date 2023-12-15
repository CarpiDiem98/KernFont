from kernfont.model.model import AlexNetRegression


def get_model(model_name):
    model_dict = {
        "AlexNetRegression": AlexNetRegression(),
        "ResNetRegression": "ResNetRegression",
        "VGGRegression": "VGGRegression",
        "DenseNetRegression": "DenseNetRegression",
    }
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise ValueError("Modello non valido: {}".format(model_name))
