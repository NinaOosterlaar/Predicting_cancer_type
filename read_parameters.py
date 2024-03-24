def retrieve_parameters(filename: str, BREAST: bool, LUNG: bool, MELANOMA: bool):
    """Retrieves the best parameters and the best score from a file."""
    with open(filename, "r") as file:
        lines = file.readlines()
    if BREAST and LUNG and MELANOMA:
        for c, line in enumerate(lines):
            if "Breast" in line and "Melanoma" in line and "Lung" in line:
                parameters = lines[c + 1].split()
                return parameters[2:]
    elif BREAST and LUNG:
        for c, line in enumerate(lines):
            if "Breast" in line and "Lung" in line:
                parameters = lines[c + 1].split()
                return parameters[2:]
    elif BREAST and MELANOMA:
        for c, line in enumerate(lines):
            if "Breast" in line and "Melanoma" in line:
                parameters = lines[c + 1].split()
                return parameters[2:]
    elif LUNG and MELANOMA:
        for c, line in enumerate(lines):
            if "Lung" in line and "Melanoma" in line:
                parameters = lines[c + 1].split()
                return parameters[2:]
  
            
def preprocess_parameters(parameters_string):
    parameters = {}
    for i in range(0, len(parameters_string), 2):
        parameters_string[i] = parameters_string[i].replace(":", "")
        parameters_string[i] = parameters_string[i].replace(",", "")
        parameters_string[i] = parameters_string[i].replace("'", "")
        parameters_string[i] = parameters_string[i].replace("{", "")
        parameters_string[i] = parameters_string[i].replace("}", "")
        parameters_string[i+1] = parameters_string[i+1].replace(":", "")
        parameters_string[i+1] = parameters_string[i+1].replace(",", "")
        parameters_string[i+1] = parameters_string[i+1].replace("'", "")
        parameters_string[i+1] = parameters_string[i+1].replace("{", "")
        parameters_string[i+1] = parameters_string[i+1].replace("}", "")
        if is_float(parameters_string[i+1]):
            parameters_string[i+1] = float(parameters_string[i+1])
        if parameters_string[i+1] == "None":
            parameters_string[i+1] = None
        parameters[parameters_string[i]] = parameters_string[i + 1]
    return parameters
    
    
def is_float(element: any):
    try:
        float(element)
        return True
    except ValueError:
        return False