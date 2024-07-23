def to_hierachical_dict(flat_dict: dict, seperator: str = "¦") -> dict:
    hierarchy = {}
    for key, value in flat_dict.items():
        name_and_param = key.split(seperator)
        unraveled_obj_key = name_and_param[0]
        unraveled_param_key = name_and_param[1]
        if unraveled_obj_key not in hierarchy:
            hierarchy[unraveled_obj_key] = {}
        hierarchy[unraveled_obj_key][unraveled_param_key] = value
    return hierarchy


def to_hierachical_dict_arbitrary_depth(flat_dict: dict, separator: str = "¦") -> dict:
    hierarchy = {}
    for key, value in flat_dict.items():
        current_dict = hierarchy
        parts = key.split(separator)
        for part in parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]
        current_dict[parts[-1]] = value
    return hierarchy
