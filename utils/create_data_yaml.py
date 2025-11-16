import os, yaml
from typing import List

def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
    if not os.path.exists(path_to_classes_txt):
        print(
            f"classes.txt file not found! Please create a classes.txt labelmap and move it to {path_to_classes_txt}"
        )
        return

    with open(path_to_classes_txt, "r") as f:
        classes = []
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            classes.append(line.strip())

    number_of_classes = len(classes)

    data = {
        "path": "data",
        "train": "train/images",
        "val": "validation/images",
        "nc": number_of_classes,
        "names": classes,
    }

    with open(path_to_data_yaml, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Created config file at {path_to_data_yaml}")

    return


def create_data_yaml_from_set_and_classes(
    path_to_set_yaml: str, path_to_classes_txt: str, path_to_data_yaml: str
):
    set_names: List[str] = []
    if os.path.exists(path_to_set_yaml):
        with open(path_to_set_yaml, "r") as f:
            set_yaml = yaml.safe_load(f) or {}
        names_candidate = set_yaml.get("names", []) or []
        if isinstance(names_candidate, list):
            set_names = names_candidate
        else:
            print(
                "Invalid names list in set data.yaml; proceeding with custom classes only"
            )

    custom_names: List[str] = []
    if os.path.exists(path_to_classes_txt):
        with open(path_to_classes_txt, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    custom_names.append(line)
    else:
        print(f"classes.txt not found at {path_to_classes_txt}")

    final_names: List[str] = []
    seen = set()
    for n in set_names:
        if n not in seen:
            final_names.append(n)
            seen.add(n)
    for n in custom_names:
        if n not in seen:
            final_names.append(n)
            seen.add(n)

    if len(final_names) == 0:
        print(
            "No classes found in either set data.yaml or classes.txt; cannot create data.yaml"
        )
        return None

    data = {
        "path": "data",
        "train": "train/images",
        "val": "validation/images",
        "nc": len(final_names),
        "names": final_names,
    }

    with open(path_to_data_yaml, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(
        f"Created config file at {path_to_data_yaml} from union of {path_to_set_yaml} and {path_to_classes_txt}"
    )

    return final_names
