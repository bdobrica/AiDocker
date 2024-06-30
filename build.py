# /usr/bin/env python3
from pathlib import Path

from pick import pick


def get_usecase(model_path: Path) -> str:
    readme_file = model_path / "README.md"
    use_case = ""
    if not readme_file.is_file():
        return use_case

    with readme_file.open("r") as fp:
        next_line_is_usecase = False
        for line in fp:
            line = line.strip()
            if "## use case" in line.lower():
                next_line_is_usecase = True
                continue

            if line and next_line_is_usecase:
                use_case = line
                break

    return use_case


def download_weights(weights_path: Path):
    if not weights_path.is_file():
        return
    with weights_path.open("r") as fp:
        for line in fp:
            line = line.strip()
            if line:
                print(line)


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    models = []
    for model_path in current_dir.glob("*"):
        if model_path.name[0] == "." or not model_path.is_dir() or not (model_path / "Dockerfile").is_file():
            continue

        model_name = model_path.name
        weights_path = model_path / "weights.txt"
        model_use_case = get_usecase(model_path)

        models.append(
            {
                "name": model_name,
                "use_case": model_use_case,
                "weights_path": weights_path,
            }
        )

    selections = pick(
        options=[f"{model['name']} : {model['use_case']}" for model in models],
        title="Select one or more models to build",
        multiselect=True,
    )

    for _, selection in selections:
        model_data = models[selection]
        download_weights(model_data["weights_path"])
