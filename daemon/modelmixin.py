from pathlib import Path

import yaml


class ModelMixin:
    @property
    def model_name(self) -> str:
        """
        Get the model name from the config file. The model name is the `name` property of the config file.
        """
        return self.config.get("name", "model")

    @property
    def model_suffix(self) -> str:
        """
        Get the model suffix from the config file. The model suffix is the `name` property of the config file with a dot in front of it.
        """
        return "." + self.model_name

    @property
    def config(self) -> dict:
        """
        Get the config dictionary from the config file. Config file is located at `/opt/app/container.yaml`.
        Failure is silent, and will return an empty dictionary.
        """
        config_path = Path("/opt/app/container.yaml")
        if not config_path.is_file():
            return {}
        with config_path.open("r") as fp:
            return yaml.safe_load(fp)
