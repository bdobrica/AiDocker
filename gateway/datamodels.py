import inspect
from enum import Enum
from pathlib import Path
from typing import Optional, Type, Union

from fastapi import Form, UploadFile
from pydantic import BaseModel
from pydantic.fields import Field, ModelField


def as_form(cls: Type[BaseModel]) -> Type[BaseModel]:
    parameters = []
    for field in cls.__fields__.values():
        field: ModelField
        parameters.append(
            inspect.Parameter(
                field.alias,
                inspect.Parameter.POSITIONAL_ONLY,
                default=Form(
                    ...,
                    title=field.field_info.title,
                    description=field.field_info.description,
                )
                if field.required
                else Form(
                    field.default,
                    title=field.field_info.title,
                    description=field.field_info.description,
                ),
                annotation=field.outer_type_,
            )
        )

    async def as_form_func(**kwargs):
        return cls(**kwargs)

    sig = inspect.signature(as_form_func)
    sig = sig.replace(parameters=parameters)
    as_form_func.__signature__ = sig
    setattr(cls, "as_form", as_form_func)
    return cls


def models_from_scan(path: Union[str, Path] = "..") -> dict:
    path = Path(path)
    found = []
    for model_path in path.glob("*"):
        if not model_path.is_dir() or not (model_path / "weights.txt").exists():
            continue
        model_name = model_path.name
        found.append((model_name.upper(), model_name))
    return dict(found)


@as_form
class ImageRequest(BaseModel):
    file: UploadFile = Field(title="Image file")
    background: Optional[str] = Field(title="Background color")


@as_form
class TextRequest(BaseModel):
    text: str = Field(title="Text")


AvailableModels = Enum("AvailableModels", models_from_scan())


class RootResponse(BaseModel):
    version: str = Field(
        title="API Version", description="The current API version"
    )
