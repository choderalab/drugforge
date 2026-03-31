import abc
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

# the original DefaultModel from openff.models is deprecated, as it only supports pydantic v1
# from openff.models.models import DefaultModel
# The BaseModel and associated functions are copied and updated from openff.models


class DefaultModel(BaseModel):
    """A custom Pydantic model used by other components."""

    model_config = ConfigDict(
        # use_enum_values=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )


class _SchemaBase(abc.ABC, DefaultModel):
    """
    A basic schema class used to define the components of the Free energy workflow
    """

    type: Literal["base"] = "base"

    def to_file(self, filename: str):
        """
        Write the model to JSON file.
        """
        from gufe.tokenization import JSON_HANDLER

        from ._util import SCOPEDKEY_CODEC

        JSON_HANDLER.add_codec(SCOPEDKEY_CODEC)

        with open(filename, "w") as output:
            json.dump(self.model_dump(), output, cls=JSON_HANDLER.encoder, indent=2)

    @classmethod
    def from_file(cls, filename: str):
        """
        Load the model from a JSON file
        """
        from gufe.tokenization import JSON_HANDLER

        from ._util import SCOPEDKEY_CODEC

        JSON_HANDLER.add_codec(SCOPEDKEY_CODEC)
        with open(filename) as f:
            return cls.model_validate(json.load(f, cls=JSON_HANDLER.decoder))


class _SchemaBaseFrozen(_SchemaBase):
    type: Literal["_SchemaBaseFrozen"] = "_SchemaBaseFrozen"

    model_config = ConfigDict(frozen=True)
