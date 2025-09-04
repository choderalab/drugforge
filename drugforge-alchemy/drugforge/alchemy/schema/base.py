import abc
import json
from typing import Literal
from pydantic import BaseModel

from openff.units import Quantity
# the original DefaultModel from openff.models is deprecated, as it only supports pydantic v1
#from openff.models.models import DefaultModel

class DefaultModel(BaseModel):
    """A custom Pydantic model used by other components."""

    model_config = ConfigDict(
        #use_enum_values=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

        # json_encoders: dict[Any, Callable] = {
        #     Quantity: custom_quantity_encoder,
        # }
        # json_loads: Callable = json_loader



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
            json.dump(self.dict(), output, cls=JSON_HANDLER.encoder, indent=2)

    @classmethod
    def from_file(cls, filename: str):
        """
        Load the model from a JSON file
        """
        from gufe.tokenization import JSON_HANDLER
        from ._util import SCOPEDKEY_CODEC

        JSON_HANDLER.add_codec(SCOPEDKEY_CODEC)
        with open(filename) as f:
            return cls.parse_obj(json.load(f, cls=JSON_HANDLER.decoder))


class _SchemaBaseFrozen(_SchemaBase):
    type: Literal["_SchemaBaseFrozen"] = "_SchemaBaseFrozen"

    class Config:
        allow_mutation = False
