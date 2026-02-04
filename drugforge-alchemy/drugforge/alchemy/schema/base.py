import abc
import json
from typing import Any, Callable, Literal

from openff.units import Quantity
from pydantic import BaseModel, ConfigDict

# the original DefaultModel from openff.models is deprecated, as it only supports pydantic v1
# from openff.models.models import DefaultModel
# The BaseModel and associated functions are copied and updated from openff.models


class QuantityEncoder(json.JSONEncoder):
    """
    JSON encoder for unit-wrapped floats and NumPy arrays.

    This is intended to operate on FloatQuantity and ArrayQuantity objects.
    """

    def default(self, obj):
        if isinstance(obj, Quantity):

            if isinstance(obj.magnitude, (float, int)):
                data = obj.magnitude
            elif isinstance(obj.magnitude, numpy.ndarray):
                data = obj.magnitude.tolist()
            else:
                # This shouldn't ever be hit if our object models
                # behave in ways we expect?
                raise UnsupportedExportError(
                    f"trying to serialize unsupported type {type(obj.magnitude)}"
                )
            return {
                "val": data,
                "unit": str(obj.units),
            }


def custom_quantity_encoder(v):
    """Wrap json.dump to use QuantityEncoder."""
    return json.dumps(v, cls=QuantityEncoder)


def json_loader(data: str) -> dict:
    """Load JSON containing custom unit-tagged quantities."""
    # TODO: recursively call this function for nested models
    out: dict = json.loads(data)
    for key, val in out.items():
        try:
            # Directly look for an encoded FloatQuantity/ArrayQuantity,
            # which is itself a dict
            v = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            # Handles some cases of the val being a primitive type
            continue
        # TODO: More gracefully parse non-FloatQuantity/ArrayQuantity dicts
        unit_ = Unit(v["unit"])
        val = v["val"]
        out[key] = unit_ * val

    return out


class DefaultModel(BaseModel):
    """A custom Pydantic model used by other components."""

    model_config = ConfigDict(
        # use_enum_values=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        json_encoders={Quantity: custom_quantity_encoder},
        json_loads=json_loader,
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
