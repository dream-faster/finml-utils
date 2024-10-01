import dataclasses


def val(t):
    """
    Validate the object `t` types based on the
    __annotations__ dictionary.
    """
    for k, v in t.__annotations__.items():
        assert isinstance(getattr(t, k), v)


@dataclasses.dataclass(frozen=True, slots=True)
class BaseDataClass:
    def __post_init__(self):
        val(self)
