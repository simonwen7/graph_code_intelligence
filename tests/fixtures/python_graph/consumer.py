import helpers
import helpers as hmod
from helpers import helper
from helpers import helper as aliased
from helpers import Base
from helpers import unused_ref
import pathlib


def use_calls() -> None:
    helper()
    aliased()
    hmod.helper()
    helpers.helper()


def use_reference() -> None:
    marker = unused_ref
    return marker


def shadow(helper: object) -> None:
    helper()


def local_shadow() -> None:
    helper = 1
    helper()


def call_dynamic(obj: object) -> None:
    obj.dynamic()
