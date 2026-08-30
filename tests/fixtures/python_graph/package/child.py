from .base import PackageBase
from . import base
from ..helpers import helper


class Child(PackageBase):
    def run(self) -> None:
        helper()
        marker = base.PackageBase
        return marker
