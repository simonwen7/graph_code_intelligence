from helpers import Base
from helpers import helper as h


class Service(Base):
    def run(self) -> None:
        self.validate()
        h()

    def validate(self) -> None:
        def inner() -> None:
            pass

        inner()

    @classmethod
    def build(cls) -> "Service":
        cls.run()
        return cls()
