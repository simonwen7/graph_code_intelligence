def helper(value: int) -> int:
    return value


def unused_ref() -> object:
    return None


def caller() -> int:
    return helper(1)


class Base:
    def shared(self) -> None:
        pass
