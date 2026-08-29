@decorator
def decorated_function(value: int) -> int:
    return value


@decorator
class DecoratedClass:
    @classmethod
    def build(cls) -> "DecoratedClass":
        return cls()
