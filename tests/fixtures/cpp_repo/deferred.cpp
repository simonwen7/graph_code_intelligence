template <typename T>
T max_value(T a, T b) {
    return a > b ? a : b;
}

template <typename T>
class Box {
public:
    T get() const { return value; }

private:
    T value;
};

enum Color { Red, Green };
union Packet { int i; float f; };

void uses_lambda() {
    auto fn = [](int x) { return x; };
    (void)fn(1);
}

namespace broken {
int incomplete(
}
