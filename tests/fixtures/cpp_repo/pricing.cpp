namespace pricing {

double calculate(int quantity) {
    return static_cast<double>(quantity);
}

double calculate(double subtotal) {
    return subtotal;
}

}  // namespace pricing

namespace pricing {

int tally() {
    return 1;
}

}  // namespace pricing (reopened)

namespace a {
namespace b {
void nested_foo() {}
}
}  // namespace a::b

int global_add(int left, int right) {
    return left + right;
}
