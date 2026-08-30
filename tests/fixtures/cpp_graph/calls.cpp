#include "alpha.hpp"
#include <vector>

namespace tools {

void unique_helper() {}

double overload(int value) { return value; }
double overload(double value) { return value; }

class Base {};

class Derived : public Base {
public:
    void run() {
        unique_helper();
        overload(1);
        this->run();
        obj.missing();
        ptr->missing();
        Alpha::act();
    }

private:
    Derived* obj;
    Derived* ptr;
};

class Multi : public Base, private Derived {};

class UnknownChild : public MissingBase {};

}  // namespace tools
