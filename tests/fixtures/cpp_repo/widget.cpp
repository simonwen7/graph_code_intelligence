namespace ui {

class Widget {
public:
    Widget(int value) : value_(value) {}
    ~Widget() {}

    int value() { return value_; }
    int value() const { return value_; }
    int value() & { return value_; }
    int value() && { return value_; }

    bool operator==(const Widget& other) const {
        return value_ == other.value_;
    }

    int operator[](int index) const { return index; }

    void touch();

private:
    int value_;
};

void Widget::touch() {}

struct Point {
    int x;
    int y;
};

class Outer {
public:
    class Inner {
    public:
        void ping() {}
    };
};

}  // namespace ui
