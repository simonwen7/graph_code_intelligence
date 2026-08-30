#pragma once

void process(int value);

class Service {
public:
    void run();
    void inline_run() { (void)0; }
};
