#pragma once
#include "vector.h"

Real halton_sequence(int index, int base) {
    Real result = 0;
    Real f = 1.0 / base;
    int i = index;
    while (i > 0) {
        result += f * (i % base);
        i = std::floor(i / base);
        f /= base;
    }
    return result;
}

Vector2 halton_sequence2d(int index) {
    return Vector2{halton_sequence(index, 2), halton_sequence(index, 3)};
}

struct halton_state {
    int index;
};

halton_state init_halton_state(int seed) {
    return halton_state{seed};
}

inline Vector2 next_halton_sequence2d(halton_state &state) {
    return halton_sequence2d(state.index++);
}