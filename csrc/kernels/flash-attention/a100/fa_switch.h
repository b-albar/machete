#pragma once

#define HEAD_DIM_SWITCH(VALUE, CONST_NAME, ...)        \
  [&] {                                                \
    if (VALUE == 64) {                                 \
      constexpr static unsigned int CONST_NAME = 64;   \
      return __VA_ARGS__();                            \
    } else if (VALUE == 128) {                         \
      constexpr static unsigned int CONST_NAME = 128;  \
      return __VA_ARGS__();                            \
    } else {                                           \
      throw std::runtime_error("Value not supported"); \
    }                                                  \
  }()