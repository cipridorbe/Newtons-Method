clang-format -i main.cpp
find src/ \( -iname '*.hpp' -o -iname '*.cpp' \) -exec clang-format -i {} +
find inc/ \( -iname '*.hpp' -o -iname '*.cpp' \) -exec clang-format -i {} +