cmake -G "NMake Makefiles" -B build -S . -D CMAKE_TOOLCHAIN_FILE=C:\Users\admin\source\repos\vcpkg\scripts\buildsystems\vcpkg.cmake -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ -D CMAKE_BUILD_TYPE=Release -D CMAKE_C_FLAGS=-g -D CMAKE_CXX_FLAGS=-g

cmake -B build -S . -D CMAKE_C_COMPILER=/usr/lib/llvm-20/bin/clang -D CMAKE_CXX_COMPILER=/usr/lib/llvm-20/bin/clang++ -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS="-g -stdlib=libc++"

https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/getting_started_windows.md
