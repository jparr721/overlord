# Cerebrum
C++ Neural Networking library.

## Getting Started
### Requirements
- Boost >= 1.69
- Armadillo >= 9.3
- cmake >= 3.9
- Boost Unit Test Framework

### Compiling
You can get the dependencies on arch linux via:
```
yay -S boost armadillo cmake 
```
This project can be cloned anywhere on your system.

CMake supports out of source builds, so we can make a build directory you may also see a warning if you do an in source build because it's gross to do that.Inside of the project directory, do the following:
```
mkdir build
cd build
cmake ..
make
```
To run the tests:
```
cd build/bin
./executable_name
```
