# QYG_Vision2.0 WSL2 Environment Setup Guide

This guide is for building `QYG_Vision2.0` under `WSL2 + Ubuntu 22.04`.

Goal:
- finish dependency installation in WSL
- pass CMake configure
- successfully build at least `minimum_vision_system`
- successfully build the main target `QYG_hero`

This guide was written against the current repository state and verified locally with:
- `Ubuntu 22.04.2 LTS`
- `WSL2`
- `cmake -S . -B /tmp/qyg_wsl_clean`
- `cmake --build /tmp/qyg_wsl_clean -j4 --target minimum_vision_system`
- `cmake --build /tmp/qyg_wsl_clean -j4 --target QYG_hero`

## 1. Recommended environment

- Windows side: Windows 11 with WSL2 enabled
- WSL distro: Ubuntu 22.04
- Suggested hardware:
  - 16 GB RAM minimum
  - 30 GB+ free disk space
- Store the project inside the Linux filesystem, for example:
  - `/home/<your_name>/RM/QYG_Vision2.0`
- Do not place the project under `/mnt/c/...` when compiling

## 2. Important warning: disable Conda first

If your shell auto-activates Conda, CMake may accidentally pull headers from Conda, which can cause
`fmt` and `spdlog` version conflicts during compilation.

Typical symptom:

```text
error: 'basic_runtime' is not a member of 'fmt'
```

Before configuring this project, use a clean shell:

```bash
conda deactivate 2>/dev/null || true
unset CONDA_PREFIX PYTHONPATH LD_LIBRARY_PATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

If you normally use Conda, the safest way is:

```bash
env -u CONDA_PREFIX -u PYTHONPATH -u LD_LIBRARY_PATH \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash
```

Then continue all following steps in that clean shell.

## 3. Install system dependencies

Update apt first:

```bash
sudo apt update
```

Install the required packages:

```bash
sudo apt install -y \
  build-essential \
  cmake \
  pkg-config \
  git \
  curl \
  wget \
  unzip \
  can-utils \
  libopencv-dev \
  libfmt-dev \
  libeigen3-dev \
  libspdlog-dev \
  libyaml-cpp-dev \
  nlohmann-json3-dev \
  libusb-1.0-0-dev \
  libceres-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  openssh-server \
  screen
```

Notes:
- `OpenCV / fmt / Eigen / spdlog / yaml-cpp / nlohmann-json / Ceres` are all needed by the current `CMakeLists.txt`
- `libusb-1.0-0-dev` is needed by the `io` layer
- `can-utils` is useful for CAN debugging, though not required for pure compile

## 4. Install OpenVINO 2024.6.0

The repository hardcodes this path in `CMakeLists.txt`:

```cmake
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")
```

So the easiest and most stable approach is to install OpenVINO exactly under:

```text
/opt/intel/openvino_2024.6.0
```

Recommended approach:
- use the official OpenVINO Linux archive installer for version `2024.6.0`
- install it into `/opt/intel`

After installation, verify these paths exist:

```bash
ls /opt/intel/openvino_2024.6.0/runtime/cmake
ls /opt/intel/openvino_2024.6.0/runtime/lib/intel64
```

If they do not exist, CMake will fail at `find_package(OpenVINO REQUIRED)`.

## 5. Camera SDK libraries

This project already contains the camera SDK headers and shared libraries inside the repository:

- `io/hikrobot/include`
- `io/hikrobot/lib/amd64/libMvCameraControl.so`
- `io/mindvision/include`
- `io/mindvision/lib/amd64/libMVSDK.so`

That means:
- you do not need to install HikRobot SDK again just for compilation
- you do not need to install MindVision SDK again just for compilation

The project links against these local `.so` files directly.

## 6. Optional: ROS2

ROS2 is optional for basic compilation.

The current project only compiles ROS2-related targets when all of the following can be found:

- `ament_cmake`
- `rclcpp`
- `std_msgs`
- `rosidl_typesupport_cpp`
- `sp_msgs`

If ROS2 or `sp_msgs` is missing:
- normal targets such as `QYG_hero`, `QYG_infantry`, `minimum_vision_system` can still be compiled
- ROS2 targets such as `sentry`, `publish_test`, `subscribe_test` will be skipped

So if your current goal is only "compile QYG_Vision successfully", you can skip ROS2 for now.

## 7. Configure the project

Enter the project directory:

```bash
cd /home/<your_name>/RM/QYG_Vision2.0
```

Use a clean shell to configure:

```bash
env -u CONDA_PREFIX -u PYTHONPATH -u LD_LIBRARY_PATH \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  cmake -S . -B build
```

Expected result:
- CMake configure finishes successfully
- if ROS2 is not complete, you may see a message like:

```text
ROS2 environment not found, skipping ROS2-related code.
```

This is acceptable for non-ROS targets.

## 8. Build targets

Build a minimal sanity-check target first:

```bash
env -u CONDA_PREFIX -u PYTHONPATH -u LD_LIBRARY_PATH \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  cmake --build build -j"$(nproc)" --target minimum_vision_system
```

Then build a main team target:

```bash
env -u CONDA_PREFIX -u PYTHONPATH -u LD_LIBRARY_PATH \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  cmake --build build -j"$(nproc)" --target QYG_hero
```

You can also build other main targets:

```bash
cmake --build build -j"$(nproc)" --target QYG_infantry
cmake --build build -j"$(nproc)" --target QYG_sentry
```

Note:
- `QYG_sentry` is not the ROS2 `sentry` target
- `QYG_sentry` is always present in the top-level CMake
- the ROS2-only target is `sentry`

## 9. Verify the binaries

Check that the binaries exist:

```bash
ls build/minimum_vision_system
ls build/QYG_hero
```

Check runtime dependencies:

```bash
ldd build/QYG_hero | rg 'MvCameraControl|MVSDK|openvino|not found'
```

If everything is correct, you should see:
- `libMvCameraControl.so` resolved to `io/hikrobot/lib/amd64/`
- `libMVSDK.so` resolved to `io/mindvision/lib/amd64/`
- `libopenvino.so` resolved to `/opt/intel/openvino_2024.6.0/runtime/lib/intel64/`
- no `not found`

## 10. Recommended first run

For a quick test, you can try a non-ROS executable:

```bash
./build/QYG_hero configs/QYG_hero.yaml
```

Or:

```bash
./build/QYG_infantry configs/QYG_infantry.yaml
```

Important:
- compilation success does not guarantee hardware can run under WSL directly
- camera, CAN, serial and USB devices may still require additional WSL passthrough setup
- this document focuses on "compile successfully in WSL"

## 11. Common problems

### Problem 1: `fmt` / `spdlog` mismatch

Symptom:

```text
error: 'basic_runtime' is not a member of 'fmt'
```

Cause:
- Conda headers are mixed into the build
- system `spdlog` and Conda `fmt` are not compatible

Fix:

```bash
conda deactivate 2>/dev/null || true
unset CONDA_PREFIX PYTHONPATH LD_LIBRARY_PATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
rm -rf build
cmake -S . -B build
cmake --build build -j"$(nproc)" --target QYG_hero
```

### Problem 2: OpenVINO not found

Symptom:

```text
Could not find a package configuration file provided by "OpenVINO"
```

Fix:
- install OpenVINO `2024.6.0`
- make sure this directory exists:

```bash
ls /opt/intel/openvino_2024.6.0/runtime/cmake
```

### Problem 3: Ceres not found

Symptom:

```text
Could not find Ceres
```

Fix:

```bash
sudo apt install -y libceres-dev libgoogle-glog-dev libgflags-dev
```

### Problem 4: ROS2 warnings during configure

Symptom:

```text
ROS2 environment not found, skipping ROS2-related code.
```

Meaning:
- this is only a warning for non-ROS targets
- you can still build `QYG_hero`, `QYG_infantry`, `minimum_vision_system`

Fix:
- no action needed if you are not compiling ROS2 targets
- if you need ROS2 targets, install ROS2 Humble and your custom `sp_msgs` package

## 12. Suggested workflow

Use the following sequence every time:

```bash
cd /home/<your_name>/RM/QYG_Vision2.0
conda deactivate 2>/dev/null || true
unset CONDA_PREFIX PYTHONPATH LD_LIBRARY_PATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

cmake -S . -B build
cmake --build build -j"$(nproc)" --target QYG_hero
```

If you only want a quick sanity check:

```bash
cmake --build build -j"$(nproc)" --target minimum_vision_system
```

## 13. Minimal success standard

You can consider the WSL environment ready when all of the following are true:

- `cmake -S . -B build` succeeds
- `cmake --build build --target minimum_vision_system` succeeds
- `cmake --build build --target QYG_hero` succeeds
- `ldd build/QYG_hero` shows no missing dynamic library

At that point, your WSL environment is sufficient for compiling the current `QYG_Vision2.0` project.
