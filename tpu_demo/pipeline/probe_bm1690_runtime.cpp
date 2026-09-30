// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
//
// Compile against the installed board runtime header, not the SDK emulator.
// This initializes the runtime and queries device contexts. It never allocates
// TPU buffers, loads a TPU module, launches a kernel, or synchronizes workloads.
#include <tpuv7_rt.h>

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

namespace {

std::string canonical_path(const std::string &path) {
  char *resolved = realpath(path.c_str(), nullptr);
  if (!resolved) return {};
  std::string result(resolved);
  free(resolved);
  return result;
}

std::string link_target(const std::string &path) {
  char buffer[PATH_MAX + 1];
  const ssize_t size = readlink(path.c_str(), buffer, sizeof(buffer) - 1);
  if (size < 0) return {};
  return std::string(buffer, static_cast<size_t>(size));
}

template <typename Function>
Function symbol(void *library, const char *name) {
  dlerror();
  void *address = dlsym(library, name);
  const char *error = dlerror();
  if (error || !address)
    throw std::runtime_error(std::string("missing runtime symbol ") + name +
                             ": " + (error ? error : "null address"));
  return reinterpret_cast<Function>(address);
}

template <typename Function, typename... Args>
void checked(const char *name, Function function, Args... args) {
  std::cout << "CALL " << name << std::endl;
  const auto status = function(args...);
  std::cout << "RETURN " << name << " rc=" << static_cast<int>(status)
            << std::endl;
  if (status != tpuRtSuccess)
    throw std::runtime_error(std::string(name) + " failed; no further queries");
}

void describe_fd(int fd) {
  if (fd < 0) throw std::runtime_error("runtime returned an invalid fd");
  struct stat info {};
  if (fstat(fd, &info) != 0)
    throw std::runtime_error(std::string("fstat runtime fd: ") + strerror(errno));
  std::cout << "FD fd=" << fd << " target="
            << std::quoted(link_target("/proc/self/fd/" + std::to_string(fd)))
            << " character_device=" << S_ISCHR(info.st_mode) << std::endl;
  if (S_ISCHR(info.st_mode)) {
    const std::string node = "/sys/dev/char/" +
        std::to_string(major(info.st_rdev)) + ":" +
        std::to_string(minor(info.st_rdev));
    std::cout << "FD_SYSFS node=" << std::quoted(node)
              << " path=" << std::quoted(canonical_path(node))
              << " device_path=" << std::quoted(canonical_path(node + "/device"))
              << std::endl;
  }
  // GetFd returns a borrowed descriptor. The runtime owns its lifetime.
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3 || argv[1][0] != '/')
      throw std::runtime_error(
          "usage: probe /absolute/path/libtpuv7_rt.so max_devices");
    char *end = nullptr;
    errno = 0;
    const long limit = strtol(argv[2], &end, 10);
    if (errno || end == argv[2] || *end || limit < 1 || limit > INT_MAX)
      throw std::runtime_error("max_devices must be a positive integer");

    const std::string requested = canonical_path(argv[1]);
    if (requested.empty()) throw std::runtime_error("runtime library not found");
    std::cout << "RUNTIME_REQUESTED=" << std::quoted(requested)
              << " QUERY_LIMIT=" << limit
              << " PROPERTIES_SIZE=" << sizeof(tpuRtDeviceProperties_t)
              << std::endl;
    void *library = dlopen(requested.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(std::string("dlopen: ") + dlerror());
    // Resolve everything before initialization. Leave the library resident:
    // the runtime may own background threads and has no teardown API here.
    const auto init = symbol<decltype(&tpuRtInit)>(library, "tpuRtInit");
    const auto count_devices =
        symbol<decltype(&tpuRtGetDeviceCount)>(library, "tpuRtGetDeviceCount");
    const auto set_device =
        symbol<decltype(&tpuRtSetDevice)>(library, "tpuRtSetDevice");
    const auto get_device =
        symbol<decltype(&tpuRtGetDevice)>(library, "tpuRtGetDevice");
    const auto properties =
        symbol<decltype(&tpuRtGetDeviceProperties)>(library, "tpuRtGetDeviceProperties");
    const auto get_fd = symbol<decltype(&tpuRtGetFd)>(library, "tpuRtGetFd");

    Dl_info identity {};
    if (!dladdr(reinterpret_cast<void *>(init), &identity) || !identity.dli_fname ||
        canonical_path(identity.dli_fname) != requested)
      throw std::runtime_error("tpuRtInit resolved outside the requested library");
    std::cout << "RUNTIME_LOADED="
              << std::quoted(canonical_path(identity.dli_fname)) << std::endl;
    checked("tpuRtInit", init);
    int count = -1;
    checked("tpuRtGetDeviceCount", count_devices, &count);
    std::cout << "DEVICE_COUNT=" << count << std::endl;
    if (count <= 0 || count > limit)
      throw std::runtime_error("device count outside query limit; no device selected");

    // These are candidate ordinal IDs, not pre-assigned physical chip IDs.
    // Stop on the first rejected ID and preserve all preceding output.
    for (int id = 0; id < count; ++id) {
      std::cout << "DEVICE_BEGIN candidate_id=" << id << std::endl;
      checked("tpuRtSetDevice", set_device, id);
      int selected = -1;
      checked("tpuRtGetDevice", get_device, &selected);
      std::cout << "SELECTED_ID=" << selected << std::endl;
      if (selected != id)
        throw std::runtime_error("SetDevice/GetDevice disagree");
      tpuRtDeviceProperties_t props {};
      checked("tpuRtGetDeviceProperties", properties, &props, id);
      const std::string name(props.name, strnlen(props.name, sizeof(props.name)));
      std::cout << "PROPERTIES name=" << std::quoted(name)
                << " total_global_mem_bytes=" << props.totalGlobalMem
                << " major=" << props.major << " minor=" << props.minor
                << " ecc_enabled=" << props.ECCEnabled << std::endl;
      // Do not guess whether pciDeviceID encodes a slot, function or product ID.
      std::cout << "PCI_FIELDS_RAW domain=" << props.pciDomainID
                << " bus=" << props.pciBusID << " device=" << props.pciDeviceID
                << std::endl;
      int fd = -1;
      checked("tpuRtGetFd", get_fd, &fd);
      describe_fd(fd);
      std::cout << "DEVICE_END candidate_id=" << id << std::endl;
    }
    std::cout << "ENUMERATION_COMPLETE; inspect PCI/sysfs mapping before choosing "
                 "a device for kernels" << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "PROBE_ERROR: " << error.what() << std::endl;
    return 1;
  }
}
