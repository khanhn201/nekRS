#include <occa/core/base.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/modes/dpcpp/utils.hpp>
#include <occa/internal/modes/dpcpp/device.hpp>
#include <occa/internal/modes/dpcpp/kernel.hpp>
#include <occa/internal/modes/dpcpp/buffer.hpp>
#include <occa/internal/modes/dpcpp/memory.hpp>
#include <occa/internal/modes/dpcpp/memoryPool.hpp>
#include <occa/internal/modes/dpcpp/stream.hpp>
#include <occa/internal/modes/dpcpp/streamTag.hpp>
#include <occa/types/primitive.hpp>
#include <occa/internal/lang/kernelMetadata.hpp>
#include <occa/internal/lang/modes/dpcpp.hpp>

namespace occa
{
  namespace dpcpp
  {
    device::device(const occa::json &properties_, 
                   const ::sycl::device& device_)
        : occa::launchedModeDevice_t(properties_), dpcppDevice(device_), dpcppContext(device_) 
    {
      occa::json &kernelProps = properties["kernel"];
      setCompilerLinkerOptions(kernelProps);
      arch = dpcppDevice.get_info<::sycl::info::device::name>();
    }

    hash_t device::hash() const
    {
      if (!hash_.initialized)
      {
        std::stringstream ss;
        auto p = dpcppDevice.get_platform();
        ss << "platform name: " << p.get_info<::sycl::info::platform::name>()
          << " platform vendor: " << p.get_info<::sycl::info::platform::vendor>()
          << " platform version: " << p.get_info<::sycl::info::platform::version>()
          << " device name: " << dpcppDevice.get_info<::sycl::info::device::name>()
          << " device vendor: " << dpcppDevice.get_info<::sycl::info::device::vendor>()
          << " device version: " << dpcppDevice.get_info<::sycl::info::device::version>();
        hash_ = occa::hash(ss.str());
      }
      return hash_;
    }

    hash_t device::kernelHash(const occa::json &props) const
    {
      return (
          occa::hash(props["compiler"])
          ^ props["compiler_flags"]
          ^ props["kernel/include_occa"]
          ^ props["kernel/link_occa"]
      );
    }

    lang::okl::withLauncher *device::createParser(const occa::json &props) const
    {
      return new lang::okl::dpcppParser(props);
    }

    //---[ Stream ]---------------------
    modeStream_t *device::createStream(const occa::json &props)
    {
      ::sycl::queue q(dpcppContext,
                      dpcppDevice,
                      {::sycl::property::queue::enable_profiling{},
                      ::sycl::property::queue::in_order{}
                      });
      return new occa::dpcpp::stream(this, props, q);
    }

    modeStream_t* device::wrapStream(void* ptr, const occa::json &props) {
      OCCA_ERROR("A nullptr was passed to dpcpp::device::wrapStream",nullptr != ptr);
      ::sycl::queue q = *static_cast<::sycl::queue*>(ptr);
      return new stream(this, props, q);
    }

    occa::streamTag device::tagStream()
    {
      return getDpcppStream(currentStream).tag();
    }

    void device::waitFor(occa::streamTag tag)
    {
      getDpcppStreamTag(tag).waitFor();
    }

    double device::timeBetween(const occa::streamTag &startTag,
                               const occa::streamTag &endTag)
    {
      auto& dpcppStartTag{getDpcppStreamTag(startTag)};
      auto& dpcppEndTag{getDpcppStreamTag(endTag)};

      dpcppEndTag.waitFor();

      return (dpcppEndTag.startTime() - dpcppStartTag.endTime());
    }


    //==================================

    //---[ Kernel ]---------------------
    modeKernel_t *device::buildKernelFromProcessedSource(
        const hash_t kernelHash,
        const std::string &hashDir,
        const std::string &kernelName,
        const std::string &sourceFilename,
        const std::string &binaryFilename,
        const bool usingOkl,
        lang::sourceMetadata_t &launcherMetadata,
        lang::sourceMetadata_t &deviceMetadata,
        const occa::json &kernelProps)
    {
      compileKernel(hashDir,
                    kernelName,
                    sourceFilename,
                    binaryFilename,
                    kernelProps);

      const bool compile_only = kernelProps.get("build/compile_only",false);
      if (compile_only) return nullptr;
      
      if (usingOkl)
      {
        return buildOKLKernelFromBinary(kernelHash,
                                        hashDir,
                                        kernelName,
                                        sourceFilename,
                                        binaryFilename,
                                        launcherMetadata,
                                        deviceMetadata,
                                        kernelProps);
      }
      else
      {
        void *kernel_dlhandle = sys::dlopen(binaryFilename);
        occa::functionPtr_t kernel_function = sys::dlsym(kernel_dlhandle, kernelName);

        return new dpcpp::kernel(this,
                                 kernelName,
                                 sourceFilename,
                                 kernel_dlhandle,
                                 kernel_function,
                                 kernelProps);
      }
    }

    void device::setArchCompilerFlags(std::string& compilerFlags)
    {
    }

    void device::compileKernel(const std::string &hashDir,
                               const std::string &kernelName,
                               const std::string &sourceFilename,
                               const std::string &binaryFilename,
                               const occa::json &kernelProps)
    {
      occa::json allProps = kernelProps;
      const bool verbose = allProps.get("verbose", false);

      const bool compilingOkl = allProps.get("okl/enabled", true);

      const std::string compiler = allProps["compiler"];
      std::string compilerFlags = allProps["compiler_flags"];
      std::string compilerSharedFlags = kernelProps["compiler_shared_flags"];
      std::string compilerLinkerFlags = kernelProps["compiler_linker_flags"];

      setArchCompilerFlags(compilerFlags);

      if (!compilingOkl)
      {
        sys::addCompilerIncludeFlags(compilerFlags);
        sys::addCompilerLibraryFlags(compilerFlags);
      }

      const bool includeOcca = kernelProps.get("kernel/include_occa", false);
      const bool linkOcca    = kernelProps.get("kernel/link_occa", false);
      if (includeOcca) {
        compilerFlags += " -I" + env::OCCA_DIR + "include";
        compilerFlags += " -I" + env::OCCA_INSTALL_DIR + "include";
      }
      if (linkOcca) {
        compilerLinkerFlags += " -L" + env::OCCA_INSTALL_DIR + "lib -locca";
      }

      std::stringstream command;
      if (allProps.has("compiler_env_script"))
      {
        command << allProps["compiler_env_script"] << " && ";
      }

      command << compiler
              << " " << compilerFlags
              << " " << compilerSharedFlags
              << " " << sourceFilename
              << " -o " << binaryFilename
              << " " << compilerLinkerFlags
              << std::endl;

      if (!verbose)
      {
        command << " > /dev/null 2>&1";
      }

      const std::string &sCommand = command.str();
      if (verbose) {
        io::stdout << "Compiling [" << kernelName << "]\n" << sCommand << "\n";
      }

      std::string commandOutput;
      const int commandExitCode = sys::call(
        sCommand.c_str(),
        commandOutput
      );

      if (commandExitCode) {
        OCCA_FORCE_ERROR(
          "Error compiling [" << kernelName << "],"
          " Command: [" << sCommand << "] exited with code " << commandExitCode << "\n\n"
          << "Output:\n\n"
          << commandOutput << "\n"
        );
      } else if (verbose) {
        io::stdout << "Output:\n\n" << commandOutput << "\n";
      }

      io::sync(binaryFilename);
    }

    modeKernel_t *device::buildOKLKernelFromBinary(const hash_t kernelHash,
                                                   const std::string &hashDir,
                                                   const std::string &kernelName,
                                                   const std::string &sourceFilename,
                                                   const std::string &binaryFilename,
                                                   lang::sourceMetadata_t &launcherMetadata,
                                                   lang::sourceMetadata_t &deviceMetadata,
                                                   const occa::json &kernelProps)
    {
      void *dl_handle = sys::dlopen(binaryFilename);

      dpcpp::kernel &k = *(new dpcpp::kernel(this,
                                             kernelName,
                                             sourceFilename,
                                             dl_handle,
                                             kernelProps));

      k.launcherKernel = buildLauncherKernel(kernelHash,
                                             hashDir,
                                             kernelName,
                                             launcherMetadata,
                                             kernelProps);
      // Find device kernels
      orderedKernelMetadata launchedKernelsMetadata = getLaunchedKernelsMetadata(
          kernelName,
          deviceMetadata);

      const int launchedKernelsCount = (int)launchedKernelsMetadata.size();
      for (int i = 0; i < launchedKernelsCount; ++i)
      {
        lang::kernelMetadata_t &metadata = launchedKernelsMetadata[i];
        auto &arguments{metadata.arguments};

        // The first two arguments are the sycl::queue
        // and nd_range: these must be removed from the
        // metadata for inline OKL to work.
        arguments.erase(arguments.begin());
        arguments.erase(arguments.begin());

        occa::functionPtr_t kernel_function = sys::dlsym(dl_handle, metadata.name);

        kernel *dpcppKernel = new dpcpp::kernel(this,
                               metadata.name,
                               sourceFilename,
                               kernel_function,
                               kernelProps);

        dpcppKernel->metadata = metadata;
        k.deviceKernels.push_back(dpcppKernel);
      }

      return &k;
    }

    modeKernel_t *device::buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::json &kernelProps)
    {
      void *kernel_dlhandle = sys::dlopen(filename);
      occa::functionPtr_t kernel_function = sys::dlsym(kernel_dlhandle, kernelName);

      return new dpcpp::kernel(this,
                               kernelName,
                               filename,
                               kernel_dlhandle,
                               kernel_function,
                               kernelProps);
    }

    int device::maxDims() const
    {
      // This is an OCCA restriction, not a SYCL one.
      return occa::dpcpp::max_dimensions;
    }

    dim device::maxOuterDims() const
    {
      // This is an OCCA restriction, not a SYCL one.
      return dim{occa::UDIM_DEFAULT, occa::UDIM_DEFAULT, occa::UDIM_DEFAULT};
    }

    dim device::maxInnerDims() const
    {
      ::sycl::id<3> max_wi_sizes{dpcppDevice.get_info<::sycl::info::device::max_work_item_sizes<3>>()};
      return dim{max_wi_sizes[occa::dpcpp::x_index],
                 max_wi_sizes[occa::dpcpp::y_index],
                 max_wi_sizes[occa::dpcpp::z_index]};
    }

    udim_t device::maxInnerSize() const
    {
      uint64_t max_wg_size{dpcppDevice.get_info<::sycl::info::device::max_work_group_size>()};
      return max_wg_size;
    }

    //==================================

    //---[ Memory ]---------------------
    modeMemory_t *device::malloc(const udim_t bytes,
                                 const void *src,
                                 const occa::json &props)
    {
      buffer *buf = new dpcpp::buffer(this, bytes, props);

      //create allocation
      buf->malloc(bytes);

      //create slice
      memory *mem = new dpcpp::memory(buf, bytes, 0);

      if (src != NULL) {
        mem->copyFrom(src, bytes, 0, props);
      }

      return mem;
    }

    modeMemory_t *device::wrapMemory(const void *ptr,
                                     const udim_t bytes,
                                     const occa::json &props)
    {
      //create allocation
      buffer *buf = new dpcpp::buffer(this, bytes, props);

      buf->wrapMemory(ptr, bytes);

      return new dpcpp::memory(buf, bytes, 0);
    }

    modeMemoryPool_t* device::createMemoryPool(const occa::json &props) {
      return new dpcpp::memoryPool(this, props);
    }

    udim_t device::memorySize() const
    {
      uint64_t global_mem_size{dpcppDevice.get_info<::sycl::info::device::global_mem_size>()};
      return global_mem_size;
    }

    void* device::unwrap() {
      return static_cast<void*>(&dpcppDevice);
    }
    //==================================
  } // namespace dpcpp
} // namespace occa
