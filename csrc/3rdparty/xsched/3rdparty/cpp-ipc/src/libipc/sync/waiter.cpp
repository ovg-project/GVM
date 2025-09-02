#include "libipc/waiter.h"

#include "libipc/platform/detail.h"
#if defined(IPC_OS_WINDOWS_)
#include "libipc/platform/win/mutex.h"
#elif defined(IPC_OS_LINUX_)
    #if defined(LIBIPC_USE_POSIX_ON_LINUX__)
        #include "libipc/platform/posix/mutex.h"
    #else
        #include "libipc/platform/linux/mutex.h"
    #endif
#elif defined(IPC_OS_QNX_)
#include "libipc/platform/posix/mutex.h"
#else/*IPC_OS*/
#   error "Unsupported platform."
#endif

namespace ipc {
namespace detail {

void waiter::init() {
    ipc::detail::sync::mutex::init();
}

} // namespace detail
} // namespace ipc
