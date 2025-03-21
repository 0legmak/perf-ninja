#include "wait_for_debugger.h"

#include <iostream>

#ifdef _WIN32

#include <windows.h>

void WaitForDebugger() {
  std::cout << "Waiting for debugger to attach..." << std::endl;
  while (!IsDebuggerPresent()) {
    Sleep(100); // Check every 100 milliseconds
  }
  //DebugBreak(); // Trigger a breakpoint once debugger is attached
}

#endif  // _WIN32
