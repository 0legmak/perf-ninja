#include "thread_pool.h"

ThreadPool& get_thread_pool() {
  static ThreadPool thread_pool(std::thread::hardware_concurrency());
  return thread_pool;
}
