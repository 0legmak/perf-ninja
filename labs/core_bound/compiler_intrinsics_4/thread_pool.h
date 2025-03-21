#include <condition_variable>
#include <future>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

class ThreadPool {
public:
  explicit ThreadPool(int thread_count) {
    for (auto i = 0; i < thread_count; ++i) {
      workers.emplace_back([this]() {
        while (true) {
          std::unique_lock lock(mutex);
          while (!is_stopped && tasks.empty()) {
            cond_var.wait(lock);
          }
          if (is_stopped && tasks.empty()) {
            break;
          }
          auto task = std::move(tasks.front());
          tasks.pop();
          lock.unlock();
          task();
        }
      });
    }
  }

  ~ThreadPool() {
    Stop();
  }

  template <class F, class... Args>
  auto enqueue(F&& f, Args... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
    using ResultType = typename std::invoke_result<F, Args...>::type;
    std::packaged_task<ResultType()> packaged_task(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    auto future = packaged_task.get_future();
    {
      std::lock_guard lock(mutex);
      tasks.push([packaged_task = std::move(packaged_task)]() mutable {
        packaged_task();
      });
    }
    cond_var.notify_one();
    return future;
  }

  void Stop() {
    {
      std::lock_guard lock(mutex);
      is_stopped = true;
    }
    cond_var.notify_all();
    for (auto& worker : workers) {
      worker.join();
    }
  }
private:
  std::vector<std::thread> workers;
  std::mutex mutex;
  std::condition_variable cond_var;
  std::queue<std::move_only_function<void()>> tasks;
  bool is_stopped = false;
};
