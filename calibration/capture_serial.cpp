#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ?  |                          | 输出命令行参数说明}"
  "{@config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{output-folder o |      assets/img_with_q   | 输出文件夹路径   }";

struct CaptureData
{
  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point timestamp;
};

void write_q(const std::string & q_path, const Eigen::Quaterniond & q)
{
  try {
    std::ofstream q_file(q_path);
    if (!q_file.is_open()) {
      tools::logger()->error("Failed to open file: {}", q_path);
      return;
    }
    Eigen::Vector4d xyzw = q.coeffs();
    // 输出顺序为wxyz
    q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
    q_file.close();
  } catch (const std::exception & e) {
    tools::logger()->error("Failed to write quaternion file {}: {}", q_path, e.what());
  }
}

void capture_loop(
  const std::string & config_path, const std::string & output_folder)
{
  // 读取标定板参数（这里只是为了和原版保持一致，当前并未直接使用）
  auto yaml = YAML::LoadFile(config_path);
  (void)yaml;

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  std::atomic<bool> quit(false);
  std::atomic<int> save_count(0);

  // 线程安全的数据队列
  std::queue<CaptureData> data_queue;
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  const size_t max_queue_size = 10;  // 限制队列大小，避免内存占用过大

  // 采集线程：从相机和串口云台读取数据
  std::thread capture_thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;

    while (!quit) {
      try {
        camera.read(img, timestamp);
        if (img.empty()) {
          tools::logger()->warn("Empty image received, skipping...");
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          continue;
        }

        // 使用串口云台的姿态（四元数）
        Eigen::Quaterniond q = gimbal.q(timestamp);

        // 添加到队列
        {
          std::lock_guard<std::mutex> lock(queue_mutex);
          // 如果队列满了，丢弃最旧的数据
          if (data_queue.size() >= max_queue_size) {
            data_queue.pop();
          }
          data_queue.push({img.clone(), q, timestamp});
          queue_cv.notify_one();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));  // 控制采集频率
      } catch (const std::exception & e) {
        tools::logger()->error("Error in capture thread: {}", e.what());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  });

  // 显示和保存线程：处理GUI和文件I/O
  std::thread display_thread([&]() {
    cv::Mat current_img;
    Eigen::Quaterniond current_q;
    std::atomic<bool> has_data(false);

    while (!quit) {
      // 从队列获取最新数据
      {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_cv.wait(lock, [&] { return !data_queue.empty() || quit; });

        if (quit) break;

        // 清空队列，只保留最新的
        while (data_queue.size() > 1) {
          data_queue.pop();
        }

        if (!data_queue.empty()) {
          current_img = data_queue.front().img.clone();
          current_q = data_queue.front().q;
          data_queue.pop();
          has_data = true;
        }
      }

      if (!has_data || current_img.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      try {
        // 在图像上显示欧拉角，用来判断云台姿态
        auto img_with_ypr = current_img.clone();
        Eigen::Vector3d zyx = tools::eulers(current_q, 2, 1, 0) * 57.3;  // degree
        tools::draw_text(img_with_ypr, fmt::format("Z {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
        tools::draw_text(img_with_ypr, fmt::format("Y {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
        tools::draw_text(img_with_ypr, fmt::format("X {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});
        tools::draw_text(
          img_with_ypr, fmt::format("Saved: {}", save_count.load()), {40, 160}, {0, 255, 0});

        cv::resize(img_with_ypr, img_with_ypr, {}, 0.5, 0.5);  // 显示时缩小图片尺寸

        // GUI
        cv::imshow("Press s to save (serial), q to quit", img_with_ypr);
        auto key = cv::waitKey(1);

        if (key == 'q' || key == 'Q') {
          quit = true;
          break;
        }

        if (key == 's' || key == 'S') {
          // 保存图片和四元数
          int count = save_count.fetch_add(1) + 1;
          auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
          auto q_path = fmt::format("{}/{}.txt", output_folder, count);

          // 异步保存文件
          std::thread save_thread([=, img = current_img.clone(), q = current_q]() {
            try {
              cv::imwrite(img_path, img);
              write_q(q_path, q);
              tools::logger()->info("[{}] Saved (serial) in {}", count, output_folder);
            } catch (const std::exception & e) {
              tools::logger()->error("Failed to save file {}: {}", img_path, e.what());
            }
          });
          save_thread.detach();
        }
      } catch (const std::exception & e) {
        tools::logger()->error("Error in display thread: {}", e.what());
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(16));  // ~60 FPS
    }
  });

  // 等待线程结束，离开该作用域时，camera和gimbal会自动关闭
  capture_thread.join();
  display_thread.join();
}

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto output_folder = cli.get<std::string>("output-folder");

  // 新建输出文件夹
  std::filesystem::create_directories(output_folder);

  // 主循环，保存图片和对应四元数（通过串口云台）
  capture_loop(config_path, output_folder);

  tools::logger()->warn("注意四元数输出顺序为wxyz（串口 Gimbal）");

  return 0;
}


