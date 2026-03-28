#include <fmt/format.h>

#include "io/gimbal/gimbal.hpp"
#include "io/cboard.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/trajectory.hpp"

// 定义命令行参数
const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  // 初始化绘图器、录制器、退出器
  tools::Plotter plotter;
  tools::Recorder recorder;
  tools::Exiter exiter;

  // 初始化云台
  // io::Gimbal gimbal(config_path);
  io::CBoard cboard(config_path);
  io::VisionToGimbal plan;
  auto last_t = std::chrono::steady_clock::now();
  plan.yaw = 0;
  // plan.yaw_vel = 0;  // field removed
  // plan.yaw_acc = 0;  // field removed
  plan.pitch = 0;
  // plan.pitch_vel = 0;  // field removed
  // plan.pitch_acc = 0;  // field removed

  while (!exiter.exit()) {
    auto now = std::chrono::steady_clock::now();
    // auto gs = gimbal.state();
    auto imu_q = cboard.imu_at(now);
    // auto euler = tools::eulers(imu_q, 2, 1, 0);  // yaw pitch roll
    if(tools::delta_time(now, last_t) > 1.600) {
        plan.mode = 2;
        tools::logger()->debug("fire!");
        last_t = now;
    } else plan.mode = 1;


    // gimbal.send(plan);
    cboard.send({true, plan.mode == 2, plan.yaw, plan.pitch, 0});

    // -------------- 调试输出 --------------

    nlohmann::json data;

    if (plan.mode != 0) {
      data["shoot"] = plan.mode == 2 ? 1 : 0;
    }

    plotter.plot(data);

    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  return 0;
}
