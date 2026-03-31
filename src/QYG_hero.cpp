#include <chrono>
#include <functional>
#include <optional>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
// #include "tasks/auto_buff/buff_aimer.hpp"
// #include "tasks/auto_buff/buff_detector.hpp"
// #include "tasks/auto_buff/buff_solver.hpp"
// #include "tasks/auto_buff/buff_target.hpp"
// #include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/QYG_hero.yaml | 位置参数,yaml配置文件路径 }";

using namespace std::chrono_literals;

struct AimTask
{
  std::optional<auto_aim::Target> target;
  std::chrono::steady_clock::time_point timestamp;
};
"""
所以 target_queue 里每一项都不是图片、也不是 armors，而是：
一个可选目标 target
一个对应时间戳 timestamp
"""


int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  """
  创建类的实例对象，也就是定义对象变量，并调用构造函数进行初始化
  如：io::Cboard cboard(config_path);
  意思就是：
  类型是 io::CBoard
  变量名是 cboard
  用参数 config_path 调它的构造函数，创建一个对象
  """

  tools::Exiter exiter;
  tools::Recorder recorder;

  io::CBoard cboard(config_path);//和主控板通信对象
  io::Camera camera(config_path);//相机对象

  auto_aim::multithread::MultiThreadDetector detector(config_path,true);//装甲板检测
"""
这表示创建了一个 detector 对象。
而 detector 这个对象内部就自带一个成员：
detector.queue_

只是因为 queue_ 是 private，你不能在 QYG_hero.cpp 里直接写 detector.queue_ 去访问它。
所以你要这样理解：
target_queue：主程序自己显式创建的队列
queue_：detector 模块自己内部维护的队列
"""

  auto_aim::Solver solver(config_path);//姿态和坐标解算
  auto_aim::Tracker tracker(config_path, solver);//目标跟踪
  auto_aim::Aimer aimer(config_path);//瞄准解算
  auto_aim::Shooter shooter(config_path);//决定是否开火
  io::Command command;

"""
你这段里每一行几乎都可以这么读：
tools::Exiter exiter;
创建一个 Exiter 对象 exiter
tools::Recorder recorder;
创建一个 Recorder 对象 recorder
io::CBoard cboard(config_path);
创建一个和主控板通信的对象
io::Camera camera(config_path);
创建一个相机对象
auto_aim::multithread::MultiThreadDetector detector(config_path, true);
创建一个多线程检测器对象
auto_aim::Solver solver(config_path);
创建一个解算器对象
auto_aim::Tracker tracker(config_path, solver);
创建一个跟踪器对象
auto_aim::Aimer aimer(config_path);
创建一个瞄准器对象
auto_aim::Shooter shooter(config_path);
创建一个开火判定对象
io::Command command;
创建一个命令对象
你可以把它们理解成：程序一开始先把这一套“工具人”都实例化出来，后面主循环里反复调用它们的方法。
"""

"""
tools::ThreadSafeQueue<AimTask, true> target_queue(1);
这句非常关键，能读出 3 件事：

元素类型是 AimTask
true 表示“队列满了就弹掉旧数据再塞新数据”
(1) 表示队列容量只有 1
所以它不是普通“排长队”的队列，而是一个 只保留最新任务的单槽邮箱。

它具体怎么工作
生产者是主线程：


"""



  tools::ThreadSafeQueue<AimTask, true> target_queue(1);
  target_queue.push({std::nullopt, std::chrono::steady_clock::now()});

  // auto_buff::Buff_Detector buff_detector(config_path);
  // auto_buff::Solver buff_solver(config_path);
  // auto_buff::SmallTarget buff_small_target;
  // auto_buff::BigTarget buff_big_target;
  // auto_buff::Aimer buff_aimer(config_path);

  std::atomic<bool> quit = false;

  std::atomic<io::Mode> mode{io::Mode::idle};
  std::atomic<io::Mode> last_mode{io::Mode::idle};
  int idle_counter = 0;  // idle模式下的计数器，用于降低发送频率
  constexpr int send_repeat_count = 10;  // 每个命令重复发送次数，提高数据到达率

  // 检测线程：异步进行图像采集和检测（负责找目标）
  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!quit && !exiter.exit()) {
      //仅在自瞄模式下持续读取图片并且向检测器投喂新的帧
      if (mode.load() == io::Mode::auto_aim) {
        //从相机封装层取出一帧图像及其时间戳
        //底层相机类会根据yaml配置自动选择具体的相机类型进行初始化和读取
        camera.read(img, t);
        //对当前帧图像做预处理，进行检测，检测是一次异步推理
        //同时将原图、时间戳和推理请求压入检测器内部队列
        //结果会被放入一个线程安全的队列中，供瞄准线程使用
        """
        推理是把图像送进神经网络得到原始的输出tensor，之后会进行后处理得到装甲板列表
        由于推理可能比较耗时，所以采用异步的方式进行，避免阻塞主线程。
        主线程会持续读取相机帧并投喂给检测器，检测器内部会有一个线程池专门处理这些帧的推理任务，
        推理完成后会把结果放入一个线程安全的队列中，供瞄准线程使用。
        """
        
        detector.push(img, t);  // 异步检测
      } else
        std::this_thread::sleep_for(10ms);
    }
  });
  //检测线程结束


  // 瞄准线程：使用Aimer进行瞄准（负责打目标）
  auto plan_thread = std::thread([&]() {
    while (!quit && !exiter.exit()) {
      if (mode.load() == io::Mode::auto_aim && !target_queue.empty()) {
        auto task = target_queue.pop();
        std::list<auto_aim::Target> targets;
        if (task.target.has_value()) {
          targets.push_back(task.target.value());
        }

        auto plan = aimer.aim(targets, task.timestamp, cboard.bullet_speed, true);
        const Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
        plan.shoot = shooter.shoot(plan, aimer, targets, gimbal_pos);

        cboard.send(plan);

        std::this_thread::sleep_for(10ms);
      } else {
        std::this_thread::sleep_for(10ms);  // 减少等待时间，提高响应速度
      }
    }
  });
  //瞄准线程结束
  
  
  //这是主循环，负责模式切换和发送命令
  while (!exiter.exit()) {
    mode = cboard.mode;
    auto current_mode = mode.load();

    if (last_mode != current_mode) {
      tools::logger()->info("Switch to {}", io::MODES[current_mode]);
      last_mode.store(current_mode);
    }

    /// 自瞄
    if (current_mode == io::Mode::auto_aim) {
      // 从检测队列获取结果（异步检测已完成）
      //push()是检测线程调用的，负责把相机帧送入检测器进行异步推理；
      //debug_pop()是瞄准线程调用的，负责从检测器获取最新的检测结果
      //检测线程已经结束，我们获得了img，armors，t
      //下一步：检测结果 armors -> 跟踪目标 targets
      auto [img, armors, t] = detector.debug_pop();

      //下面四行合起来的意义是：
      //把“这一帧图像上的检测结果 armors”放到正确的时间和姿态背景里
      //最终变成可持续跟踪的目标 targets。
      
      //1. 从主控板获取当前时间戳对应的IMU数据，得到当前云台姿态 q
      //拿时间戳t这一刻的IMU的四元数，还会用前后两帧的IMU做插值
      //目的是为了让图像和姿态对齐
      auto q = cboard.imu_at(t);
      //2.把图像姿态时间戳按照设定的频率记录下来，保存成视频和文本（主要用于调试）
      recorder.record(img, q, t); 
      //3.它用IMU的四元数q去更新R_gimbal2world，这样就把检测结果放在了正确的姿态背景里
      //把云台坐标系和世界坐标系之间当前怎么旋转告诉solve解算器
      solver.set_R_gimbal2world(q);

      //4.结合当前帧检测结果和历史状态，完成目标解算与跟踪，得到稳定目标
      auto targets = tracker.track(armors, t);//把检测结果变成稳定的跟踪目标
      if (!targets.empty()) {
        target_queue.push({targets.front(), t});
      } else {
        target_queue.push({std::nullopt, t});
      }
    }

    // /// 打符
    // else if (mode.load() == io::Mode::small_buff || mode.load() == io::Mode::big_buff) {
    //   buff_solver.set_R_gimbal2world(q);

    //   auto power_runes = buff_detector.detect(img);

    //   buff_solver.solve(power_runes);

    //   auto_aim::Plan buff_plan;
    //   if (mode.load() == io::Mode::small_buff) {
    //     buff_small_target.get_target(power_runes, t);
    //     auto target_copy = buff_small_target;
    //     buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
    //   } else if (mode.load() == io::Mode::big_buff) {
    //     buff_big_target.get_target(power_runes, t);
    //     auto target_copy = buff_big_target;
    //     buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
    //   }
    //   cboard.send(io::command(plan.control, plan.fire,plan.yaw,plan.pitch,plan.horizon_distance));
    // }
    
    else if (current_mode == io::Mode::idle) {
      command = {false, false, 0, 0, 0};
      // idle模式下降低发送频率（每10次循环发送一次，约500ms）
      if (++idle_counter >= 10) {
        // 发送多次以提高数据到达率（中间可能有损耗）
        for (int i = 0; i < send_repeat_count; ++i) {
          cboard.send(command);
        }
        idle_counter = 0;
      }
      std::this_thread::sleep_for(50ms);  // 降低CPU占用
    }

  }
  //主循环结束

  quit = true;
  if (detect_thread.joinable()) detect_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  command = {false, false, 0, 0, 0};
  cboard.send(command);  // 退出前发送停止命令

  return 0;
}