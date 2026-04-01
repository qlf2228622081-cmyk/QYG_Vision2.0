#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <string>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/thread_safe_queue.hpp"

/*
把一帧里的 armors（检测结果）变成一个持续、稳定、可预测的 target（跟踪目标）
Tracker 负责把 armors 转换成 targets，核心是一个状态机，状态包括 lost、detecting、tracking、temp_lost、switching
lost -> detecting -> tracking -> temp_lost -> lost
lost -> switching -> detecting -> tracking -> temp_lost -> lost
lost -> switching -> lost
其中 switching 是 lost 和 tracking 之间的过渡状态，主要用于切换目标时的过渡，防止目标丢失过快；temp_lost 是 tracking 和 lost 之间的过渡状态，主要用于目标暂时丢失时的处理。

所以它不是检测器，也不是瞄准器，而是中间那层“目标管理器”。
*/


namespace auto_aim
{
class Tracker
{
public:
  Tracker(const std::string & config_path, Solver & solver);
  Tracker(const std::string & config_path, Solver & solver,std::string enemy_color);
  void set_enemy_color(const std::string & enemy_color);

  std::string state() const;

  /*
  返回一个Target链表
  参数1：参数 1：std::list<Armor> & armors,也就是“传进来一个 Armor 链表，并且是引用”
  参数2：time_point t也就是“传进来一个时间戳”
  参数3：bool use_enemy_color = true 也就是“传进来一个布尔值，默认值是true”
  */
  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, bool use_enemy_color = true);

private:
  Solver & solver_;
  Color enemy_color_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  std::string state_, pre_state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  ArmorPriority omni_target_priority_;

  void state_machine(bool found);

  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  bool update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP
