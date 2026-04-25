#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>

namespace auto_aim
{
namespace multithread
{

/**
 * @brief MultiThreadDetector 构造函数
 * 逻辑：初始化 OpenVINO 异步推理引擎。
 * 通过配置 THROUGHPUT 模式，利用多推理请求（InferRequests）来优化吞吐量。
 */
MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();
  auto model_path = yaml[yolo_name + "_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();

  // 1. 读取并预处理模型
  auto model = core_.read_model(model_path);
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  // 配置硬件预处理流水线
  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3}) // 统一输入尺寸
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  model = ppp.build();
  
  // 2. 编译模型：开启 THROUGHPUT 模式以分发到多个推理请求
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));

  tools::logger()->info("[MultiThreadDetector] initialized !");
}

/**
 * @brief 推送检测任务
 * 逻辑：在主线程（捕获图像线程）执行预处理（如 Resize），
 * 然后调起异步推理（start_async），将请求句柄压入队列。
 */
void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  // 1. 软件预处理 (Resize 并保持长宽比)
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(img, input(roi), {w, h});

  // 2. 获取并启动异步推理请求
  auto infer_request = compiled_model_.create_infer_request();
  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);
  infer_request.set_input_tensor(input_tensor);
  
  // start_async 会立即返回，不阻塞主线程
  infer_request.start_async();

  // 3. 将任务上下文存入线程安全队列
  queue_.push({img.clone(), t, std::move(infer_request)});
}

/**
 * @brief 弹出检测结果
 * 逻辑：从队列取出最老的一个请求，调用 wait() 等待硬件计算结束，随后执行后处理解析。
 */
std::list<Armor> MultiThreadDetector::pop_logic(
  cv::Mat & img, std::chrono::steady_clock::time_point & t, ov::InferRequest & infer_request)
{
  // 阻塞直到该请求计算完成
  infer_request.wait();

  // 1. 获取张量输出并执行转换
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  
  // 2. 比例还原参数
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  
  // 3. 转发至具体版本的 YOLO 后处理解析器
  return yolo_.postprocess(scale, output, img, 0);
}

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  auto [img, t, infer_request] = queue_.pop();
  auto armors = pop_logic(img, t, infer_request);
  return {std::move(armors), t};
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  auto [img, t, infer_request] = queue_.pop();
  auto armors = pop_logic(img, t, infer_request);
  return {img, std::move(armors), t};
}

}  // namespace multithread
}  // namespace auto_aim
