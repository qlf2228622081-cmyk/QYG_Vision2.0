#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>

namespace auto_aim
{
namespace multithread
{

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();
  auto model_path = yaml[yolo_name + "_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();

  auto model = core_.read_model(model_path);
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3})  // TODO
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    // .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
    .scale(255.0);

  model = ppp.build();
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));

  tools::logger()->info("[MultiThreadDetector] initialized !");
}

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  // preproces
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(img, input(roi), {w, h});

  auto input_port = compiled_model_.input();
  auto infer_request = compiled_model_.create_infer_request();
  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  infer_request.set_input_tensor(input_tensor);
  infer_request.start_async();
  queue_.push({img.clone(), t, std::move(infer_request)});
}

/*
push函数负责将图像数据和时间戳打包成一个任务，并将其放入线程安全队列中。它首先根据输入图像的尺寸计算缩放比例，然后将图像缩放到640x640的输入尺寸。
接着，它创建一个OpenVINO的推理请求，将预处理后的图像数据设置为输入张量，并异步启动推理。
最后，它将原始图像、时间戳和推理请求一起推入队列中，以供后续处理线程使用。


总结就是：接收一帧图像，完成送入检测模型前的预处理
启动一次异步推理，然后把“原图副本 + 时间戳 + 这次推理请求”放进内部队列
供后续 pop()/debug_pop() 取结果。
*/

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  auto [img, t, infer_request] = queue_.pop();
  infer_request.wait();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  return {std::move(armors), t};
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  auto [img, t, infer_request] = queue_.pop();
  infer_request.wait();

  // postprocess
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);  //暂不支持ROI

  return {img, std::move(armors), t};
}

}  // namespace multithread

}  // namespace auto_aim
