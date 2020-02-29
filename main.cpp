#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <exception>
#include <string>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>

static void print_tensor_shape(const torch::Tensor &t)
{
  using namespace std;

  vector<int64_t> shape;

  for (int i = 0; i < t.dim(); ++i) {
    shape.push_back(t.size(i));
  }
  cout << "Tensor " << t.name() << " shape: {" << shape << "}" << endl;
}

static torch::Tensor area_of(const torch::Tensor &top_left,
                             const torch::Tensor &bottom_right)
{
  torch::Tensor hw = bottom_right - top_left;
  hw.clamp_min(torch::Scalar(0.f));

  return hw.select(-1, 0) * hw.select(-1, 1);
}

static torch::Tensor iou_of(const torch::Tensor &boxes0,
                            const torch::Tensor &boxes1,
                            float eps = 1.e-5f)
{
  torch::Tensor overlap_top_left = torch::max(boxes0.slice(-1, 0, 2),
                                              boxes1.slice(-1, 0, 2));
  torch::Tensor overlap_bottom_right = torch::min(boxes0.slice(-1, 2),
                                                  boxes1.slice(-1, 2));
  torch::Tensor overlap_area = area_of(overlap_top_left, overlap_bottom_right);
  torch::Tensor area0 = area_of(boxes0.slice(-1, 0, 2), boxes0.slice(-1, 2));
  torch::Tensor area1 = area_of(boxes1.slice(-1, 0, 2), boxes1.slice(-1, 2));

  return overlap_area / (area0 + area1 - overlap_area + torch::Scalar(eps));
}

torch::Tensor hard_nms(torch::Tensor t, float iou_threshold, int top_k,
                       int candidates)
{
  using namespace std;

  torch::Tensor scores = t.select(1, -1);
  torch::Tensor boxes = t.slice(1, 0, -1);

  torch::Tensor indices = scores.argsort(-1, true).slice(0, 0, candidates);

  vector<int64_t> picked;
  while (0 < indices.numel()) {
    torch::Tensor current = indices.select(0, 0);
    picked.push_back(current.item().toLong());
    if ((0 < top_k && static_cast<int>(picked.size())) ||
        1 == indices.numel()) break;
    torch::Tensor current_box = boxes.select(0, current.item().toLong());
    indices = indices.narrow(0, 1, indices.numel() - 1);
    torch::Tensor remained = boxes.index_select(0, indices);

    torch::Tensor iou = iou_of(remained, current_box.unsqueeze(0));
    indices = indices.masked_select(iou <= iou_threshold);
  }

  torch::Tensor t_picked =
    torch::from_blob(picked.data(), {static_cast<int64_t>(picked.size())},
                     torch::TensorOptions(torch::ScalarType::Long));

  return t.index_select(0, t_picked);
}

int main(int argc, const char* argv[])
{
  using namespace std;
  using namespace std::chrono;
  using namespace boost;
  using namespace boost::program_options;

  options_description desc(
    "\nA demo application running SSD inference with PyTorch TorchScript.\n"
    "\nAllowed arguments");

  // Define command line arguments using either formats:
  //
  //     (“long-name,short-name”, “Description of argument”)
  //     for flag values or
  //
  //     (“long-name,short-name”, <data-type>, 
  //     “Description of argument”) arguments with values
  //
  // Remember that arguments with values may be multi-values
  // and must be vectors
  desc.add_options()
    ("help,h", "Print help message.")
    ("torch-script,s", value<string>(), "Path to the TorchScript file.")
    ("labels,l", value<string>(), "Path to the object label file.")
    ("network-resolution,r", value<int>(), "Network input image size, "
     "default: 300")
    ("candidate-size,c", value<int>(), "Candidate size, default: 200")
    ("top-k,k", value<int>(), "Top-K candidates, default: -1")
    ("probability-threshold,p", value<float>(), "object score threshold, "
     "default: 0.01")
    ("iou-threshold,u", value<float>(), "IOU threshold, default: 0.45")
    ("output-file,o", value<string>(), "Path to the output image, "
     "default: detected.jpg")
    ("input-file,i", value<string>(), "Path to the input image.");

  // Map positional parameters to their tag valued types 
  // (e.g. –input-file parameters)
  positional_options_description p;
  p.add("input-file", -1);

  // Parse the command line catching and displaying any 
  // parser errors
  variables_map vm;
  try {
    store(command_line_parser(argc, argv)
          .options(desc).positional(p).run(), vm);
    notify(vm);
  } catch (std::exception &e) {
    cout << endl << e.what() << endl;
    cout << desc << endl;
  }

  // Display help text when requested
  if (vm.count("help")) {
    cout << "–help specified" << endl;
    cout << desc << endl;
    return EXIT_SUCCESS;
  }

  // Display the state of the arguments supplied
  string ts_filename;
  if (vm.count("torch-script")) {
    ts_filename = vm["torch-script"].as<string>();
    cout << "–torch-script specified with value = " << ts_filename << endl;
  }

  string label_filename("voc-model-labels.txt");
  if (vm.count("labels")) {
    label_filename = vm["labels"].as<string>();
    cout << "–labels specified with value = " << label_filename << endl;
  }

  int network_resolution = 300;
  if (vm.count("network-resolution")) {
    network_resolution = vm["network-resolution"].as<int>();
    cout << "–network-resolution specified with value = "
         << network_resolution << endl;
  }

  int candidate_size = 200;
  if (vm.count("candidate-size")) {
    candidate_size = vm["candidate-size"].as<int>();
    cout << "–candidate-size specified with value = "
         << candidate_size << endl;
  }

  int top_k = -1;
  if (vm.count("top-k")) {
    top_k = vm["top-k"].as<int>();
    cout << "–top-k specified with value = "
         << top_k << endl;
  }

  float probability_threshold = 0.01f;
  if (vm.count("probability-threshold")) {
    probability_threshold = vm["probability-threshold"].as<float>();
    cout << "–probability-threshold specified with value = "
         << probability_threshold << endl;
  }

  float iou_threshold = 0.45f;
  if (vm.count("iou-threshold")) {
    iou_threshold = vm["iou-threshold"].as<float>();
    cout << "–iou-threshold specified with value = "
         << iou_threshold << endl;
  }

  string out_filename("detected.jpg");
  if (vm.count("output-file")) {
    out_filename = vm["output-file"].as<string>();
    cout << "–output-file specified with value = " << out_filename << endl;
  }

  string in_filename;
  if (vm.count("input-file")) {
    in_filename = vm["input-file"].as<string>();
    cout << "–input-file specified with value = " << in_filename << endl;
  }

  torch::DeviceType device_type;

  cout << "cuDNN: "
     << (torch::cuda::cudnn_is_available() ? "Yes" : "No") << endl;
  cout << "CUDA: " << (torch::cuda::is_available() ?  "Yes" : "No") << endl;
  if (torch::cuda::is_available() ) {
    device_type = torch::kCUDA;
  } else {
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  assert(!in_filename.empty());

  assert(!label_filename.empty());
  ifstream f_labels(label_filename);
  string line;
  vector<string> labels;
  while (getline(f_labels, line)) {
    labels.push_back(line);
  }

  torch::jit::script::Module module;
  try {
    module = torch::jit::load(ts_filename);
    cout << "Loaded TorchScript " << ts_filename << endl;
  } catch (const c10::Error &e) {
    cerr << "Error loading the TorchScript " << ts_filename << endl;
    return EXIT_FAILURE;
  }

  cout << "Start inferencing ..." << endl;

  cv::Mat input_image, resized_image;

  input_image = cv::imread(in_filename, CV_LOAD_IMAGE_COLOR);
  cout << "Original image size [width, height] = [" << input_image.cols
       << ", " << input_image.rows << "]" << endl;
  cv::cvtColor(input_image, resized_image, cv::COLOR_BGR2RGB);
  cv::resize(resized_image, resized_image,
             cv::Size(network_resolution, network_resolution));

  cv::Mat img_float;
  resized_image.convertTo(img_float, CV_32F, 1.0 / 128, -127.0 / 128);

  auto img_tensor =
    torch::from_blob(img_float.data, {1, network_resolution, network_resolution, 3})
    .to(device)
    .permute({0, 3, 1, 2});

  auto start = std::chrono::high_resolution_clock::now();

  vector<torch::jit::IValue> inputs;
  inputs.push_back(img_tensor);
  auto output = module.forward(inputs).toTuple()->elements();

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = duration_cast<milliseconds>(end - start);

  // It should be known that it takes longer time at first time
  cout << "Inference done in " << duration.count() << " ms" << endl;

  torch::Tensor scores = output[0].toTensor().select(0, 0).to(torch::kCPU);
  torch::Tensor boxes = output[1].toTensor().select(0, 0).to(torch::kCPU);

  //cout << "Result dimension is " << scores.dim() << endl;
  print_tensor_shape(scores);
  print_tensor_shape(boxes);

  torch::Tensor picked_box_probs = torch::empty({0});
  vector<int> picked_labels;
  for (int class_index = 1; class_index < scores.size(1); ++class_index) {
    torch::Tensor mask = scores.select(1, class_index) > probability_threshold;
    torch::Tensor prob = scores.select(1, class_index).masked_select(mask);
    torch::Tensor selected_boxes = boxes.index_select(0, mask.nonzero().squeeze());

    if (0 == selected_boxes.size(0)) continue;

    cout << "Class index [" << class_index << "]: "
         << labels.at(class_index) << endl;
    torch::Tensor box_prob = torch::cat({selected_boxes, prob.reshape({-1, 1})}, 1);
    box_prob = hard_nms(box_prob, iou_threshold, top_k, candidate_size);
    picked_box_probs = torch::cat({picked_box_probs, box_prob}, 0);
    picked_labels.insert(picked_labels.end(), box_prob.size(0), class_index);
  }

  print_tensor_shape(picked_box_probs);
  assert(picked_box_probs.size(0) == static_cast<int64_t>(picked_labels.size()));
  if (0 == picked_box_probs.size(0)) {
    cout << "No object detected." << endl;
    return EXIT_SUCCESS;
  }

  auto ra = picked_box_probs.accessor<float, 2>();
  for (int i = 0; i < ra.size(0); ++i) {
    ra[i][0] *= input_image.cols;
    ra[i][1] *= input_image.rows;
    ra[i][2] *= input_image.cols;
    ra[i][3] *= input_image.rows;

    cv::rectangle(input_image, cv::Point(ra[i][0], ra[i][1]),
                  cv::Point(ra[i][2], ra[i][3]), cv::Scalar(255, 255, 0), 4);
    ostringstream oss;
    oss.precision(3);
    oss << labels.at(picked_labels.at(i)) << ": " << ra[i][4];
    cv::putText(input_image, oss.str(), cv::Point(ra[i][0] + 20, ra[i][1] + 40),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 2);
  }

  cv::imwrite(out_filename, input_image);

  return EXIT_SUCCESS;
}

