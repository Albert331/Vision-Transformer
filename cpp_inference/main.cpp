#include <torch/script.h>
#include <iostream>

int main() {
    torch::jit::script::Module module;

    try {
        module = torch::jit::load("model_ft_traced.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        std::cerr << e.what() << std::endl;
        return -1;
    }

    std::cout << "Model loaded successfully.\n";

    
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    module.to(device);
    std::cout << "Running on: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({1, 3, 224, 224}).to(device));

    at::Tensor output = module.forward(inputs).toTensor();

    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "Predicted class: " << output.argmax(1).item<int>() << std::endl;

    return 0;
}