#include "Workflow.h"
#include <iostream>
#include <string>

/**
 * Detector 总调度室
 * 用法:
 *   1. 批量训练模式: Detector.exe --train
 *   2. 单文件识别模式: Detector.exe --predict <path_to_step>
 */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "请指定模式: --train 或 --predict <file>" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "--train") {
        // 调用批量处理流程
        Workflow::RunBatchTrainingExport("data/dirty_training_set", "data/full_training_set.csv");
    } 
    else if (mode == "--predict" && argc >= 3) {
        // 调用单文件识别流程
        std::string filePath = argv[2];
        Workflow::RunSingleInferenceExport(filePath, "data/current_inference.csv");
    } 
    else {
        std::cout << "未知命令或缺少参数。" << std::endl;
        return 1;
    }

    return 0;
}