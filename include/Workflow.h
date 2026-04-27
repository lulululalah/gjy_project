#pragma once
#include <string>

// 批量提取训练数据流程
void RunBatchTrainingExport(const std::string& inputDir, const std::string& outputCsv);
    
// 单文件识别准备流程
void RunSingleInferenceExport(const std::string& inputFile, const std::string& outputCsv);
