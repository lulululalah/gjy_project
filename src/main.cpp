#include "FeatureExtractor.h"
#include "ModelGenerator.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

int main()
{
    // 确保 data 目录存在
    if (!fs::exists("data")) {
        fs::create_directory("data");
    }

    // 1. 生成混合特征测试模型 (既有铆钉又有孔)
    std::cout << "正在生成包含 [铆钉] 和 [盲孔] 的混合测试模型..." << std::endl;
    TopoDS_Shape model = ModelGenerator::GenerateMixedPlate();
    
    std::string modelPath = "data/mixed_test.stp";
    if (ModelGenerator::SaveToStep(model, modelPath)) {
        std::cout << "测试模型已保存至: " << modelPath << std::endl;
    }

    FeatureExtractor extractor(model);
    extractor.Extract();

    auto results = extractor.GetResults();

    // 2. 自动标注：区分铆钉和孔
    // 逻辑：面积小 且 几何中心位于平板表面(z=4)上方 的才是铆钉
    int rivetCount = 0;
    int holeCount = 0;
    for (auto &f : results) {
        if (f.area < 50.0) {
            if (f.centerZ > 4.05) {
                f.semanticTag = 1; // 铆钉
                rivetCount++;
            } else {
                f.semanticTag = 0; // 盲孔 (视为背景或非目标)
                holeCount++;
            }
        } else {
            f.semanticTag = 0; // 大平面
        }
    }

    std::cout << "\n========== 特征统计 ==========" << std::endl;
    std::cout << "识别出铆钉面数 (Target): " << rivetCount << std::endl;
    std::cout << "识别出盲孔面数 (Background): " << holeCount << std::endl;

    std::cout << "\n========== 拓扑连接详情 ==========" << std::endl;
    for (size_t i = 0; i < results.size() && i < 30; ++i) 
    {
        const auto &f = results[i];
        std::cout << "面 ID: " << std::setw(3) << f.id
                  << " | 面积: " << std::setw(8) << (int)f.area
                  << " | Z中心: " << std::fixed << std::setprecision(1) << f.centerZ
                  << " | 标签: " << (f.semanticTag == 1 ? "铆钉" : "背景/孔")
                  << std::endl;
    }

    // 导出 CSV (包含 centerZ 以便 AI 分析)
    std::string csvPath = "data/train_data.csv";
    std::ofstream dataFile(csvPath);
    dataFile << "id,area,relativeArea,perimeter,compactness,surfaceType,nx,ny,nz,centerZ,neighbors,edge_types,label\n";
    for (const auto &f : results)
    {
        dataFile << f.id << "," << f.area << "," << f.relativeArea << "," << f.perimeter << "," << f.compactness << "," << f.surfaceType 
                 << "," << f.normalX << "," << f.normalY << "," << f.normalZ << "," << f.centerZ << ",\"";
        for (size_t j = 0; j < f.neighborIds.size(); ++j) dataFile << f.neighborIds[j] << (j == f.neighborIds.size() - 1 ? "" : " ");
        dataFile << "\",\"";
        for (size_t j = 0; j < f.neighborEdgeTypes.size(); ++j) dataFile << f.neighborEdgeTypes[j] << (j == f.neighborEdgeTypes.size() - 1 ? "" : " ");
        dataFile << "\"," << f.semanticTag << "\n";
    }
    dataFile.close();
    std::cout << "\n数据集已更新: train_data.csv (已包含混合特征)" << std::endl;

    return 0;
}