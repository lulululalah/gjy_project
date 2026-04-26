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

    // 1. 生成铆钉阵列测试模型
    std::cout << "正在生成包含铆钉阵列的测试模型..." << std::endl;
    TopoDS_Shape model = ModelGenerator::GenerateRivetPlate(3, 3);
    
    std::string modelPath = "data/rivet_plate.stp";
    if (ModelGenerator::SaveToStep(model, modelPath)) {
        std::cout << "测试模型已保存至: " << modelPath << std::endl;
    }

    FeatureExtractor extractor(model);
    extractor.Extract();

    auto results = extractor.GetResults();

    // 2. 自动标注：区分基体、铆钉和碎面(垃圾)
    int rivetCount = 0;
    int garbageCount = 0;
    for (auto &f : results) {
        if (f.area > 50.0) {
            f.semanticTag = 0; // 大平面 (背景)
        } else {
            if (f.centerZ > 2.05) {
                // 如果紧致度比较小 (接近1或在合理范围内)，判定为铆钉
                if (f.compactness < 5.0) {
                    f.semanticTag = 1; // 铆钉 (零件)
                    rivetCount++;
                } else {
                    f.semanticTag = 2; // 碎面/毛刺 (垃圾)
                    garbageCount++;
                }
            } else {
                f.semanticTag = 0; // 盲孔 (视为背景)
            }
        }
    }

    std::cout << "\n========== 特征统计 ==========" << std::endl;
    std::cout << "识别出零件面数 (Keep): " << rivetCount << std::endl;
    std::cout << "识别出垃圾面数 (Delete): " << garbageCount << std::endl;

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

    // 导出 CSV (包含新特征以便 AI 分析)
    std::string csvPath = "data/train_data.csv";
    std::ofstream dataFile(csvPath);
    dataFile << "id,area,relativeArea,perimeter,compactness,surfaceType,nx,ny,nz,centerZ,meanCurvature,numWires,numEdges,neighbors,edge_types,label\n";
    for (const auto &f : results)
    {
        dataFile << f.id << "," << f.area << "," << f.relativeArea << "," << f.perimeter << "," << f.compactness << "," << f.surfaceType 
                 << "," << f.normalX << "," << f.normalY << "," << f.normalZ << "," << f.centerZ 
                 << "," << f.meanCurvature << "," << f.numWires << "," << f.numEdges << ",\"";
        for (size_t j = 0; j < f.neighborIds.size(); ++j) dataFile << f.neighborIds[j] << (j == f.neighborIds.size() - 1 ? "" : " ");
        dataFile << "\",\"";
        for (size_t j = 0; j < f.neighborEdgeTypes.size(); ++j) dataFile << f.neighborEdgeTypes[j] << (j == f.neighborEdgeTypes.size() - 1 ? "" : " ");
        dataFile << "\"," << f.semanticTag << "\n";
    }
    dataFile.close();
    std::cout << "\n数据集已更新: train_data.csv (已包含混合特征)" << std::endl;

    return 0;
}