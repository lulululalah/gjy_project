#include "FeatureExtractor.h"
#include "FeatureClassifier.h"
#include "GeometryModifier.h"
#include <STEPControl_Reader.hxx>
#include <iostream>
#include <iomanip>
#include <fstream>

int main()
{

    std::string filePath = std::string(TEST_DATA_DIR) + "/test_1.stp";

    STEPControl_Reader reader;

    if (reader.ReadFile(filePath.c_str()) != IFSelect_RetDone)
    {
        // 强制刷新缓冲区，确保能看到错误
        std::cerr << "CRITICAL ERROR: File not found or unreadable!" << std::endl;
        return -1;
    }

    reader.TransferRoots();
    TopoDS_Shape model = reader.OneShape();

    FeatureExtractor extractor(model);
    extractor.Extract();

    auto results = extractor.GetResults();

    // 验证环节：打印邻居关系 (AAG构建验证)
    std::cout << "\n========== 拓扑连接验证 (属性邻接图) ==========" << std::endl;
    for (size_t i = 0; i < results.size() && i < 15; ++i) // 打印前15个面看看
    {
        const auto &f = results[i];
        std::cout << "面 ID: " << std::setw(3) << f.id
                  << " | 面积: " << std::setw(8) << f.area
                  << " | 邻居数: " << f.neighborIds.size()
                  << " | 邻居IDs: [ ";
        for (int nbId : f.neighborIds)
        {
            std::cout << nbId << " ";
        }
        std::cout << "]" << std::endl;
    }
    if (results.size() > 15)
        std::cout << "... 共计 " << results.size() << " 个面 ..." << std::endl;
    std::cout << "==============================================\n"
              << std::endl;

    // 3. 自动分类 (研究内容二：模式匹配/聚类)
    FeatureClassifier classifier;
    // 假设以 15644.8 为阈值进行初步标注 (后续可升级为 GCN 分类)
    classifier.Classify(results, 15644.8);

    // ========== 数据集导出：为 AI 训练准备 (JSON/CSV格式) ==========
    std::ofstream dataFile("train_data.csv");
    dataFile << "id,area,perimeter,compactness,surfaceType,neighbors,edge_types,label\n";
    for (const auto &f : results)
    {
        dataFile << f.id << "," << f.area << "," << f.perimeter << "," << f.compactness << "," << f.surfaceType << ",\"";
        for (size_t j = 0; j < f.neighborIds.size(); ++j)
        {
            dataFile << f.neighborIds[j] << (j == f.neighborIds.size() - 1 ? "" : " ");
        }
        dataFile << "\",\"";
        for (size_t j = 0; j < f.neighborEdgeTypes.size(); ++j)
        {
            dataFile << f.neighborEdgeTypes[j] << (j == f.neighborEdgeTypes.size() - 1 ? "" : " ");
        }
        dataFile << "\"," << f.semanticTag << "\n";
    }
    dataFile.close();
    std::cout << "AI 训练数据集已导出至: train_data.csv" << std::endl;

    // //4. 执行几何消除 (研究内容三：基于连通性的消除)
    // GeometryModifier modifier(model);
    // TopoDS_Shape simplifiedModel = modifier.ExecuteElimination(results);

    // // 5. 保存结果 (研究内容四：拓扑一致性维护)
    // std::string outPath = "simplified_model.stp";
    // if (modifier.SaveStep(simplifiedModel, outPath))
    // {
    //     std::cout << "简化模型已保存至: " << outPath << std::endl;
    // }

    return 0;
}