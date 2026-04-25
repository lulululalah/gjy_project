#pragma once
#include <TopoDS_Shape.hxx>
#include <string>

class ModelGenerator
{
public:
    // 生成一个带铆钉阵列的平板
    static TopoDS_Shape GenerateRivetPlate(int rows = 3, int cols = 3);

    // 生成一个既有铆钉又有孔的混合平板 (用于卷积层挑战)
    static TopoDS_Shape GenerateMixedPlate();

    // 保存为 STEP 文件
    static bool SaveToStep(const TopoDS_Shape& shape, const std::string& filename);
};
