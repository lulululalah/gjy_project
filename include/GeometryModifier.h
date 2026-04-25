#ifndef GEOMETRY_MODIFIER_H
#define GEOMETRY_MODIFIER_H

#include <TopoDS_Shape.hxx>
#include <vector>
#include <string>
#include "FeatureInfo.h"

class GeometryModifier
{
public:
    GeometryModifier(const TopoDS_Shape &shape);

    // 核心执行函数：根据特征信息列表进行消除
    TopoDS_Shape ExecuteElimination(const std::vector<FaceFeature> &features);

    // 保存处理后的模型
    bool SaveStep(const TopoDS_Shape &shape, const std::string &filename);

private:
    TopoDS_Shape myOriginalShape;
};

#endif