#ifndef FEATURE_INFO_H

#pragma once
#include <vector>
#include <string>
#include <GeomAbs_SurfaceType.hxx>
#include <TopTools_ListOfShape.hxx>

enum EdgeType
{
    CONVEX = 1,
    CONCAVE = 2,
    SMOOTH = 3,
    OTHER = 4
};

struct FaceFeature
{
    int id;
    double area;
    double relativeArea; // 面积占比 (0.0 ~ 1.0)
    double perimeter;
    double compactness;
    int surfaceType;
    double normalX, normalY, normalZ; // 面中心法向
    double centerZ; // 几何中心 Z 坐标
    double meanCurvature; // 平均曲率
    int numWires; // 边界回路数量
    int numEdges; // 边的数量
    std::vector<int> neighborIds;
    std::vector<int> neighborEdgeTypes; // 与 neighborIds 一一对应 (Convex:1, Concave:-1, Smooth:0)
    int semanticTag;
};

#endif