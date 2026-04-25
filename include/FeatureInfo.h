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
    double perimeter;
    double compactness;
    int surfaceType;
    std::vector<int> neighborIds;
    std::vector<int> neighborEdgeTypes; // 与 neighborIds 一一对应
    int semanticTag;
};

#endif