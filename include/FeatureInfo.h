#ifndef FEATURE_INFO_H
#define FEATURE_INFO_H

#pragma once

#include <string>
#include <vector>

#include <GeomAbs_SurfaceType.hxx>

enum EdgeType
{
    CONVEX = 1,
    CONCAVE = 2,
    SMOOTH = 3,
    OTHER = 4
};

struct FaceFeature
{
    int id = 0;
    std::string faceKey;
    double area = 0.0;
    double relativeArea = 0.0;
    double perimeter = 0.0;
    double compactness = 0.0;
    int surfaceType = 0;
    double normalX = 0.0;
    double normalY = 0.0;
    double normalZ = 0.0;
    double centerX = 0.0;
    double centerY = 0.0;
    double centerZ = 0.0;
    double meanCurvature = 0.0;
    double radius = 0.0;
    int numWires = 0;
    int innerWireCount = 0;
    double minInnerWireLength = 0.0;
    double maxInnerWireLength = 0.0;
    std::vector<double> innerWireLengths;
    std::vector<double> innerWireCenterXs;
    std::vector<double> innerWireCenterYs;
    std::vector<double> innerWireCenterZs;
    int numEdges = 0;
    std::vector<int> neighborIds;
    std::vector<int> neighborEdgeTypes;
    int semanticTag = 0;
};

#endif
