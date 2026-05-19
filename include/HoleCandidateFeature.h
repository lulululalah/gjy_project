#pragma once

#include <string>

struct HoleCandidateFeature
{
    int candidateId = 0;
    int graphId = 0;
    std::string modelName;

    int hostFaceId = 0;
    int localWireIndex = 0;

    double hostFaceArea = 0.0;
    double hostFaceRelativeArea = 0.0;
    double hostFacePerimeter = 0.0;
    int hostInnerWireCount = 0;
    int hostNeighborFaceCount = 0;

    double wireLength = 0.0;
    double wireLengthRatio = 0.0;
    double estimatedHoleRadius = 0.0;
    double estimatedHoleArea = 0.0;
    double wireCenterX = 0.0;
    double wireCenterY = 0.0;
    double wireCenterZ = 0.0;

    int adjacentFaceCount = 0;
    int adjacentCylinderCount = 0;
    int adjacentSmallCylinderCount = 0;
    double minAdjacentCylinderRadius = 0.0;
    double maxAdjacentCylinderRadius = 0.0;
    double concaveEdgeRatio = 0.0;

    int label = 0;
};
