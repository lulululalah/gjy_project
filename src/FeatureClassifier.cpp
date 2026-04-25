#include "FeatureClassifier.h"
#include <GeomAbs_SurfaceType.hxx>

void FeatureClassifier::Classify(std::vector<FaceFeature> &features, double maxArea)
{
    // 1. 动态阈值：最大面积的 5%
    double threshold = maxArea * 0.05;

    for (auto &f : features)
    {
        f.semanticTag = Identify(f, threshold);
    }
}

int FeatureClassifier::Identify(const FaceFeature &f, double threshold)
{
    // 判定逻辑 A：面积小于阈值
    if (f.area < threshold)
    {
        return 1; // 1 代表微小特征
    }

    // 判定逻辑 B：形状畸形 (紧致度过大，通常是狭长面)
    if (f.compactness > 15.0)
    {
        return 2; // 2 代表狭长面
    }

    return 0; // 正常面
}