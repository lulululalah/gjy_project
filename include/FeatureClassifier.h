#ifndef FEATURE_CLASSIFIER_H
#define FEATURE_CLASSIFIER_H

#include <vector>
#include "FeatureInfo.h"

class FeatureClassifier
{
public:
    // 输入提取到的特征列表，输出带标签的列表
    void Classify(std::vector<FaceFeature> &features, double maxArea);

private:
    // 具体的判定规则函数
    int Identify(const FaceFeature &feat, double areaThreshold);
};

#endif