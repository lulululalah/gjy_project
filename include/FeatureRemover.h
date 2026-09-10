#pragma once

#include <string>

int RunPredictedRivetRemoval(
    const std::string& inputFile,
    const std::string& predictionsFile,
    const std::string& outputFile);

int RunPredictedSurfaceFeatureRemoval(
    const std::string& inputFile,
    const std::string& predictionsFile,
    const std::string& outputFile);

int RunInvalidSurfaceHostRebuild(
    const std::string& inputFile,
    const std::string& predictionsFile,
    const std::string& outputFile);

int RunSplitWindowSkinRebuild(
    const std::string& inputFile,
    const std::string& predictionsFile,
    const std::string& outputFile);

int RunBridgeSplitWindowFace(
    const std::string& inputFile,
    int windowFaceId,
    const std::string& outputFile);

int RunEmbeddedWindowHostRebuild(
    const std::string& inputFile,
    const std::string& predictionsFile,
    const std::string& outputFile,
    const std::string& rebuildProfile = "auto");

int RunBatchEmbeddedWindowHostRebuild(
    const std::string& inputDir,
    const std::string& predictionsDir,
    const std::string& outputDir);
