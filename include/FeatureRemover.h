#pragma once

#include <string>

int RunPredictedRivetRemoval(
    const std::string& inputFile,
    const std::string& predictionsFile,
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
