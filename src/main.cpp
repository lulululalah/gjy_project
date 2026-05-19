#include "Workflow.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace
{
    struct DetectorPaths
    {
        std::filesystem::path trainingInputDir;
        std::filesystem::path faceTrainingCsv;
        std::filesystem::path faceInferenceCsv;
        std::filesystem::path holeTrainingCsv;
        std::filesystem::path holeInferenceCsv;
    };

    DetectorPaths ResolvePaths(const std::filesystem::path &exePath)
    {
        const std::filesystem::path projectRoot = exePath.parent_path().parent_path().parent_path();
        return {
            projectRoot / "data" / "dirty_training_set",
            projectRoot / "data" / "full_training_set.csv",
            projectRoot / "data" / "current_inference.csv",
            projectRoot / "data" / "hole_candidate_training_set.csv",
            projectRoot / "data" / "current_hole_candidates.csv",
        };
    }

    void PrintUsage()
    {
        std::cout
            << "Usage:\n"
            << "  Detector.exe --train\n"
            << "  Detector.exe --predict <file>\n"
            << "  Detector.exe --train-hole\n"
            << "  Detector.exe --predict-hole <file>\n";
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        PrintUsage();
        return 1;
    }

    const std::string mode = argv[1];
    const auto paths = ResolvePaths(std::filesystem::absolute(argv[0]));

    if (mode == "--train")
    {
        RunBatchTrainingExport(paths.trainingInputDir.string(), paths.faceTrainingCsv.string());
    }
    else if (mode == "--train-hole")
    {
        RunHoleCandidateTrainingExport(paths.trainingInputDir.string(), paths.holeTrainingCsv.string());
    }
    else if (mode == "--predict" && argc >= 3)
    {
        const std::string filePath = argv[2];
        RunSingleInferenceExport(filePath, paths.faceInferenceCsv.string());
    }
    else if (mode == "--predict-hole" && argc >= 3)
    {
        const std::string filePath = argv[2];
        RunSingleHoleCandidateInferenceExport(filePath, paths.holeInferenceCsv.string());
    }
    else
    {
        std::cout << "Unknown command or missing argument." << std::endl;
        PrintUsage();
        return 1;
    }

    return 0;
}
