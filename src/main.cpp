#include "FeatureInjector.h"
#include "Workflow.h"

#include <filesystem>
#include <exception>
#include <iostream>
#include <string>

namespace
{
    struct DetectorPaths
    {
        std::filesystem::path wingRivetDatasetDir;
        std::filesystem::path wingRivetTrainingCsv;
        std::filesystem::path faceInferenceCsv;
        std::filesystem::path faceDumpJson;
    };

    DetectorPaths ResolvePaths(const std::filesystem::path &exePath)
    {
        const std::filesystem::path projectRoot = exePath.parent_path().parent_path().parent_path();
        return {
            projectRoot / "data" / "plane_model",
            projectRoot / "data" / "wing_rivet_training_set.csv",
            projectRoot / "data" / "current_inference.csv",
            projectRoot / "data" / "current_faces.json",
        };
    }

    void PrintUsage()
    {
        std::cout
            << "Usage:\n"
            << "  Detector.exe --train\n"
            << "  Detector.exe --predict <file>\n"
            << "  Detector.exe --dump-faces <file>\n"
            << "  Detector.exe --check-face-id <file>\n"
            << "  Detector.exe --inject-wing-rivets <file>\n"
            << "  Detector.exe --inject-star-decals <after-rivet-file> [--host-face <id>]\n"
            << "  Detector.exe --inject-wing-rivets-batch <dir>\n"
            << "  Detector.exe --validate-wing-rivet-dataset <dir>\n"
            << "  Detector.exe --export-wing-rivet-training <dir>\n";
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
        return RunWingRivetTrainingExport(paths.wingRivetDatasetDir.string(), paths.wingRivetTrainingCsv.string());
    }
    else if (mode == "--predict" && argc >= 3)
    {
        const std::string filePath = argv[2];
        RunSingleInferenceExport(filePath, paths.faceInferenceCsv.string());
    }
    else if (mode == "--dump-faces" && argc >= 3)
    {
        const std::string filePath = argv[2];
        RunSingleFaceDump(filePath, paths.faceDumpJson.string());
    }
    else if (mode == "--check-face-id" && argc >= 3)
    {
        const std::string filePath = argv[2];
        return RunFaceIdConsistencyCheck(filePath);
    }
    else if (mode == "--inject-wing-rivets" && argc >= 3)
    {
        const std::string filePath = argv[2];
        return RunWingRivetInjection(filePath);
    }
    else if (mode == "--inject-star-decals" && argc >= 3)
    {
        const std::string filePath = argv[2];
        if (argc == 3) {
            return RunStarDecalInjection(filePath);
        }
        if (argc == 5 && std::string(argv[3]) == "--host-face") {
            try {
                return RunStarDecalInjection(filePath, std::stoi(argv[4]));
            } catch (const std::exception&) {
                std::cout << "Invalid host face ID: " << argv[4] << std::endl;
                return 1;
            }
        }
        std::cout << "Expected optional argument: --host-face <id>" << std::endl;
        return 1;
    }
    else if (mode == "--inject-wing-rivets-batch" && argc >= 3)
    {
        const std::string dirPath = argv[2];
        return RunBatchWingRivetInjection(dirPath);
    }
    else if (mode == "--validate-wing-rivet-dataset" && argc >= 3)
    {
        const std::string dirPath = argv[2];
        return RunWingRivetDatasetValidation(dirPath);
    }
    else if (mode == "--export-wing-rivet-training" && argc >= 3)
    {
        const std::string dirPath = argv[2];
        return RunWingRivetTrainingExport(dirPath, paths.wingRivetTrainingCsv.string());
    }
    else
    {
        std::cout << "Unknown command or missing argument." << std::endl;
        PrintUsage();
        return 1;
    }

    return 0;
}
