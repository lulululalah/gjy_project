"""One command entry for the current aircraft recognition and CAD workflow."""
import argparse
import importlib
import sys

COMMANDS = {
    'train': ('recognition.train', 'main', 'Retrain the current three-class model'),
    'predict': ('recognition.predict', 'main', 'Predict or verify the frozen recognition bundle'),
    'cad': ('cad.pipeline', 'main', 'Run the staged CAD / paired-mesh experiment'),
    'complete-windows': ('cad.windows', 'main', 'Complete supported cosmetic window patches'),
    'mesh': ('geometry.measure', 'mesh_main', 'Measure a STEP surface mesh'),
    'compare-geometry': ('geometry.measure', 'compare_main', 'Compare two STEP geometries'),
    'qem': ('geometry.qem', 'main', 'Measure matched-target QEM components'),
    'labels-init': ('tools.labels', 'initialise_main', 'Initialise background labels for a STEP'),
    'labels-apply': ('tools.labels', 'annotate_main', 'Apply confirmed native window / decal labels'),
    'pick-face': ('tools.faces', 'pick_main', 'Click and inspect a CAD face identifier'),
    'select-host': ('tools.faces', 'host_main', 'Select a Boolean-operation host face'),
    'view-predictions': ('visualization.predictions', 'main', 'Inspect predictions and optional truth'),
    'view-cad': ('visualization.cad', 'main', 'Render native CAD / paper figures'),
    'view-edges': ('visualization.cad', 'wireframe_main', 'View a final STEP and its real edges'),
}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        command = sys.argv[1]
        module, function, _ = COMMANDS[command]
        sys.argv = [f'{sys.argv[0]} {command}', *sys.argv[2:]]
        result = getattr(importlib.import_module(module), function)()
        return 0 if result is None else result
    descriptions = '\n'.join(f'  {name:20} {item[2]}' for name, item in COMMANDS.items())
    parser = argparse.ArgumentParser(description='Current three-class aircraft workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=descriptions)
    parser.add_argument('command', choices=COMMANDS, help='Use <command> --help for its options')
    parser.parse_args()


if __name__ == '__main__':
    raise SystemExit(main())
