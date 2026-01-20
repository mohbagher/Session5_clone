"""
Phase 2 Setup Script
====================
Install and configure MATLAB backend integration.
"""

import sys
import subprocess
from pathlib import Path


def setup_phase2():
    """Setup Phase 2 MATLAB integration."""

    print("=" * 70)
    print("PHASE 2 SETUP: MATLAB BACKEND INTEGRATION")
    print("=" * 70)
    print()

    # Step 1: Check Python version
    print("[1/5] Checking Python version...")
    if sys.version_info < (3, 8):
        print("    ❌ Python 3.8+ required")
        return False
    print("    ✓ Python version OK")
    print()

    # Step 2: Create directory structure
    print("[2/5] Creating directory structure...")
    dirs = [
        'physics/matlab_backend',
        'physics/matlab_backend/matlab_scripts',
        'tests'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"    ✓ Created: {dir_path}/")
    print()

    # Step 3: Check for MATLAB
    print("[3/5] Checking MATLAB installation...")
    try:
        result = subprocess.run(
            ['matlab', '-batch', 'version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("    ✓ MATLAB found")
        else:
            print("    ⚠ MATLAB not found in PATH")
    except:
        print("    ⚠ MATLAB not found (install from mathworks.com)")
    print()

    # Step 4: Check MATLAB Engine for Python
    print("[4/5] Checking MATLAB Engine for Python...")
    try:
        import matlab.engine
        print("    ✓ MATLAB Engine for Python installed")
    except ImportError:
        print("    ❌ MATLAB Engine for Python NOT installed")
        print()
        print("    To install:")
        print("    1. Locate your MATLAB installation:")
        print("       - Windows: C:\\Program Files\\MATLAB\\R2023b\\extern\\engines\\python")
        print("       - Linux: /usr/local/MATLAB/R2023b/extern/engines/python")
        print("       - Mac: /Applications/MATLAB_R2023b.app/extern/engines/python")
        print()
        print("    2. Run:")
        print("       cd <path_to_matlab>/extern/engines/python")
        print("       python setup.py install")
        print()
    print()

    # Step 5: Generate MATLAB scripts
    print("[5/5] Generating MATLAB scripts...")
    try:
        from physics.matlab_backend.script_generator import MATLABScriptGenerator

        generator = MATLABScriptGenerator()
        generator.generate_all_scripts()

        print("    ✓ MATLAB scripts generated")
    except Exception as e:
        print(f"    ⚠ Could not generate scripts: {e}")
    print()

    # Summary
    print("=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. If MATLAB Engine is not installed, follow instructions above")
    print("  2. Test the integration:")
    print("     python tests/test_matlab_backend.py")
    print()
    print("  3. Use in dashboard:")
    print("     - Set 'Physics Backend' to 'MATLAB'")
    print("     - Select scenario (e.g., 'rayleigh_basic' or 'cdl_ris')")
    print("     - Run experiment!")
    print()
    print("Documentation:")
    print("  - Phase 2 features: docs/PHASE2_README.md")
    print("  - Troubleshooting: docs/MATLAB_TROUBLESHOOTING.md")
    print()


if __name__ == "__main__":
    setup_phase2()