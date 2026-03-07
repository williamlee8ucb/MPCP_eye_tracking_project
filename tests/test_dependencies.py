from importlib import import_module
from packaging import version

REQUIRED_VERSIONS = {
    "cv2": "4.8.0",        # opencv-python
    "mediapipe": "0.10.0",
    "numpy": "1.24.0",
    "scipy": "1.10.0"
}



def check_package(import_name, min_version):
    try:
        module = import_module(import_name)
        installed_version = getattr(module, "__version__", None)

        if installed_version is None:
            print(f"[!] {import_name} installed but version could not be detected")
            return False

        if version.parse(installed_version) >= version.parse(min_version):
            print(
                f"[✓] {import_name} version {installed_version} "
                f"(meets requirement >= {min_version})"
            )
            return True
        else:
            print(
                f"[✗] {import_name} version {installed_version} "
                f"is below required >= {min_version}"
            )
            return False

    except ImportError:
        print(f"[✗] {import_name} is not installed")
        return False


def main():
    print("Checking dependency versions...\n")

    results = []
    for pkg, min_ver in REQUIRED_VERSIONS.items():
        results.append(check_package(pkg, min_ver))

    print("\nSummary:")
    if all(results):
        print("✅ All dependencies satisfy the required versions.")
    else:
        print("⚠ Some dependencies are missing or outdated.")
        print("Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
