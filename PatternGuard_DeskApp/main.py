import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_app_class():
    try:
        from gui import PatternGuardApp
    except ModuleNotFoundError as exc:
        missing_package = exc.name or "a required dependency"
        print(
            "PatternGuard could not start because a required dependency is missing: "
            f"{missing_package}.\n"
            "Install the project requirements with: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return PatternGuardApp


def main():
    PatternGuardApp = _load_app_class()
    app = PatternGuardApp()
    app.mainloop()


if __name__ == "__main__":
    main()
