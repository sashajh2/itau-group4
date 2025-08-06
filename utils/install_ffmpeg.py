import shutil
import subprocess
import sys
import platform

def install_ffmpeg():
    # Check if ffmpeg is already available
    if shutil.which("ffmpeg"):
        print("✅ FFmpeg is already installed. Skipping installation.")
        return

    system = platform.system()
    print(f"🔧 FFmpeg not found. Attempting installation on {system}...")

    try:
        if system == "Darwin":  # macOS
            subprocess.run(["brew", "--version"], check=True, capture_output=True)
            subprocess.run(["brew", "install", "ffmpeg"], check=True)
        elif system == "Linux":
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)
        else:
            raise OSError(f"Unsupported operating system: {system}")
    except subprocess.CalledProcessError as e:
        print("❌ Installation failed with the following output:")
        print("STDOUT:", e.stdout.decode() if e.stdout else "(no stdout)")
        print("STDERR:", e.stderr.decode() if e.stderr else "(no stderr)")
        raise RuntimeError("FFmpeg installation failed.") from e

    # Final check
    if shutil.which("ffmpeg"):
        print("✅ FFmpeg successfully installed.")
    else:
        raise RuntimeError("FFmpeg installation completed but ffmpeg not found in PATH.")

if __name__ == "__main__":
    install_ffmpeg()


