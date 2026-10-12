"""
Ru'ya Smart Glasses — Pi Deployment (single script)
=====================================================
Deploys the entire project to a Raspberry Pi over SSH/SFTP.

Usage:
    python ssh_deploy.py

Strategy:
  1. Upload Python source files (with Pi-compatible text_to_speech)
  2. Upload .pt model files + tts_models/ + torch_wheels/
  3. Install system deps, create venv, install all packages offline
  4. Install whisper/pyaudio from PyPI (source dists need compile)
  5. Install Piper TTS ARM64 binary
  6. Patch orchestrator for live camera & verify
"""

import paramiko
import sys
import time
import os
import tempfile
import shutil

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Configuration ─────────────────────────────────────────────────────────────
# Set these in the environment or in a local .env file (see .env.example).
HOSTNAME = os.environ.get("PI_HOST", "raspberrypi.local")
USERNAME = os.environ.get("PI_USER", "pi")
PASSWORD = os.environ.get("PI_PASSWORD")
WHISPER_MODELS = ("tiny", "base")


def prepare_local_whisper_models(models):
    try:
        import whisper
    except Exception as exc:
        raise RuntimeError(f"openai-whisper is required locally to bundle models: {exc}")

    tmp_dir = tempfile.mkdtemp(prefix="ruya_whisper_models_")
    files = []
    for name in models:
        whisper.load_model(name, device="cpu", download_root=tmp_dir)
        expected = os.path.join(tmp_dir, f"{name}.pt")
        if os.path.isfile(expected):
            files.append((expected, f"{name}.pt"))
            continue
        candidates = [p for p in os.listdir(tmp_dir) if p.endswith(".pt")]
        raise RuntimeError(f"Whisper model '{name}' downloaded but '{expected}' not found. Candidates: {candidates}")
    return tmp_dir, files


def run_ssh_command(client, cmd, password=None, timeout=1200):
    """Run an SSH command with live output streaming."""
    print(f"\n{'='*60}\nRunning: {cmd[:200]}{'...' if len(cmd) > 200 else ''}\n{'='*60}")
    stdin, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=timeout)
    password_sent = False
    full_output = ""
    while True:
        if stdout.channel.recv_ready():
            chunk_bytes = stdout.channel.recv(4096)
            try:
                chunk = chunk_bytes.decode('utf-8', errors='ignore')
            except Exception:
                chunk = ''
            sys.stdout.buffer.write(chunk.encode('utf-8', errors='ignore'))
            sys.stdout.flush()
            full_output += chunk
            if ("password for" in chunk.lower() or "[sudo]" in chunk.lower()) and not password_sent and password:
                stdin.write(password + '\n')
                stdin.flush()
                password_sent = True
        if stdout.channel.exit_status_ready():
            while stdout.channel.recv_ready():
                extra = stdout.channel.recv(4096).decode('utf-8', errors='ignore')
                sys.stdout.buffer.write(extra.encode('utf-8', errors='ignore'))
                sys.stdout.flush()
                full_output += extra
            break
        time.sleep(0.1)
    exit_code = stdout.channel.recv_exit_status()
    print(f"\nExit code: {exit_code}\n{'='*60}\n")
    return exit_code, full_output


def remote_mkdir_p(sftp, remote_directory):
    """Recursively create remote directories."""
    dirs = []
    head = remote_directory
    while head and head != "/":
        dirs.append(head)
        head = os.path.dirname(head)
    dirs.reverse()
    for d in dirs:
        try:
            sftp.stat(d)
        except IOError:
            try:
                sftp.mkdir(d)
            except Exception:
                pass


def sftp_upload_file(sftp, local_path, remote_path):
    """Upload a single file with progress for large files."""
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    fname = os.path.basename(local_path)
    if size_mb > 5:
        print(f"  Uploading {fname} ({size_mb:.1f} MB)...")
    try:
        sftp.put(local_path, remote_path)
        return True
    except Exception as e:
        print(f"  WARNING: failed to upload {fname}: {e}")
        return False


def upload_directory(sftp, local_dir, remote_dir, exclude_dirs=None, exclude_exts=None,
                     exclude_files=None, include_exts=None):
    """Upload an entire directory tree via SFTP."""
    exclude_dirs = exclude_dirs or set()
    exclude_exts = exclude_exts or set()
    exclude_files = exclude_files or set()

    file_count = 0
    for root, dirs, files in os.walk(local_dir):
        rel = os.path.relpath(root, local_dir)
        if rel == '.':
            rel = ''
        parts = rel.split(os.sep) if rel else []
        if any(p in exclude_dirs for p in parts):
            continue
        remote_path = os.path.join(remote_dir, rel).replace('\\', '/')
        remote_mkdir_p(sftp, remote_path)
        for fname in files:
            if fname.startswith('.'):
                continue
            if fname in exclude_files:
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in exclude_exts:
                continue
            if include_exts and ext not in include_exts:
                continue
            local_path = os.path.join(root, fname)
            remote_file = remote_path + '/' + fname
            if sftp_upload_file(sftp, local_path, remote_file):
                file_count += 1
    return file_count


def main():
    print("\n" + "=" * 60)
    print("  Ru'ya Smart Glasses — Pi Deployment")
    print("=" * 60)

    if not PASSWORD:
        sys.exit("Set PI_PASSWORD (see .env.example)")

    local_project_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Connect ───────────────────────────────────────────────────────────────
    print(f"\nConnecting to {USERNAME}@{HOSTNAME}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOSTNAME, username=USERNAME, password=PASSWORD,
                   timeout=30, banner_timeout=200,
                   look_for_keys=False, allow_agent=False)
    print("Connected!\n")

    exit_code, home_out = run_ssh_command(client, "echo $HOME")
    remote_home = home_out.strip().splitlines()[-1] if home_out else "/home/" + USERNAME
    project_dir = f"{remote_home}/ruya-copy"

    run_ssh_command(client, f"mkdir -p '{project_dir}'")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE A — SFTP UPLOAD
    # ══════════════════════════════════════════════════════════════════════════
    sftp = client.open_sftp()

    # A1. Python source files
    print("\n--- A1: Uploading Python source files ---")
    exclude_dirs = {'.venv', '.git', '__pycache__', 'OCRmodels', 'tts_models',
                    'torch_wheels', 'all_wheels', '.vscode'}
    exclude_exts = {'.pt', '.onnx', '.mp4', '.avi', '.exe'}
    # Files that should NOT be deployed to the Pi
    exclude_files = {
        'text_to_speech.py',        # Windows version — replaced by Pi version
        'text_to_speech_pi.py',     # Uploaded separately as text_to_speech.py
        'piper.exe',                # Windows binary
        'ssh_deploy.py',            # This script — not needed on Pi
        'ssh_deploy_fixed.py', 'ssh_deploy_resume.py',
        'ssh_deploy_final.py', 'ssh_fix_hfxet.py',
        'ssh_check_os.py', 'ssh_check_python.py', 'ssh_install_python11.py',
        'cv2.py',                   # Dummy file
        'testvideo.mp4',
    }

    count = upload_directory(sftp, local_project_dir, project_dir,
                             exclude_dirs=exclude_dirs,
                             exclude_exts=exclude_exts,
                             exclude_files=exclude_files)
    print(f"  Uploaded {count} source files.")

    # Upload Pi-compatible text_to_speech as text_to_speech.py
    pi_tts = os.path.join(local_project_dir, "text_to_speech_pi.py")
    if os.path.exists(pi_tts):
        print("  Uploading Pi-compatible text_to_speech.py...")
        sftp_upload_file(sftp, pi_tts, f"{project_dir}/text_to_speech.py")
    else:
        print("  WARNING: text_to_speech_pi.py not found, uploading original")
        orig_tts = os.path.join(local_project_dir, "text_to_speech.py")
        if os.path.exists(orig_tts):
            sftp_upload_file(sftp, orig_tts, f"{project_dir}/text_to_speech.py")

    # A2. YOLO model files (.pt)
    print("\n--- A2: Uploading YOLO model files ---")
    for model_file in ("yolo11n.pt", "final.pt"):
        local_model = os.path.join(local_project_dir, model_file)
        if os.path.exists(local_model):
            sftp_upload_file(sftp, local_model, f"{project_dir}/{model_file}")
        else:
            print(f"  WARNING: {model_file} not found!")

    # A3. TTS voice models
    print("\n--- A3: Uploading TTS voice models ---")
    tts_local = os.path.join(local_project_dir, "tts_models")
    if os.path.exists(tts_local):
        count = upload_directory(sftp, tts_local, f"{project_dir}/tts_models")
        print(f"  Uploaded {count} TTS files.")

    # A4. Pre-built aarch64 wheels
    print("\n--- A4: Uploading pre-built wheels (this takes a few minutes) ---")
    wheels_local = os.path.join(local_project_dir, "torch_wheels")
    if os.path.exists(wheels_local):
        count = upload_directory(sftp, wheels_local, f"{project_dir}/torch_wheels",
                                 include_exts={'.whl', '.tar.gz', '.zip'})
        print(f"  Uploaded {count} wheel/sdist files.")
    else:
        print("  FATAL: torch_wheels/ directory not found!")
        client.close()
        sys.exit(1)

    print("\n--- A5: Bundling Whisper model weights for offline use ---")
    print("  Attempting to bundle Whisper weights from this machine...")
    whisper_cache_remote = f"{remote_home}/.cache/whisper"
    remote_mkdir_p(sftp, whisper_cache_remote)
    whisper_tmp_dir = None
    bundled_locally = False
    try:
        whisper_tmp_dir, whisper_files = prepare_local_whisper_models(WHISPER_MODELS)
        for local_path, filename in whisper_files:
            sftp_upload_file(sftp, local_path, f"{whisper_cache_remote}/{filename}")
        print(f"  Uploaded Whisper models: {', '.join(name for name in WHISPER_MODELS)}")
        bundled_locally = True
    except Exception as exc:
        print(f"  WARNING: Local Whisper bundling failed: {exc}")
        print("  Will download/cache Whisper models on the Pi during setup instead.")
    finally:
        if whisper_tmp_dir:
            shutil.rmtree(whisper_tmp_dir, ignore_errors=True)

    sftp.close()
    print("\n--- Upload complete ---\n")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE B — INSTALLATION
    # ══════════════════════════════════════════════════════════════════════════

    steps = [
        # B1: System packages
        ("Install system packages",
         "sudo apt-get update -qq && "
         "sudo apt-get install -y -qq "
         "build-essential zlib1g-dev libncurses-dev libgdbm-dev libnss3-dev "
         "libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev "
         "liblzma-dev portaudio19-dev espeak-ng libespeak-dev "
         "ffmpeg libasound2-dev alsa-utils",
         True, True),  # (needs_sudo, critical)

        # B2: Python 3.11
        ("Check / Install Python 3.11",
         "if python3.11 --version 2>/dev/null; then "
         "  echo 'Python 3.11 already installed'; "
         "else "
         "  cd /tmp && "
         "  wget -q https://www.python.org/ftp/python/3.11.12/Python-3.11.12.tgz && "
         "  tar -xf Python-3.11.12.tgz && "
         "  cd /tmp/Python-3.11.12 && "
         "  ./configure --enable-optimizations --prefix=/usr/local 2>&1 | tail -5 && "
         "  make -j4 2>&1 | tail -5 && "
         "  sudo make altinstall 2>&1 | tail -5; "
         "fi",
         True, True),

        # B3: Venv
        ("Ensure virtual environment exists",
         f"cd '{project_dir}' && "
         f"if [ -f .venv/bin/activate ]; then "
         f"  echo 'Venv already exists'; "
         f"else "
         f"  python3.11 -m venv .venv; "
         f"fi",
         False, True),

        # B4: Upgrade pip
        ("Upgrade pip",
         f"cd '{project_dir}' && source .venv/bin/activate && "
         f"pip install --upgrade pip setuptools wheel 2>&1 | tail -3",
         False, True),

        # B5: Install packages offline from local wheels
        ("Install packages from local wheels",
         f"cd '{project_dir}' && source .venv/bin/activate && "
         f"pip install --no-index --find-links=./torch_wheels "
         f"torch torchvision "
         f"ultralytics timm "
         f"easyocr piper-tts "
         f"huggingface-hub safetensors "
         f"httpx httpcore h11 anyio sniffio "
         f"pydantic==2.13.4 pydantic-core==2.46.4 annotated-types typing-inspection "
         f"psutil pyttsx3 sounddevice SpeechRecognition "
         f"ollama requests pyyaml "
         f"opencv-python",
         False, True),

        # B6: openai-whisper (offline from local wheel/sdist)
        ("Install openai-whisper (offline)",
         f"cd '{project_dir}' && source .venv/bin/activate && "
         f"pip install --no-index --find-links=./torch_wheels openai-whisper 2>&1 | tail -20",
         False, True),

        ("Install pyserial (online)",
         f"cd '{project_dir}' && source .venv/bin/activate && "
         f"pip install --upgrade pyserial 2>&1 | tail -20",
         False, False),

        ("Cache Whisper models on Pi (for offline runtime)",
         f"cd '{project_dir}' && source .venv/bin/activate && "
         f"python3 -c \"import whisper; "
         f"[whisper.load_model(m, device='cpu') for m in {list(WHISPER_MODELS)}]; "
         f"print('Whisper cached')\"",
         False, False),

        # B8: Piper TTS ARM64 binary
        ("Install Piper TTS binary",
         f"cd '{project_dir}' && "
         f"if [ -x ./piper ]; then "
         f"  echo 'Piper binary already exists'; "
         f"else "
         f"  wget -q --timeout=60 https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz && "
         f"  tar -xf piper_arm64.tar.gz && cp piper/piper . && chmod +x piper && "
         f"  rm -rf piper_arm64.tar.gz piper/; "
         f"fi",
         False, False),

        # B9: Verify
        ("Verify installation",
         f"cd '{project_dir}' && source .venv/bin/activate && "
         f"python3 -c \""
         f"import whisper; whisper.load_model('tiny', device='cpu'); print('whisper tiny OK'); "
         f"import torch; print(f'torch {{torch.__version__}}'); "
         f"import ultralytics; print(f'ultralytics {{ultralytics.__version__}}'); "
         f"import cv2; print(f'opencv {{cv2.__version__}}'); "
         f"import numpy; print(f'numpy {{numpy.__version__}}'); "
         f"import easyocr; print('easyocr OK'); "
         f"import ollama; print('ollama OK'); "
         f"import sounddevice; print('sounddevice OK'); "
         f"import psutil; print('psutil OK'); "
         f"import timm; print(f'timm {{timm.__version__}}'); "
         f"print('ALL IMPORTS OK'); "
         f"\"",
         False, False),

        ("Ensure user is in audio/video groups",
         f"sudo usermod -aG audio,video {USERNAME} || true",
         True, False),

        ("Ensure user is in dialout group (serial devices)",
         f"sudo usermod -aG dialout {USERNAME} || true",
         True, False),

        ("Pre-cache MiDaS model (for offline runtime)",
         f"cd '{project_dir}' && source .venv/bin/activate && "
         f"python3 -c \"import torch; torch.hub.load('intel-isl/MiDaS','MiDaS_small'); print('MiDaS cached')\"",
         False, False),

        ("Install systemd service",
         f"sudo bash -c 'cat > /etc/systemd/system/ruya.service << \"EOF\"\n"
         f"[Unit]\n"
         f"Description=Ruya Smart Glasses\n"
         f"After=network-online.target sound.target\n"
         f"Wants=network-online.target\n"
         f"\n"
         f"[Service]\n"
         f"Type=simple\n"
         f"User={USERNAME}\n"
         f"WorkingDirectory={project_dir}\n"
         f"Environment=PYTHONUNBUFFERED=1\n"
         f"Environment=RUYA_VIDEO_SOURCE=camera\n"
         f"Environment=RUYA_CAMERA_INDEX=0\n"
        f"Environment=RUYA_SERIAL_PORT=/dev/ttyUSB0\n"
        f"Environment=RUYA_SERIAL_BAUD=921600\n"
        f"Environment=RUYA_VISION_INTERVAL=1.0\n"
         f"Environment=RUYA_TTS_ENGINE=auto\n"
        f"ExecStart={project_dir}/.venv/bin/python {project_dir}/orchestrator.py vision\n"
         f"Restart=always\n"
         f"RestartSec=2\n"
         f"\n"
         f"[Install]\n"
         f"WantedBy=multi-user.target\n"
         f"EOF'",
         True, True),

        ("Enable and start ruya service",
         "sudo systemctl daemon-reload && sudo systemctl enable --now ruya.service && sudo systemctl --no-pager -l status ruya.service | head -n 50",
         True, True),
    ]

    failed_noncritical = []

    for i, (name, cmd, needs_sudo, critical) in enumerate(steps, 1):
        print(f"\n\n{'#'*60}")
        print(f"# Step {i}/{len(steps)}: {name}")
        print(f"{'#'*60}")

        exit_code, output = run_ssh_command(
            client, cmd,
            password=PASSWORD if needs_sudo else None,
            timeout=1800
        )

        if exit_code != 0:
            if critical:
                print(f"\n  FATAL: '{name}' failed (exit {exit_code}). Aborting.")
                client.close()
                sys.exit(1)
            else:
                print(f"  WARNING: '{name}' failed (exit {exit_code}), continuing...")
                failed_noncritical.append(name)

    # ══════════════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  DEPLOYMENT COMPLETE!")
    print("=" * 60)

    if failed_noncritical:
        print("\n  Non-critical failures (can fix manually later):")
        for name in failed_noncritical:
            print(f"    - {name}")

    print(f"\n  To run on the Pi:")
    print(f"    ssh {USERNAME}@{HOSTNAME}")
    print(f"    cd {project_dir}")
    print(f"    source .venv/bin/activate")
    print(f"    python orchestrator.py interactive")
    print()
    print("  Service control:")
    print("    sudo systemctl status ruya.service")
    print("    sudo systemctl restart ruya.service")
    print("    journalctl -u ruya.service -n 200 --no-pager")
    print()

    client.close()


if __name__ == "__main__":
    main()
