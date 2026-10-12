import os
import sys
import time

import paramiko

import ssh_deploy as deploy


def run_ssh_command(client, cmd, password=None, timeout=1200):
    stdin, stdout, _ = client.exec_command(cmd, get_pty=True, timeout=timeout)
    password_sent = False
    output = ""
    while True:
        if stdout.channel.recv_ready():
            chunk = stdout.channel.recv(4096).decode("utf-8", errors="ignore")
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                output += chunk
                if ("password for" in chunk.lower() or "[sudo]" in chunk.lower()) and not password_sent and password:
                    stdin.write(password + "\n")
                    stdin.flush()
                    password_sent = True
        if stdout.channel.exit_status_ready():
            while stdout.channel.recv_ready():
                chunk = stdout.channel.recv(4096).decode("utf-8", errors="ignore")
                if chunk:
                    sys.stdout.write(chunk)
                    output += chunk
            sys.stdout.flush()
            break
        time.sleep(0.1)
    return stdout.channel.recv_exit_status(), output


def main() -> int:
    project_dir = f"/home/{deploy.USERNAME}/ruya-copy"
    local_orchestrator = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orchestrator.py")
    local_perception = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_perception.py")
    local_serial_camera = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serial_camera.py")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        deploy.HOSTNAME,
        username=deploy.USERNAME,
        password=deploy.PASSWORD,
        timeout=30,
        banner_timeout=200,
        look_for_keys=False,
        allow_agent=False,
    )

    sftp = client.open_sftp()
    sftp.put(local_orchestrator, f"{project_dir}/orchestrator.py")
    sftp.put(local_perception, f"{project_dir}/ai_perception.py")
    sftp.put(local_serial_camera, f"{project_dir}/serial_camera.py")
    sftp.close()

    unit = "\n".join(
        [
            "[Unit]",
            "Description=Ruya Smart Glasses",
            "After=network-online.target sound.target",
            "Wants=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"User={deploy.USERNAME}",
            f"WorkingDirectory={project_dir}",
            "Environment=PYTHONUNBUFFERED=1",
            "Environment=RUYA_VIDEO_SOURCE=camera",
            "Environment=RUYA_CAMERA_INDEX=0",
            "Environment=RUYA_SERIAL_PORT=/dev/ttyUSB0",
            "Environment=RUYA_SERIAL_BAUD=921600",
            "Environment=RUYA_VISION_INTERVAL=1.0",
            "Environment=RUYA_TTS_ENGINE=auto",
            f"ExecStart={project_dir}/.venv/bin/python {project_dir}/orchestrator.py vision",
            "Restart=always",
            "RestartSec=2",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "",
        ]
    )

    cmd = (
        "sudo bash -c 'cat > /etc/systemd/system/ruya.service << \"EOF\"\n"
        + unit
        + "EOF'"
    )
    code, _ = run_ssh_command(client, cmd, password=deploy.PASSWORD, timeout=1200)
    if code != 0:
        client.close()
        return code

    code, _ = run_ssh_command(
        client,
        f"sudo usermod -aG dialout {deploy.USERNAME} || true",
        password=deploy.PASSWORD,
        timeout=1200,
    )
    if code != 0:
        client.close()
        return code

    code, _ = run_ssh_command(
        client,
        f"cd '{project_dir}' && source .venv/bin/activate && pip install --upgrade pyserial 2>&1 | tail -20",
        password=None,
        timeout=1200,
    )
    if code != 0:
        client.close()
        return code

    code, _ = run_ssh_command(
        client,
        "sudo systemctl daemon-reload && sudo systemctl restart ruya.service && sudo systemctl --no-pager -l status ruya.service | head -n 60",
        password=deploy.PASSWORD,
        timeout=1200,
    )
    client.close()
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
