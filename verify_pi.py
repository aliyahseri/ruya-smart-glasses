import paramiko

import ssh_deploy as deploy


def main() -> int:
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

    cmd = (
        "cd ~/ruya-copy && "
        "source .venv/bin/activate && "
        "python3 -c 'import whisper; "
        "whisper.load_model(\"tiny\", device=\"cpu\"); "
        "whisper.load_model(\"base\", device=\"cpu\"); "
        "print(\"whisper models OK\")'"
    )
    code, _ = deploy.run_ssh_command(client, cmd, timeout=1800)
    if code != 0:
        client.close()
        return int(code)

    cmd = (
        "cd ~/ruya-copy && "
        "source .venv/bin/activate && "
        "python3 -c 'import orchestrator; print(\"orchestrator import OK\")'"
    )
    code, _ = deploy.run_ssh_command(client, cmd, timeout=600)
    client.close()
    return int(code)


if __name__ == '__main__':
    raise SystemExit(main())
