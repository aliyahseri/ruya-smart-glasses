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

    cmds = [
        "sudo apt-get update -qq",
        "sudo apt-get install -y -qq curl",
        "curl -fsSL https://ollama.com/install.sh | sh",
        "sudo systemctl enable --now ollama",
        "ollama pull llama3.2:1b",
        "ollama list",
    ]

    for cmd in cmds:
        code, _ = deploy.run_ssh_command(
            client,
            cmd,
            password=deploy.PASSWORD,
            timeout=7200,
        )
        if code != 0:
            client.close()
            return int(code)

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

