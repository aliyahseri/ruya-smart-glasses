import sys
import time

import paramiko

import ssh_deploy as deploy


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python pi_exec.py <remote shell command>")
        return 2

    cmd = sys.argv[1]

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

    _, stdout, stderr = client.exec_command(cmd, get_pty=True)
    while True:
        if stdout.channel.recv_ready():
            chunk = stdout.channel.recv(4096).decode("utf-8", errors="ignore")
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()

        if stderr.channel.recv_ready():
            chunk = stderr.channel.recv(4096).decode("utf-8", errors="ignore")
            if chunk:
                sys.stderr.write(chunk)
                sys.stderr.flush()

        if stdout.channel.exit_status_ready():
            while stdout.channel.recv_ready():
                chunk = stdout.channel.recv(4096).decode("utf-8", errors="ignore")
                if chunk:
                    sys.stdout.write(chunk)
            while stderr.channel.recv_ready():
                chunk = stderr.channel.recv(4096).decode("utf-8", errors="ignore")
                if chunk:
                    sys.stderr.write(chunk)
            sys.stdout.flush()
            sys.stderr.flush()
            break

        time.sleep(0.1)

    exit_code = stdout.channel.recv_exit_status()
    client.close()
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
