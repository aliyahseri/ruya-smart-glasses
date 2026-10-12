import shutil

import paramiko

import ssh_deploy as deploy


def main() -> int:
    tmp_dir = None
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
    remote_cache = f"/home/{deploy.USERNAME}/.cache/whisper"
    deploy.remote_mkdir_p(sftp, remote_cache)

    try:
        tmp_dir, files = deploy.prepare_local_whisper_models(deploy.WHISPER_MODELS)
        for local_path, filename in files:
            deploy.sftp_upload_file(sftp, local_path, f"{remote_cache}/{filename}")
    finally:
        sftp.close()
        client.close()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

