import os
import asyncssh

HOST_KEY_FILE = "ssh_host_key"

if not os.path.exists(HOST_KEY_FILE):
    print("Generating SSH host key (Python)...")
    # 生成 RSA 私钥对象
    key_obj = asyncssh.generate_private_key('ssh-rsa')
    # 导出为 OpenSSH 私钥格式（返回 bytes）
    private_key_bytes = key_obj.export_private_key('openssh')
    # 转成字符串再写入文件
    private_key_str = private_key_bytes.decode('utf-8')

    with open(HOST_KEY_FILE, "w", encoding="utf-8") as f:
        f.write(private_key_str)
    print("Host key generated:", HOST_KEY_FILE)
