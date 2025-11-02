#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simlinux_stream.py
一个基于 asyncssh 的“模拟 Linux 主机”SSH 服务：
- 不会执行真实命令（所有命令由模板或 LLM 模拟输出）
- 支持多发行版配置（distros/*.json）
- 当调用 LLM 时，将 LLM 输出**按行**流式写回 SSH 客户端（在一个代码块内）
- 简单的危险命令过滤与会话内切换发行版命令

依赖:
    pip install asyncssh openai aiofiles

运行:
    export OPENAI_API_KEY="sk-..."
    python simlinux_stream.py
然后在局域网内用 ssh 连接 (默认端口 2222)：ssh user@<host> -p 2222  密码: password
"""

import asyncio
import asyncssh
import os
import json
import logging
from datetime import datetime
import openai
import aiofiles
import sys

# -------------------- 配置区 --------------------
OPENAI_API_KEY = os.environ.get("sk-738a8fb30d6b40b084e2ee8a7afb0a46")  # 请在环境变量里设置
BASE_URL = "https://api.deepseek.com/"
HOST = "0.0.0.0"
PORT = 2222
BANNER = "Welcome to SimLinux (SIMULATED). This is a virtual shell; no real commands are executed.\n"
DISTRO_DIR = "distros"
HOST_KEY = "ssh_host_key"
USERNAME = "root"
PASSWORD = "123"  # 测试用；真实部署请用密钥
# 可自定义：按行显示间隔（秒）
LINE_DELAY = 0.01
# LLM 模型与流式开关
OPENAI_MODEL = "deepseek-chat"
USE_OPENAI_STREAM = True
# -------------------------------------------------

openai.api_key = OPENAI_API_KEY
openai.base_url = BASE_URL
logging.basicConfig(level=logging.INFO)

# 危险关键字，出现则拒绝（你可以自行扩展）
DANGEROUS = []

# 基础虚拟文件系统（可扩展）
BASE_VFS = {
    "/README.txt": "This machine is simulated for learning purposes.\n",
}

# -------------------- 辅助：加载/初始化发行版 --------------------
def load_distros():
    distros = {}
    os.makedirs(DISTRO_DIR, exist_ok=True)
    for fn in os.listdir(DISTRO_DIR):
        if fn.endswith(".json"):
            path = os.path.join(DISTRO_DIR, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "id" not in data:
                        data["id"] = fn[:-5]
                    distros[data["id"]] = data
            except Exception as e:
                logging.warning("Failed to load distro %s: %s", fn, e)
    return distros

DISTROS = load_distros()

# 若没有任何 distro，则生成两个示例并要求重启
if not DISTROS:
    example_ubuntu = {
        "id": "ubuntu-22.04",
        "name": "Ubuntu 22.04 LTS",
        "os_release": {
            "NAME":"Ubuntu",
            "VERSION":"22.04 LTS (Jammy Jellyfish)",
            "ID":"ubuntu",
            "VERSION_ID":"22.04",
            "PRETTY_NAME":"Ubuntu 22.04 LTS"
        },
        "uname": "Linux {hostname} 5.15.0-50-generic x86_64 GNU/Linux",
        "uname_vars": {"hostname":"simlinux","kernel":"Linux","release":"5.15.0-50-generic","arch":"x86_64"},
        "pkg_manager": "apt",
        "installed_packages": ["bash","coreutils","python3","openssh-server"],
        "issue": "Ubuntu 22.04 LTS \\n"
    }
    example_arch = {
        "id": "arch",
        "name": "Arch Linux",
        "os_release": {"NAME":"Arch Linux","ID":"arch","PRETTY_NAME":"Arch Linux"},
        "uname": "Linux {hostname} 6.1.0-arch x86_64 GNU/Linux",
        "uname_vars": {"hostname":"simlinux","kernel":"Linux","release":"6.1.0-arch","arch":"x86_64"},
        "pkg_manager": "pacman",
        "installed_packages": ["bash","coreutils","python"],
        "issue": "Arch Linux \\n"
    }
    os.makedirs(DISTRO_DIR, exist_ok=True)
    with open(os.path.join(DISTRO_DIR,"ubuntu-22.04.json"), "w", encoding="utf-8") as f:
        json.dump(example_ubuntu, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DISTRO_DIR,"arch.json"), "w", encoding="utf-8") as f:
        json.dump(example_arch, f, ensure_ascii=False, indent=2)
    print("Created example distros in ./distros. Edit them if needed and restart the program.")
    sys.exit(0)

#DEFAULT_DISTRO = next(iter(DISTROS.keys()))
# -------------------- 默认发行版选择 --------------------
DISTRO_LIST = list(DISTROS.keys())

print("Available distros:")
for idx, key in enumerate(DISTRO_LIST, 1):
    print(f"{idx}. {DISTROS[key].get('name')} ({key})")

# 等待用户输入选择
while True:
    try:
        choice = int(input(f"Select a distro to use (1-{len(DISTRO_LIST)}): "))
        if 1 <= choice <= len(DISTRO_LIST):
            DEFAULT_DISTRO = DISTRO_LIST[choice - 1]
            print(f"Selected distro: {DISTROS[DEFAULT_DISTRO].get('name')}")
            break
    except ValueError:
        pass
    print("Invalid input, please enter a number corresponding to the distro.")


# -------------------- 渲染函数 --------------------
def render_os_release(distro):
    fields = distro.get("os_release", {})
    lines = [f'{k}="{v}"' for k,v in fields.items()]
    return "\n".join(lines) + "\n"

def render_uname(distro, hostname):
    template = distro.get("uname", "Linux {hostname} {release} {arch}")
    vars = dict(distro.get("uname_vars", {}))
    vars.setdefault("hostname", hostname)
    return template.format(**vars)

def render_pkg_list(distro):
    mgr = distro.get("pkg_manager", "apt")
    pkgs = distro.get("installed_packages", [])
    if mgr == "apt":
        return "\n".join([f"{name}/{distro.get('id','sim')} 1.0" for name in pkgs]) + ("\n" if pkgs else "Listing...\n")
    if mgr in ("yum","dnf"):
        return "\n".join([f"{p} x86_64 1.0" for p in pkgs]) + ("\n" if pkgs else "Listing...\n")
    if mgr == "pacman":
        return "\n".join(pkgs) + ("\n" if pkgs else "Listing...\n")
    return "\n".join(pkgs) + ("\n")

# -------------------- LLM 流式请求（OpenAI示例） --------------------
async def ask_llm_stream(command: str, distro: dict, context: str = ""):
    """
    使用 OpenAI ChatCompletion 流式返回（生成器），每次 yield 一段文本（可能不是整行）。
    如果 OPENAI 的流式接口不可用，fallback 会返回整段文本一次性 yield。
    """
    system_prompt = (
        "You are simulating a restricted Linux terminal. DO NOT execute any real commands. "
        "Base your output only on the provided simulated distro information (os_release, installed_packages, pkg_manager) "
        "and the virtual filesystem. Reply with terminal-style output only; do not include explanations. "
        "When streaming, produce the terminal output text without adding extra markdown fences (the server will add fences)."
    )

    user_prompt = (
        f"DISTRO_ID: {distro.get('id')}\n"
        f"OS_RELEASE: {json.dumps(distro.get('os_release',{}), ensure_ascii=False)}\n"
        f"INSTALLED_PACKAGES: {distro.get('installed_packages', [])}\n"
        f"PKG_MANAGER: {distro.get('pkg_manager')}\n"
        f"CONTEXT: {context}\n"
        f"COMMAND: {command}\n"
        "Generate plausible terminal output. Respect distro's installed packages and versions."
    )

    # 如果没有 API key，抛出清晰错误
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set; set environment variable or modify the code to use a local model.")

    # 尝试使用 OpenAI 的流式接口
    if USE_OPENAI_STREAM:
        # 这里使用 ChatCompletion.create(stream=True) 的经典流式写法
        try:
            # 注意：不同 openai sdk 版本流式接口略有不同。此处使用常见模式。
            stream_resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role":"system","content":system_prompt},
                    {"role":"user","content":user_prompt}
                ],
                temperature=0.2,
                stream=True,
                max_tokens=800
            )
            # stream_resp 是一个可迭代对象，逐个 chunk 返回
            partial = ""
            for chunk in stream_resp:
                # chunk 结构取决于 SDK 版本 - 试图兼容常见 delta 结构
                try:
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content") or chunk.get("text") or ""
                except Exception:
                    content = chunk.get("text", "") or ""
                if not content:
                    continue
                # 直接 yield 内容片段
                yield content
            return
        except Exception as e:
            logging.warning("OpenAI streaming failed, falling back to non-stream. Reason: %s", e)
            # fallthrough to non-stream
    # 非流或回退：一次性请求完整答案并切分为片段 yield
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        temperature=0.2,
        max_tokens=800
    )
    content = ""
    try:
        # 不同版本返回结构差异
        content = resp.choices[0].message.content
    except Exception:
        content = resp.choices[0].text if hasattr(resp.choices[0], "text") else str(resp)
    # yield small chunks but ensure we preserve lines — 接下来由 caller 按行组装
    for i in range(0, len(content), 256):
        yield content[i:i+256]

# -------------------- SSH 会话处理 --------------------
'''
class SimSSHSession(asyncssh.SSHServerSession):
    def __init__(self, *args, **kwargs):
        # 捕获所有参数，避免 asyncssh 传参报错
        self._chan = None
        self._input_buffer = ""
        self.distro = DISTROS.get(DEFAULT_DISTRO, {"id":"sim","os_release":{},"installed_packages":[]})
        self.vfs = dict(BASE_VFS)
        self.vfs["/etc/os-release"] = render_os_release(self.distro)
        self.vfs["/etc/issue"] = self.distro.get("issue", "")
        self.hostname = "simlinux"

    def connection_made(self, chan):
        self._chan = chan
        self._chan.write(BANNER)
        self._chan.write(self.prompt())

    def shell_requested(self):
        return True

    def pty_requested(self, term_type, term_size, term_modes):
        return True  # 接受 PTY

    def prompt(self):
        return f"user@{self.hostname}:~$ "

    def data_received(self, data, datatype):
        print("Received:", repr(data))  # 检查是否收到输入
        response = f"Simulated shell received: {data}"
        self._chan.write(response)
        self._chan.write(self.prompt())

    def eof_received(self):
        self._chan.write("\nBye!\n")
        return True

    async def write_line(self, line: str):
        """按行写回客户端并延迟，增强真实感。"""
        self._chan.write(line + "\n")
        await asyncio.sleep(LINE_DELAY)

    async def stream_and_write(self, stream_generator):
        """
        接收 LLM 的片段流，把片段拼成行，按行写回。
        在开始时写入代码块起始标记，在结束时写入结束标记。
        """
        # 先输出三反引号（在 SSH 端可看到），确保“唯一代码块”的视觉效果
        self._chan.write("```" + "\n")
        buffer = ""
        try:
            async for chunk in stream_generator:
                # chunk 可能是很短的片段；累积直到有换行再写行
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    await self.write_line(line)
            # 结束时把残留 buffer 作为最后一行写出（如果非空）
            if buffer:
                # 可能没换行，作为最后一行写出
                await self.write_line(buffer)
        except TypeError:
            # 如果传入的 stream_generator 不是异步生成器（兼容 fallback），尝试同步迭代
            try:
                for chunk in stream_generator:
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        await self.write_line(line)
                if buffer:
                    await self.write_line(buffer)
            except Exception as e:
                await self.write_line(f"Error streaming LLM: {e}")
        except Exception as e:
            await self.write_line(f"Error streaming LLM: {e}")
        # 最后关闭代码块
        self._chan.write("```" + "\n")

    async def handle_command(self, cmd: str):
        logging.info("[%s] CMD: %s", datetime.utcnow().isoformat(), cmd)
        # 会话命令处理：list-distros, set-distro <id>
        if cmd.startswith("set-distro "):
            chosen = cmd.split(" ",1)[1].strip()
            if chosen in DISTROS:
                self.distro = DISTROS[chosen]
                self.vfs["/etc/os-release"] = render_os_release(self.distro)
                self.vfs["/etc/issue"] = self.distro.get("issue", "")
                await self.write_line(f"Switched distro to {chosen}")
            else:
                await self.write_line(f"No such distro: {chosen}")
            self._chan.write(self.prompt())
            return

        if cmd.strip() == "list-distros":
            for k in DISTROS:
                await self.write_line(f"- {k}: {DISTROS[k].get('name')}")
            self._chan.write(self.prompt())
            return

        # 危险指令拦截
        for bad in DANGEROUS:
            if bad in cmd:
                await self.write_line(f"sim-shell: command '{cmd}' is not allowed in simulated environment.")
                self._chan.write(self.prompt())
                return

        # 内置命令快速响应（避免每次都走 LLM）
        if cmd == "cat /etc/os-release":
            out = self.vfs.get("/etc/os-release", "")
            for line in out.splitlines():
                await self.write_line(line)
            self._chan.write(self.prompt()); return

        if cmd == "uname -a":
            out = render_uname(self.distro, self.hostname)
            await self.write_line(out)
            self._chan.write(self.prompt()); return

        if cmd == "ls":
            # 非精确，只列 root 的文件名（演示）
            roots = sorted({p.strip("/") for p in self.vfs.keys() if p.count("/") <= 1})
            await self.write_line("  ".join(roots))
            self._chan.write(self.prompt()); return

        if cmd == "pkg-list":
            out = render_pkg_list(self.distro)
            for line in out.splitlines():
                await self.write_line(line)
            self._chan.write(self.prompt()); return

        # 以上情况都不是 -> 使用 LLM（流式）生成输出并按行写回
        try:
            # ask_llm_stream 返回一个异步生成器（yield 内容片段）
            gen = ask_llm_stream(cmd, self.distro, context="session-limited")
            # 将同步/异步生成器统一封装为异步生成器
            if hasattr(gen, "__aiter__"):
                await self.stream_and_write(gen)
            else:
                # 如果 gen 是普通生成器（fallback），将其包成 async generator
                async def to_async(g):
                    for item in g:
                        yield item
                await self.stream_and_write(to_async(gen))
        except Exception as e:
            await self.write_line(f"Error contacting LLM: {e}")

        # 写回提示符
        self._chan.write(self.prompt())
'''
class SimSSHSession(asyncssh.SSHServerSession):
    def __init__(self, *args, **kwargs):
        self._chan = None
        self._input_buffer = ""

    def connection_made(self, chan):
        self._chan = chan
        self._chan.write("Welcome to SimLinux\n")
        self._chan.write(self.prompt())

    def pty_requested(self, term_type, term_size, term_modes):
        # 返回 True 才能接收输入
        self._term_type = term_type
        self._term_size = term_size
        return True

    def shell_requested(self):
        return True

    def data_received(self, data, datatype):
        # Windows 下 data 可能包含 \r\n
        lines = data.replace('\r', '').split('\n')
        for line in lines:
            line = line.strip()
            if line:
                # 用 asyncio 调用异步处理
                asyncio.create_task(self.handle_command(line))
            else:
                self._chan.write(self.prompt())

    async def handle_command(self, cmd):
        # 模拟 LLM 输出或者内置命令
        self._chan.write(f"Simulated output: {cmd}\n")
        self._chan.write(self.prompt())

    def prompt(self):
        return "user@simlinux:~$ "
#上面是临时
class SimSSHServer(asyncssh.SSHServer):
    def connection_made(self, conn):
        logging.info("Connection from %s", conn.get_extra_info("peername"))

    # 告诉 SSH 服务器需要认证
    def begin_auth(self, username):
        # 返回 True 表示启用密码认证
        return True

    # 告诉 SSH 服务器支持密码认证
    def password_auth_supported(self):
        return True

    # 验证用户名和密码
    def validate_password(self, username, password):
        if username == USERNAME and password == PASSWORD:
            return True
        return False

# -------------------- 启动函数 --------------------
async def start_server():
    # 生成 host key（如果没有）
    if not os.path.exists(HOST_KEY):
        os.system(f"ssh-keygen -t rsa -f {HOST_KEY} -N '' >/dev/null 2>&1")

    await asyncssh.listen(HOST, PORT,
                          server_factory=SimSSHServer,
                          server_host_keys=[HOST_KEY],
                          session_factory=SimSSHSession)

    logging.info("SimLinux server started on %s:%d", HOST, PORT)
    await asyncio.Future()  # 永远运行

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except (OSError, asyncssh.Error) as exc:
        print("Error starting server:", exc)
