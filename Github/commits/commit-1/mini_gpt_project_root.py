# This script implements a minimal GPT-like language model using PyTorch.
# - It defines a simple Transformer-based neural network (MiniGPT) for character-level language modeling.
# - CharTokenizer encodes/decodes text to/from integer tokens.
# - The train() function trains the model on a repeated sample text.
# - The generate() function produces text given a prompt using greedy decoding.
# - Run this script to train the model and generate a sample completion.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import subprocess
import glob
import platform
import webbrowser
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        b, t = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :t, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# Simple character-level tokenizer
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    def encode(self, s):
        return [self.stoi[c] for c in s]
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

# Training loop (toy example)
def train(model, data, tokenizer, block_size, epochs=10, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for i in range(0, len(data) - block_size, block_size):
            x = torch.tensor([tokenizer.encode(data[i:i+block_size])])
            y = torch.tensor([tokenizer.encode(data[i+1:i+block_size+1])])
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Inference (greedy sampling)
def generate(model, tokenizer, prompt, max_new_tokens=100):
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)])
    block_size = model.pos_emb.size(1)
    for _ in range(max_new_tokens):
        # Only feed the last block_size tokens to the model
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        next_id = torch.argmax(logits[0, -1]).unsqueeze(0)
        idx = torch.cat([idx, next_id.unsqueeze(0)], dim=1)
    return tokenizer.decode(idx[0].tolist())

def listen_microphone():
    if sr is None:
        print("speech_recognition not installed. Falling back to text input.")
        return input("You: ")
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source, timeout=5)
        try:
            text = r.recognize_google(audio)
            print(f"You (mic): {text}")
            return text
        except Exception as e:
            print("Could not understand audio:", e)
            return input("Fallback to text input. You: ")
    except Exception as e:
        print("Microphone not available or error occurred:", e)
        return input("Fallback to text input. You: ")

def speak(text):
    if pyttsx3 is not None:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

def search_web(query, mode="text"):
    import requests
    from bs4 import BeautifulSoup
    print(f"Searching for: {query}")
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        # Try to get the featured snippet or first result
        answer = None
        # Featured snippet
        snippet = soup.find("div", class_="BNeawe s3v9rd AP7Wnd")
        if snippet:
            answer = snippet.get_text()
        # Fallback: first result description
        if not answer:
            desc = soup.find("div", class_="BNeawe vvjwJb AP7Wnd")
            if desc:
                answer = desc.get_text()
        # Fallback: first paragraph or text
        if not answer:
            paragraphs = soup.find_all("p")
            if paragraphs:
                answer = paragraphs[0].get_text()
        if not answer:
            answer = "No direct answer found, but I've opened the search in your browser."
        print("Search result:", answer)
        if mode == "mic" and pyttsx3 is not None:
            speak(answer)
        # Always open the browser for more info
        webbrowser.open(url)
    except Exception as e:
        print("Could not perform web search:", e)
        if mode == "mic" and pyttsx3 is not None:
            speak("Sorry, I could not get a search result.")

def solve_code_error(code, error_message, language="python"):
    """
    Placeholder for code error solving.
    In a real system, this would use a large language model API (e.g., OpenAI GPT-4, Code Llama).
    """
    print(f"\n[Assistant] Detected error in {language} code:")
    print("Error message:", error_message)
    print("Attempting to solve... (stub)")
    # Here you would call an LLM API with the code and error_message as context.
    print("[Assistant] Solution: (This is a placeholder. Integrate with an LLM for real solutions.)\n")

def find_app_path(app_name):
    """
    Search for an application executable in common system paths.
    Returns the full path if found, else None.
    """
    app_name = app_name.strip().lower()
    # Accept any file extension, not just .exe/.lnk
    possible_names = [app_name]
    if os.name == "nt" and not os.path.splitext(app_name)[1]:
        possible_names += [app_name + ext for ext in [".exe", ".lnk", ".bat", ".cmd", ".com"]]
    # Search Desktop, Program Files, Windows, PATH, etc.
    search_dirs = [os.path.join(os.path.expanduser("~"), "Desktop")]
    if os.name == "nt":
        search_dirs += [
            os.path.join(os.environ.get("ProgramFiles", ""), ""),
            os.path.join(os.environ.get("ProgramFiles(x86)", ""), ""),
            os.path.join(os.environ.get("SystemRoot", ""), ""),
            os.path.join(os.environ.get("SystemDrive", ""), "\\"),
        ]
    search_dirs += os.environ.get("PATH", "").split(os.pathsep)
    for search_dir in search_dirs:
        if not search_dir or not os.path.isdir(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                file_lower = file.lower()
                for name in possible_names:
                    if file_lower == name:
                        return os.path.join(root, file)
    return None

def close_app_by_name(app_name):
    """
    Attempt to close an application by searching for its process name.
    """
    app_name = app_name.strip().lower()
    if os.name == "nt":
        # Try to kill by process name
        try:
            subprocess.call(f"taskkill /im {app_name}.exe /f", shell=True)
            subprocess.call(f"taskkill /im {app_name} /f", shell=True)
        except Exception as e:
            print(f"Error closing app: {e}")
    else:
        # Linux/Mac
        try:
            subprocess.call(["pkill", "-f", app_name])
        except Exception as e:
            print(f"Error closing app: {e}")

def write_in_app(app_name, content):
    """
    Try to write content in a running app's window (Windows only, Notepad example).
    """
    if os.name != "nt":
        print("Writing in app is only supported on Windows for Notepad in this demo.")
        return
    import pyautogui
    import time
    import win32gui
    import win32con

    # Try to find the app window
    def enum_handler(hwnd, result):
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd).lower()
            if app_name.lower() in window_text:
                result.append(hwnd)
    hwnds = []
    win32gui.EnumWindows(enum_handler, hwnds)
    if not hwnds:
        print(f"No open window found for '{app_name}'.")
        return
    # Bring the first found window to foreground
    hwnd = hwnds[0]
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.5)
    pyautogui.typewrite(content)
    print(f"Wrote '{content}' in {app_name} window.")

def open_in_app(app_name, target):
    """
    Try to open a file or page in an open app (Notepad or browser demo).
    """
    if os.name != "nt":
        print("Open in app is only demoed for Windows (Notepad, browsers).")
        return
    import pyautogui
    import time
    import win32gui
    import win32con

    # For browsers, just open the URL
    browsers = ["chrome", "firefox", "edge", "opera", "brave"]
    if any(b in app_name.lower() for b in browsers):
        import webbrowser
        webbrowser.open(target)
        print(f"Opened {target} in browser.")
        return

    # For Notepad, send Ctrl+O and type filename
    if "notepad" in app_name.lower():
        def enum_handler(hwnd, result):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd).lower()
                if app_name.lower() in window_text:
                    result.append(hwnd)
        hwnds = []
        win32gui.EnumWindows(enum_handler, hwnds)
        if not hwnds:
            print(f"No open window found for '{app_name}'.")
            return
        hwnd = hwnds[0]
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.5)
        pyautogui.hotkey('ctrl', 'o')
        time.sleep(0.5)
        pyautogui.typewrite(target)
        pyautogui.press('enter')
        print(f"Opened {target} in Notepad.")

def parse_and_execute(command, input_mode="text"):
    # Try smart intent first
    if smart_intent(command):
        return
    cmd = command.lower()
    # Use only one "if/elif/else" chain for all command handling
    if cmd == "open chrome":
        print("Opening Chrome browser...")
        if os.name == "nt":
            chrome_path = find_app_path("chrome")
            if chrome_path:
                os.startfile(chrome_path)
            else:
                print("Chrome not found.")
        else:
            subprocess.Popen(["google-chrome"])
        return
    elif cmd.startswith("search "):
        query = command[7:].strip()
        search_web(query, mode=input_mode)
        return
    # Optionally, handle "find", "lookup", or "google" as search triggers
    elif cmd.startswith("find "):
        query = command[5:].strip()
        search_web(query, mode=input_mode)
        return
    elif cmd.startswith("lookup "):
        query = command[7:].strip()
        search_web(query, mode=input_mode)
        return
    elif cmd.startswith("google "):
        query = command[7:].strip()
        search_web(query, mode=input_mode)
        return
    elif cmd == "open all apps on the desktop":
        # Open all apps (shortcuts and executables) on the user's Desktop
        try:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            files = os.listdir(desktop)
            opened = 0
            for f in files:
                path = os.path.join(desktop, f)
                if os.path.isfile(path) and (f.endswith(".lnk") or f.endswith(".exe") or f.endswith(".app")):
                    try:
                        os.startfile(path)
                        print(f"Opened: {f}")
                        opened += 1
                    except Exception as e:
                        print(f"Could not open {f}: {e}")
            if opened == 0:
                print("No apps found to open on the Desktop.")
        except Exception as e:
            print("Error opening apps on the Desktop:", e)
        return
    elif cmd.startswith("open "):
        app = cmd[5:].strip()
        print(f"Searching for and opening {app}...")
        app_path = find_app_path(app)
        if app_path:
            try:
                if os.name == "nt":
                    os.startfile(app_path)
                else:
                    subprocess.Popen([app_path])
                print(f"Opened: {app_path}")
            except Exception as e:
                print(f"Error opening {app}: {e}")
        else:
            print(f"Could not find application '{app}'. Try specifying the exact .exe or .lnk filename if on Windows.")
    elif cmd.startswith("close "):
        app = cmd[6:].strip()
        print(f"Searching for and closing {app}...")
        close_app_by_name(app)
        print(f"Attempted to close '{app}'.")
    elif "write" in cmd and ("on a file called" in cmd or "to file" in cmd or "in file" in cmd):
        # Example: write hello world on a file called indexy.html
        # Supports: "write ... on a file called ...", "write ... to file ...", "write ... in file ..."
        try:
            if "on a file called" in cmd:
                content = cmd.split("write",1)[1].split("on a file called")[0].strip()
                filename = cmd.split("on a file called",1)[1].strip()
            elif "to file" in cmd:
                content = cmd.split("write",1)[1].split("to file")[0].strip()
                filename = cmd.split("to file",1)[1].strip()
            elif "in file" in cmd:
                content = cmd.split("write",1)[1].split("in file")[0].strip()
                filename = cmd.split("in file",1)[1].strip()
            else:
                print("Could not parse write command.")
                return
            # Support writing to any text file extension
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Wrote '{content}' to {filename}")
            auto_commit(
                f"Wrote content to file '{filename}' via command: {command}",
                diff=f"File '{filename}' created/overwritten with content: {content}\n"
            )
        except Exception as e:
            print("Error writing file:", e)
    elif cmd.startswith("create file "):
        # Example: create file index.html
        try:
            filename = cmd.split("create file ",1)[1].strip()
            with open(filename, "w", encoding="utf-8") as f:
                f.write("")
            print(f"Created file {filename}")
            auto_commit(
                f"Created file '{filename}' via command: {command}",
                diff=f"File '{filename}' created as empty file.\n"
            )
        except Exception as e:
            print("Error creating file:", e)
    elif cmd.startswith("destroy file "):
        # Example: destroy file index.html
        try:
            filename = cmd.split("destroy file ",1)[1].strip()
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Destroyed file {filename}")
                auto_commit(
                    f"Destroyed file '{filename}' via command: {command}",
                    diff=f"File '{filename}' deleted from filesystem.\n"
                )
            else:
                print(f"File {filename} does not exist.")
        except Exception as e:
            print("Error destroying file:", e)
    elif cmd.startswith("cd "):
        # Change directory
        try:
            path = command[3:].strip()
            os.chdir(path)
            print(f"Changed directory to {os.getcwd()}")
            auto_commit(
                f"Changed directory to {os.getcwd()} via command: {command}",
                diff=f"Working directory changed to: {os.getcwd()}\n"
            )
        except Exception as e:
            print("Error changing directory:", e)
    elif cmd.strip() == "ls" or cmd.strip() == "dir":
        # List files in current directory
        try:
            files = os.listdir()
            for f in files:
                print(f)
            auto_commit(
                f"Listed files in directory {os.getcwd()} via command: {command}",
                diff=f"Directory listing: {os.listdir()}\n"
            )
        except Exception as e:
            print("Error listing directory:", e)
    elif cmd.startswith("pwd"):
        # Print working directory
        print(os.getcwd())
        auto_commit(
            f"Printed working directory via command: {command}",
            diff=f"Current working directory: {os.getcwd()}\n"
        )
    elif cmd.strip() == "show commit history":
        show_commit_history()
        return
    elif cmd.startswith("open a file on ") or cmd.startswith("open file "):
        # Support both "open a file on filename" and "open file filename"
        if cmd.startswith("open a file on "):
            filename = command[len("open a file on "):].strip()
        else:
            filename = command[len("open file "):].strip()
        filename = os.path.abspath(filename)
        if not os.path.exists(filename):
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    pass
                print(f"File '{filename}' did not exist and was created.")
                auto_commit(
                    f"Created file '{filename}' via command: {command}",
                    diff=f"File '{filename}' created as empty file.\n"
                )
            except Exception as e:
                print(f"Could not create file '{filename}': {e}")
                return
        else:
            print(f"File '{filename}' already exists.")
        # Open the file with the default application
        try:
            if os.name == "nt":
                os.startfile(filename)
            else:
                subprocess.Popen(["xdg-open", filename])
            print(f"Opened file: {filename}")
            auto_commit(
                f"Opened file '{filename}' via command: {command}",
                diff=f"File '{filename}' opened with default application.\n"
            )
        except Exception as e:
            print(f"Could not open file '{filename}': {e}")
        return
    else:
        # Self-learning: if seen before, simulate that the model now knows how to do it
        if cmd in self_learned_tasks:
            print(f"[MiniGPT] I have learned how to handle this task since last time!")
            print(self_learned_tasks[cmd])
            print("If you want to learn how to do this task, type 'help' or check minigpt_commands.txt for supported commands.")
        else:
            print("Command not recognized or supported.")
            learn_from_error(command, "Command not recognized or supported.")

# Memory for error-driven learning (in-memory, not persistent)
error_memory = []
self_learned_tasks = {}

def learn_from_error(prompt, error_message):
    """
    Store prompt and error for future learning.
    Simulate self-training: after first failure, next time the same prompt is seen, it will be 'understood'.
    Additionally, provide a message to help the user learn how to do the task.
    """
    error_memory.append({"prompt": prompt, "error": error_message})
    print("[MiniGPT] Noted this error for future learning.")
    # Simulate self-training: after first error, learn the task
    self_learned_tasks[prompt.strip().lower()] = (
        f"Simulated learned solution for: {prompt}\n"
        "Tip: If you want to perform this task, try rephrasing your command or use one of the supported commands. "
        "Type 'help' or see the commands list for guidance."
    )

def review_error_memory():
    """
    Display all stored errors and prompts.
    """
    if not error_memory:
        print("No errors or unknowns have been recorded yet.")
        return
    print("Learned from the following errors/unknowns:")
    for i, entry in enumerate(error_memory, 1):
        print(f"{i}. Prompt: {entry['prompt']}\n   Error: {entry['error']}")

def linux_terminal():
    import shutil
    import sys
    import getpass
    import socket
    try:
        import readline  # For command history and editing (Linux/macOS)
    except ImportError:
        readline = None  # On Windows, readline may not be available
    from datetime import datetime

    print("\n--- MiniGPT Linux-like Terminal ---")
    print("Type 'exit' to leave the terminal.\n")
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    # Check for common Linux tools and warn if missing
    required_tools = [
        "ls", "cat", "grep", "awk", "sed", "curl", "wget", "nano", "vim", "top", "htop",
        "ps", "kill", "chmod", "chown", "df", "du", "tar", "gzip", "unzip", "ssh",
        "scp", "ping", "ifconfig", "netstat", "airmon-ng", "msfconsole"
    ]
    missing_tools = [tool for tool in required_tools if shutil.which(tool) is None]
    if missing_tools:
        print(f"{YELLOW}Warning: The following Linux tools are not found in your PATH and some commands may not work as expected:{RESET}")
        print(", ".join(missing_tools))
        print("To install these tools, use your system's package manager (e.g., apt, yum, dnf, zypper, pacman, brew, choco).")
        print("Example for Ubuntu/Debian:\n  sudo apt update && sudo apt install " + " ".join(missing_tools))

    # Built-in commands for a more powerful terminal
    def builtin_help():
        print(f"""{CYAN}
MiniGPT Linux Terminal - Built-in Commands:
  help                Show this help message
  clear               Clear the terminal screen
  history             Show command history
  time                Show current system time
  whoami              Show current user
  hostname            Show system hostname
  sudo su             Open a new administrator/root shell (Windows: opens admin cmd)
  exit                Exit the terminal
  Any other command is executed by your system shell.
{RESET}""")

    history = []

    while True:
        try:
            cwd = os.getcwd()
            user = getpass.getuser()
            host = socket.gethostname()
            prompt = f"{GREEN}{user}@{host}:{cwd}$ {RESET}"
            cmd = input(prompt)
            if not cmd.strip():
                continue
            history.append(cmd)
            if cmd.strip() == "exit":
                break
            if cmd.strip() == "help":
                builtin_help()
                continue
            if cmd.strip() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            if cmd.strip() == "history":
                for i, h in enumerate(history, 1):
                    print(f"{i}  {h}")
                continue
            if cmd.strip() == "time":
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                continue
            if cmd.strip() == "whoami":
                print(user)
                continue
            if cmd.strip() == "hostname":
                print(host)
                continue
            if cmd.strip() == "sudo su":
                if os.name == "nt":
                    # Windows: open admin cmd
                    print("Opening Administrator Command Prompt...")
                    try:
                        subprocess.run('powershell -Command "Start-Process cmd -Verb runAs"', shell=True)
                    except Exception as e:
                        print(f"Failed to open admin cmd: {e}")
                else:
                    # Linux/Mac: open root shell
                    print("Switching to root shell (requires password)...")
                    try:
                        subprocess.run("sudo su", shell=True)
                    except Exception as e:
                        print(f"Failed to open root shell: {e}")
                continue
            # Special handling for airmon-ng
            if cmd.strip().startswith("airmon-ng"):
                print("Running airmon-ng (requires system support and root privileges)...")
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.stdout:
                        print(result.stdout, end="")
                    if result.stderr:
                        print(result.stderr, end="")
                except Exception as e:
                    print(f"airmon-ng error: {e}")
                continue
            # Special handling for metasploit (msfconsole)
            if cmd.strip().startswith("msfconsole") or cmd.strip().startswith("metasploit"):
                print("Launching Metasploit Framework (requires system support and installation)...")
                msf_path = shutil.which("msfconsole")
                if not msf_path:
                    print("Metasploit is not installed or not in your PATH. Please install Metasploit Framework and ensure 'msfconsole' is available in your system PATH.")
                    print("See installation instructions above in the script comments.")
                    continue
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.stdout:
                        print(result.stdout, end="")
                    if result.stderr:
                        print(result.stderr, end="")
                except Exception as e:
                    print(f"Metasploit error: {e}")
                continue
            # Try to handle with Python if possible, else pass to system shell
            if cmd.startswith("cd "):
                try:
                    path = cmd[3:].strip()
                    os.chdir(path)
                except Exception as e:
                    print(f"cd: {e}")
            else:
                try:
                    # Use /bin/bash if available for better Linux compatibility
                    bash_path = shutil.which("bash")
                    if bash_path:
                        result = subprocess.run(cmd, shell=True, executable=bash_path, capture_output=True, text=True)
                    else:
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.stdout:
                        print(result.stdout, end="")
                    if result.stderr:
                        print(result.stderr, end="")
                        # Learn from errors in terminal commands
                        learn_from_error(cmd, result.stderr)
                except Exception as e:
                    print(f"Command error: {e}")
                    learn_from_error(cmd, str(e))
        except (KeyboardInterrupt, EOFError):
            print("\nExiting terminal.")
            break

        # Add a command to review error memory in the terminal
            if cmd.strip() == "review errors":
                review_error_memory()
                continue

# Import commit history functions from LLM-History/history.py
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "LLM-History"))
try:
    from history import auto_commit, show_commit_history
except ImportError:
    # Fallback to local dummy if not found
    def auto_commit(description, diff=None, path=None, lines=None, explanation=None):
        pass
    def show_commit_history():
        print("Commit history module not found.")

# Automatically log every edit/commit by this AI
def auto_commit(description, diff=None):
    log_commit(description, diff)

# Example usage: call auto_commit after every code change (for demonstration, add at the end of parse_and_execute)
def parse_and_execute(command, input_mode="text"):
    # Try smart intent first
    if smart_intent(command):
        return
    cmd = command.lower()
    # Use only one "if/elif/else" chain for all command handling
    if cmd == "open chrome":
        print("Opening Chrome browser...")
        if os.name == "nt":
            chrome_path = find_app_path("chrome")
            if chrome_path:
                os.startfile(chrome_path)
            else:
                print("Chrome not found.")
        else:
            subprocess.Popen(["google-chrome"])
        return
    elif cmd.startswith("search "):
        query = command[7:].strip()
        search_web(query, mode=input_mode)
        return
    # Optionally, handle "find", "lookup", or "google" as search triggers
    elif cmd.startswith("find "):
        query = command[5:].strip()
        search_web(query, mode=input_mode)
        return
    elif cmd.startswith("lookup "):
        query = command[7:].strip()
        search_web(query, mode=input_mode)
        return
    elif cmd.startswith("google "):
        query = command[7:].strip()
        search_web(query, mode=input_mode)
        return
    elif cmd == "open all apps on the desktop":
        # Open all apps (shortcuts and executables) on the user's Desktop
        try:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            files = os.listdir(desktop)
            opened = 0
            for f in files:
                path = os.path.join(desktop, f)
                if os.path.isfile(path) and (f.endswith(".lnk") or f.endswith(".exe") or f.endswith(".app")):
                    try:
                        os.startfile(path)
                        print(f"Opened: {f}")
                        opened += 1
                    except Exception as e:
                        print(f"Could not open {f}: {e}")
            if opened == 0:
                print("No apps found to open on the Desktop.")
        except Exception as e:
            print("Error opening apps on the Desktop:", e)
        return
    elif cmd.startswith("open "):
        app = cmd[5:].strip()
        print(f"Searching for and opening {app}...")
        app_path = find_app_path(app)
        if app_path:
            try:
                if os.name == "nt":
                    os.startfile(app_path)
                else:
                    subprocess.Popen([app_path])
                print(f"Opened: {app_path}")
            except Exception as e:
                print(f"Error opening {app}: {e}")
        else:
            print(f"Could not find application '{app}'. Try specifying the exact .exe or .lnk filename if on Windows.")
    elif cmd.startswith("close "):
        app = cmd[6:].strip()
        print(f"Searching for and closing {app}...")
        close_app_by_name(app)
        print(f"Attempted to close '{app}'.")
    elif "write" in cmd and ("on a file called" in cmd or "to file" in cmd or "in file" in cmd):
        # Example: write hello world on a file called indexy.html
        # Supports: "write ... on a file called ...", "write ... to file ...", "write ... in file ..."
        try:
            if "on a file called" in cmd:
                content = cmd.split("write",1)[1].split("on a file called")[0].strip()
                filename = cmd.split("on a file called",1)[1].strip()
            elif "to file" in cmd:
                content = cmd.split("write",1)[1].split("to file")[0].strip()
                filename = cmd.split("to file",1)[1].strip()
            elif "in file" in cmd:
                content = cmd.split("write",1)[1].split("in file")[0].strip()
                filename = cmd.split("in file",1)[1].strip()
            else:
                print("Could not parse write command.")
                return
            # Support writing to any text file extension
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Wrote '{content}' to {filename}")
            auto_commit(
                f"Wrote content to file '{filename}' via command: {command}",
                diff=f"File '{filename}' created/overwritten with content: {content}\n"
            )
        except Exception as e:
            print("Error writing file:", e)
    elif cmd.startswith("create file "):
        # Example: create file index.html
        try:
            filename = cmd.split("create file ",1)[1].strip()
            with open(filename, "w", encoding="utf-8") as f:
                f.write("")
            print(f"Created file {filename}")
            auto_commit(
                f"Created file '{filename}' via command: {command}",
                diff=f"File '{filename}' created as empty file.\n"
            )
        except Exception as e:
            print("Error creating file:", e)
    elif cmd.startswith("destroy file "):
        # Example: destroy file index.html
        try:
            filename = cmd.split("destroy file ",1)[1].strip()
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Destroyed file {filename}")
                auto_commit(
                    f"Destroyed file '{filename}' via command: {command}",
                    diff=f"File '{filename}' deleted from filesystem.\n"
                )
            else:
                print(f"File {filename} does not exist.")
        except Exception as e:
            print("Error destroying file:", e)
    elif cmd.startswith("cd "):
        # Change directory
        try:
            path = command[3:].strip()
            os.chdir(path)
            print(f"Changed directory to {os.getcwd()}")
            auto_commit(
                f"Changed directory to {os.getcwd()} via command: {command}",
                diff=f"Working directory changed to: {os.getcwd()}\n"
            )
        except Exception as e:
            print("Error changing directory:", e)
    elif cmd.strip() == "ls" or cmd.strip() == "dir":
        # List files in current directory
        try:
            files = os.listdir()
            for f in files:
                print(f)
            auto_commit(
                f"Listed files in directory {os.getcwd()} via command: {command}",
                diff=f"Directory listing: {os.listdir()}\n"
            )
        except Exception as e:
            print("Error listing directory:", e)
    elif cmd.startswith("pwd"):
        # Print working directory
        print(os.getcwd())
        auto_commit(
            f"Printed working directory via command: {command}",
            diff=f"Current working directory: {os.getcwd()}\n"
        )
    elif cmd.strip() == "show commit history":
        show_commit_history()
        return
    elif cmd.startswith("open a file on ") or cmd.startswith("open file "):
        # Support both "open a file on filename" and "open file filename"
        if cmd.startswith("open a file on "):
            filename = command[len("open a file on "):].strip()
        else:
            filename = command[len("open file "):].strip()
        filename = os.path.abspath(filename)
        if not os.path.exists(filename):
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    pass
                print(f"File '{filename}' did not exist and was created.")
                auto_commit(
                    f"Created file '{filename}' via command: {command}",
                    diff=f"File '{filename}' created as empty file.\n"
                )
            except Exception as e:
                print(f"Could not create file '{filename}': {e}")
                return
        else:
            print(f"File '{filename}' already exists.")
        # Open the file with the default application
        try:
            if os.name == "nt":
                os.startfile(filename)
            else:
                subprocess.Popen(["xdg-open", filename])
            print(f"Opened file: {filename}")
            auto_commit(
                f"Opened file '{filename}' via command: {command}",
                diff=f"File '{filename}' opened with default application.\n"
            )
        except Exception as e:
            print(f"Could not open file '{filename}': {e}")
        return
    else:
        # Self-learning: if seen before, simulate that the model now knows how to do it
        if cmd in self_learned_tasks:
            print(f"[MiniGPT] I have learned how to handle this task since last time!")
            print(self_learned_tasks[cmd])
            print("If you want to learn how to do this task, type 'help' or check minigpt_commands.txt for supported commands.")
        else:
            print("Command not recognized or supported.")
            learn_from_error(command, "Command not recognized or supported.")

# Explanation:
# The current "self-learning" feature only simulates learning by storing prompts and returning a canned response.
# It does NOT actually update the neural network or teach the user how to do the task.
# To truly "train the user," you could display a step-by-step explanation or instructions for the failed command.

# Note:
# A language model project does not have to be a single file.
# For small demos, one file is fine. For larger or production projects, it is better to split code into multiple files:
# - Model definition (e.g., model.py)
# - Training logic (e.g., train.py)
# - Utilities (e.g., utils.py)
# - Inference or CLI (e.g., main.py)
# This improves readability, maintainability, and testing.

# To install the speech_recognition package, run:
# pip install SpeechRecognition

# To install Metasploit Framework on Windows:
# 1. Download the Windows installer from: https://windows.metasploit.com/metasploitframework-latest.msi
# 2. Run the installer and follow the prompts.
# 3. After installation, add the Metasploit directory (containing msfconsole.bat) to your system PATH.
# 4. Open a new Command Prompt and run: msfconsole

# For more details, see: https://docs.metasploit.com/docs/using-metasploit/getting-started/nightly-installers.html

# After installation, you can run 'msfconsole' from your terminal.
# Make sure 'msfconsole' is in your system PATH to use it from this script.

# To remove firewall rules and allow Metasploit or other tools to work (Windows example):
# 1. Open Command Prompt as Administrator.
# 2. To list all firewall rules:
#    netsh advfirewall firewall show rule name=all
# 3. To delete a specific rule (replace "RuleName" with the actual rule name):
#    netsh advfirewall firewall delete rule name="RuleName"
# 4. To disable Windows Firewall completely (not recommended for security reasons):
#    netsh advfirewall set allprofiles state off

# On Linux, use (as root/sudo):
#   ufw disable
#   or to delete a rule:
#   ufw delete <rule_number>
#   or with iptables:
#   iptables -F

# WARNING: Disabling or removing firewall rules can expose your system to security risks.
# Only do this in a safe, controlled environment and re-enable your firewall when done.

# To install missing Linux tools, use your system's package manager.
# For Ubuntu/Debian, run the following in your terminal (not in Python):
# sudo apt update
# sudo apt install ls cat grep awk sed curl wget nano vim top htop ps kill chmod chown df du tar gzip unzip ssh scp ping ifconfig netstat airmon-ng metasploit-framework

# For Fedora/CentOS/RHEL:
# sudo dnf install coreutils grep gawk sed curl wget nano vim util-linux procps-ng htop openssh-clients net-tools aircrack-ng metasploit

# For Arch Linux:
# sudo pacman -S coreutils grep gawk sed curl wget nano vim procps-ng htop openssh net-tools aircrack-ng metasploit

# On macOS (with Homebrew):
# brew install coreutils grep gawk gnu-sed curl wget nano vim htop openssh net-tools aircrack-ng metasploit

# On Windows, you can install many Linux-like tools using Chocolatey:
# 1. Open PowerShell as Administrator.
# 2. Install Chocolatey if you haven't already:
#    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
# 3. Then install tools (as Administrator):
#    choco install git grep sed gawk curl wget nano vim busybox openssh gnuwin32-coreutils procps-ng netcat nmap aircrack-ng metasploit

# Alternatively, you can use Cygwin or Windows Subsystem for Linux (WSL) for a full Linux environment.

# NOTE: The ChatGPT dataset (OpenAI's conversational data) is not publicly available.
# For further training, you can use open-source conversational datasets such as:
# - OpenAssistant Conversations (https://huggingface.co/datasets/OpenAssistant/oasst1)
# - ShareGPT (https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
# - Alpaca, Dolly, or other instruction-tuning datasets

# Example: Using OpenAssistant Conversations for further training
# (Requires: pip install datasets)

# from datasets import load_dataset
# dataset = load_dataset("OpenAssistant/oasst1")
# # Concatenate all messages for simple character-level training
# text_data = "\n".join([msg["text"] for msg in dataset["train"] if "text" in msg])
# tokenizer = CharTokenizer(text_data)
# vocab_size = len(tokenizer.chars)
# block_size = 32
# model = MiniGPT(vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=block_size)
# train(model, text_data, tokenizer, block_size, epochs=5)

# To use this, install the datasets library:
# pip install datasets

# About the OpenAssistant Conversations dataset:
# - The dataset contains multi-turn conversations between users and assistants.
# - Each entry includes fields such as:
#   - "text": The message content (user or assistant)
#   - "role": "user" or "assistant"
#   - "parent_id": ID of the previous message in the conversation
#   - "message_id": Unique message identifier
#   - "lang": Language code (e.g., "en")
#   - "rank": Ranking information (for some splits)
#   - "labels": Quality or moderation labels (optional)
# - The dataset is designed for training and evaluating conversational AI models.
# - See: https://huggingface.co/datasets/OpenAssistant/oasst1

if __name__ == "__main__":
    # Example usage
    text = "hello world. this is a minimal gpt example. " * 100
    tokenizer = CharTokenizer(text)
    vocab_size = len(tokenizer.chars)
    block_size = 32
    model = MiniGPT(vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=block_size)
    train(model, text, tokenizer, block_size, epochs=5)
    prompt = "hello"
    print("ChatGPT:", generate(model, tokenizer, prompt, max_new_tokens=50))
    print("\n--- Voice/Text Command Mode ---")
    print("Say or type commands like:")
    print("  open notepad")
    print("  close notepad")
    print("  write hello world on a file called indexy.html")
    print("  create file index.html")
    print("  destroy file index.html")
    print("Type 'exit' to return to mode selection.\n")
    print("Type 'terminal' to enter a Linux-like shell connected to your system.\n")

    while True:
        mode = input("Input mode? (mic/text/terminal/exit): ").strip().lower()
        if mode == "exit":
            break
        if mode == "terminal":
            linux_terminal()
            continue
        elif mode == "mic":
            if sr is None:
                print("speech_recognition not installed. Falling back to text input.")
            print("Mic mode. Say 'exit' to return to mode selection.")
            while True:
                command = listen_microphone()
                if command.strip().lower() == "exit":
                    break
                parse_and_execute(command, input_mode="mic")
        elif mode == "text":
            print("Text mode. Type 'exit' to return to mode selection.")
            while True:
                command = input("You: ")
                if command.strip().lower() == "exit":
                    break
                parse_and_execute(command, input_mode="text")
        else:
            print("Unknown mode. Use 'mic', 'text', 'terminal', or 'exit'.")
