"""Run a logged command and terminate its owned process tree on interruption."""

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


class OwnedProcessCleanupError(RuntimeError):
    pass


class WindowsJob:
    def __init__(self):
        import ctypes
        from ctypes import wintypes as w

        class Limits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong), ("flags", w.DWORD),
                        ("minimum_working_set", ctypes.c_size_t), ("maximum_working_set", ctypes.c_size_t),
                        ("active_processes", w.DWORD), ("affinity", ctypes.c_size_t),
                        ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", Limits), ("io", ctypes.c_ulonglong * 6), ("process_memory", ctypes.c_size_t),
                        ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t),
                        ("peak_job_memory", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {"CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
                      "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
                      "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
                      "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
                      "QueryInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL),
                      "OpenProcess": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
                      "IsProcessInJob": ([w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)], w.BOOL),
                      "WaitForSingleObject": ([w.HANDLE, w.DWORD], w.DWORD),
                      "CloseHandle": ([w.HANDLE], w.BOOL)}
        for name, (arguments, result) in signatures.items():
            getattr(kernel, name).argtypes = arguments
            getattr(kernel, name).restype = result
        self.kernel, self.ctypes = kernel, ctypes
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000
        if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach_and_resume(self, process):
        ctypes, kernel = self.ctypes, self.kernel
        from ctypes import wintypes as w

        if not kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

        class ThreadEntry(ctypes.Structure):
            _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("thread", w.DWORD), ("process", w.DWORD),
                        ("base_priority", w.LONG), ("delta_priority", w.LONG), ("flags", w.DWORD)]

        signatures = {"CreateToolhelp32Snapshot": ([w.DWORD, w.DWORD], w.HANDLE),
                      "Thread32First": ([w.HANDLE, ctypes.POINTER(ThreadEntry)], w.BOOL),
                      "Thread32Next": ([w.HANDLE, ctypes.POINTER(ThreadEntry)], w.BOOL),
                      "OpenThread": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
                      "ResumeThread": ([w.HANDLE], w.DWORD)}
        for name, (arguments, result) in signatures.items():
            getattr(kernel, name).argtypes = arguments
            getattr(kernel, name).restype = result
        snapshot = kernel.CreateToolhelp32Snapshot(4, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        entry, resumed = ThreadEntry(), False
        entry.size = ctypes.sizeof(entry)
        try:
            present = kernel.Thread32First(snapshot, ctypes.byref(entry))
            while present:
                if entry.process == process.pid:
                    thread = kernel.OpenThread(2, False, entry.thread)
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        count = kernel.ResumeThread(thread)
                        if count == 0xFFFFFFFF:
                            raise ctypes.WinError(ctypes.get_last_error())
                        if count > 1:
                            raise RuntimeError("Owned main thread has an unexpected suspension count")
                        resumed = count == 1
                    finally:
                        kernel.CloseHandle(thread)
                    if resumed:
                        break
                present = kernel.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            kernel.CloseHandle(snapshot)
        if not resumed:
            raise RuntimeError("The suspended owned process has no resumable main thread")

    def terminate(self):
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def finish(self):
        ctypes = self.ctypes
        from ctypes import wintypes as w

        class Accounting(ctypes.Structure):
            _fields_ = [("times", ctypes.c_longlong * 4), ("page_faults", w.DWORD), ("total", w.DWORD),
                        ("active", w.DWORD), ("terminated", w.DWORD)]

        handles = []
        try:
            buffer = ctypes.create_string_buffer(8 + ctypes.sizeof(ctypes.c_size_t) * 1024)
            if not self.kernel.QueryInformationJobObject(self.handle, 3, buffer, len(buffer), None):
                raise ctypes.WinError(ctypes.get_last_error())
            count = w.DWORD.from_buffer(buffer, 4).value
            identifiers = (ctypes.c_size_t * count).from_buffer(buffer, 8)
            for identifier in identifiers:
                process = self.kernel.OpenProcess(0x100000 | 0x1000, False, identifier)
                if not process:
                    if ctypes.get_last_error() == 87:
                        continue
                    raise ctypes.WinError(ctypes.get_last_error())
                member = w.BOOL()
                if not self.kernel.IsProcessInJob(process, self.handle, ctypes.byref(member)):
                    self.kernel.CloseHandle(process)
                    raise ctypes.WinError(ctypes.get_last_error())
                if member.value:
                    handles.append(process)
                else:
                    self.kernel.CloseHandle(process)
            self.terminate()
            deadline = time.monotonic() + 10
            for process in handles:
                result = self.kernel.WaitForSingleObject(process, max(1, int((deadline - time.monotonic()) * 1000)))
                if result != 0:
                    raise RuntimeError("Owned Windows process did not release its resources")
            while True:
                value = Accounting()
                if not self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(value), ctypes.sizeof(value), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                if value.active == 0:
                    return
                if time.monotonic() >= deadline:
                    raise RuntimeError("Owned Windows descendants did not finish terminating")
                time.sleep(.01)
        finally:
            for process in handles:
                self.kernel.CloseHandle(process)
            self.close()

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


OWNER_KEY = "UNIBRAIN_OWNED_PROCESS_TOKENS"


def linux_identity(path, token):
    try:
        if path.stat().st_uid != os.getuid():
            return None
        entries = (path / "environ").read_bytes().split(b"\0")
        marker = OWNER_KEY.encode() + b"="
        owner = next((entry[len(marker):] for entry in entries if entry.startswith(marker)), b"")
        if token.encode() not in owner.split(b","):
            return None
        return (path / "stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def stop_linux_owned(token):
    deadline = time.monotonic() + 5
    while True:
        found = False
        for path in Path("/proc").iterdir():
            if not path.name.isdecimal() or int(path.name) == os.getpid():
                continue
            identity = linux_identity(path, token)
            if identity is None:
                continue
            found = True
            if linux_identity(path, token) == identity:
                try:
                    os.kill(int(path.name), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if not found:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Owned Linux descendants could not be terminated")
        time.sleep(.01)


def run_process(command, timeout, cwd, env, stream=None, input_text=None):
    if timeout is not None and timeout <= 0:
        raise ValueError("Timeout must be positive")
    environment = dict(os.environ if env is None else env)
    token = uuid.uuid4().hex if sys.platform.startswith("linux") else None
    if token:
        inherited = {value for source in (os.environ, environment) for value in source.get(OWNER_KEY, "").split(",") if value}
        environment[OWNER_KEY] = ",".join(sorted(inherited | {token}))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    job = WindowsJob() if os.name == "nt" else None
    process = None
    try:
        process = subprocess.Popen(command, cwd=cwd, env=environment,
                                   stdin=subprocess.PIPE if stream is None else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE if stream is None else stream,
                                   stderr=subprocess.PIPE if stream is None else subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   creationflags=flags | (4 if job else 0), start_new_session=os.name != "nt")
        if job:
            # Assign the suspended launcher before it can create descendants.
            job.attach_and_resume(process)
        if stream is None:
            stdout, stderr = process.communicate(input=input_text, timeout=timeout)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        return process.wait(timeout=timeout)
    finally:
        try:
            if job:
                job.finish()
            elif token:
                stop_linux_owned(token)
            elif process is not None:
                # Other POSIX platforms retain process-group ownership only.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except Exception as error:
            raise OwnedProcessCleanupError("Owned process tree cleanup failed; stop subsequent jobs") from error
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()


def run_logged(command, log, timeout, cwd=None, env=None):
    with Path(log).open("w", encoding="utf-8") as stream:
        return run_process(command, timeout, cwd, env, stream=stream)


def run_capture(command, input_text, timeout, cwd=None, env=None):
    return run_process(command, timeout, cwd, env, input_text=input_text)
