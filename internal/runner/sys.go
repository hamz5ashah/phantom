package runner

import (
	"os/exec"
	"path/filepath"
	"runtime"
)

var isWindows = runtime.GOOS == "windows"

func joinDir(exe, name string) string { return filepath.Join(filepath.Dir(exe), name) }

// hideWindow keeps llama-server from opening a console window on Windows.
func hideWindow(cmd *exec.Cmd) { hideWindowOS(cmd) }
