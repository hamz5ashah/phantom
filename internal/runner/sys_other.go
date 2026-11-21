//go:build !windows

package runner

import "os/exec"

func hideWindowOS(*exec.Cmd) {}
