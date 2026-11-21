package setup

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"
)

// Real file names from llama.cpp's release-publish workflow.
var releaseAssets = []asset{
	{Name: "cudart-llama-bin-win-cuda-12.4-x64.zip"},
	{Name: "cudart-llama-b7000-bin-ubuntu-cuda-12.8-x64.tar.gz"},
	{Name: "llama-b7000-bin-macos-arm64.tar.gz"},
	{Name: "llama-b7000-bin-macos-x64.tar.gz"},
	{Name: "llama-b7000-bin-ubuntu-x64.tar.gz"},
	{Name: "llama-b7000-bin-ubuntu-arm64.tar.gz"},
	{Name: "llama-b7000-bin-ubuntu-s390x.tar.gz"},
	{Name: "llama-b7000-bin-ubuntu-26.04-s390x.tar.gz"},
	{Name: "llama-b7000-bin-ubuntu-vulkan-x64.tar.gz"},
	{Name: "llama-b7000-bin-ubuntu-vulkan-arm64.tar.gz"},
	{Name: "llama-b7000-bin-ubuntu-cuda-12.8-x64.tar.gz"},
	{Name: "llama-b7000-bin-ubuntu-rocm-10.0-x64.tar.gz"},
	{Name: "llama-b7000-bin-ubuntu-openvino-2026.4.1-x64.tar.gz"},
	{Name: "llama-b7000-bin-linux-arm64-snapdragon.tar.gz"},
	{Name: "llama-b7000-bin-win-cpu-x64.zip"},
	{Name: "llama-b7000-bin-win-cpu-arm64.zip"},
	{Name: "llama-b7000-bin-win-cuda-12.4-x64.zip"},
	{Name: "llama-b7000-bin-win-vulkan-x64.zip"},
	{Name: "llama-b7000-bin-win-vulkan-arm64.zip"},
	{Name: "llama-b7000-xcframework.zip"},
}

func TestPick(t *testing.T) {
	cases := []struct{ goos, arch, backend, want string }{
		{"windows", "amd64", "vulkan", "llama-b7000-bin-win-vulkan-x64.zip"},
		{"windows", "amd64", "cpu", "llama-b7000-bin-win-cpu-x64.zip"},
		{"windows", "amd64", "cuda", "llama-b7000-bin-win-cuda-12.4-x64.zip"},
		{"windows", "arm64", "cpu", "llama-b7000-bin-win-cpu-arm64.zip"},
		{"linux", "amd64", "cpu", "llama-b7000-bin-ubuntu-x64.tar.gz"},
		{"linux", "amd64", "vulkan", "llama-b7000-bin-ubuntu-vulkan-x64.tar.gz"},
		{"darwin", "arm64", "metal", "llama-b7000-bin-macos-arm64.tar.gz"},
	}
	for _, c := range cases {
		a, err := pick(releaseAssets, c.goos, c.arch, c.backend)
		if err != nil || a.Name != c.want {
			t.Errorf("%s/%s/%s: got %q (%v), want %q", c.goos, c.arch, c.backend, a.Name, err, c.want)
		}
	}
	if a, _ := pick(releaseAssets, "linux", "arm64", "cpu"); a.Name != "llama-b7000-bin-ubuntu-arm64.tar.gz" {
		t.Errorf("linux arm64 cpu: got %q", a.Name)
	}
	if _, err := pick(releaseAssets, "linux", "arm64", "rocm"); err == nil {
		t.Error("expected an error when no build exists")
	}
}

func TestUnzipRejectsPathTraversal(t *testing.T) {
	dir := t.TempDir()
	zp := filepath.Join(dir, "evil.zip")
	f, _ := os.Create(zp)
	zw := zip.NewWriter(f)
	w, _ := zw.Create("../escape.txt")
	w.Write([]byte("x"))
	zw.Close()
	f.Close()
	if err := unzip(zp, filepath.Join(dir, "out")); err == nil {
		t.Fatal("expected unsafe path error")
	}
	if _, err := os.Stat(filepath.Join(dir, "escape.txt")); err == nil {
		t.Fatal("file escaped the target folder")
	}
}
