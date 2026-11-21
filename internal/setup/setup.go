// Package setup downloads a prebuilt llama.cpp (llama-server) for this PC
// from the official ggml-org/llama.cpp GitHub releases.
package setup

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// ReleasesAPI lists recent llama.cpp releases (overridable in tests).
var ReleasesAPI = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8"

// RuntimeDir is where llama-server is installed: PHANTOM_RUNTIME or ~/.phantom/runtime.
func RuntimeDir() string {
	if d := os.Getenv("PHANTOM_RUNTIME"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".phantom", "runtime")
}

type asset struct {
	Name string `json:"name"`
	URL  string `json:"browser_download_url"`
	Size int64  `json:"size"`
}

// DefaultBackend picks the GPU backend that works on most machines:
// Metal on Apple Silicon, Vulkan elsewhere (it also runs on the CPU).
func DefaultBackend() string {
	if runtime.GOOS == "darwin" {
		return "metal"
	}
	return "vulkan"
}

// pick chooses the release asset for an OS, CPU architecture and backend.
func pick(assets []asset, goos, arch, backend string) (asset, error) {
	osTok := map[string]string{"windows": "-win-", "linux": "-ubuntu-", "darwin": "-macos-"}[goos]
	archTok := map[string]string{"amd64": "x64", "arm64": "arm64"}[arch]
	if osTok == "" || archTok == "" {
		return asset{}, fmt.Errorf("no prebuilt llama.cpp for %s/%s; build llama-server yourself and set PHANTOM_LLAMA_SERVER", goos, arch)
	}
	var best *asset
	for i, a := range assets {
		n := strings.ToLower(a.Name)
		if !strings.HasPrefix(n, "llama-") || !strings.Contains(n, "-bin"+osTok) || !strings.Contains(n, archTok) ||
			strings.Contains(n, "cudart") || !(strings.HasSuffix(n, ".zip") || strings.HasSuffix(n, ".tar.gz")) {
			continue
		}
		hasBackend := func(b string) bool { return strings.Contains(n, "-"+b+"-") }
		var ok bool
		switch backend {
		case "cpu":
			ok = (goos == "windows" && hasBackend("cpu")) ||
				(goos == "linux" && !hasBackend("vulkan") && !hasBackend("cuda") && !hasBackend("rocm") && !hasBackend("sycl") && !hasBackend("openvino") && !hasBackend("zdnn") && !strings.Contains(n, "s390x"))
		case "metal":
			ok = goos == "darwin"
		default:
			ok = hasBackend(backend)
		}
		if ok && (best == nil || len(a.Name) < len(best.Name)) {
			best = &assets[i]
		}
	}
	if best == nil {
		return asset{}, fmt.Errorf("the latest llama.cpp release has no %s build for %s/%s", backend, goos, arch)
	}
	return *best, nil
}

// Install downloads and unpacks llama-server into dir. It returns the binary path.
func Install(ctx context.Context, dir, backend string, log func(string)) (string, error) {
	if log == nil {
		log = func(string) {}
	}
	if backend == "" {
		backend = DefaultBackend()
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, ReleasesAPI, nil)
	req.Header.Set("Accept", "application/vnd.github+json")
	if tok := os.Getenv("GITHUB_TOKEN"); tok != "" {
		// Avoids GitHub's anonymous rate limit on shared machines (CI).
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("could not reach GitHub: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("GitHub returned %s", resp.Status)
	}
	var releases []struct {
		Tag        string  `json:"tag_name"`
		Draft      bool    `json:"draft"`
		Prerelease bool    `json:"prerelease"`
		Assets     []asset `json:"assets"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&releases); err != nil {
		return "", err
	}
	// llama.cpp publishes several times a day and a brand-new release can
	// still be missing files, so take the newest one that has our build.
	var rel struct {
		Tag string
	}
	var a asset
	err = errors.New("no llama.cpp releases found")
	for _, r := range releases {
		if r.Draft || r.Prerelease {
			continue
		}
		if a, err = pick(r.Assets, runtime.GOOS, runtime.GOARCH, backend); err == nil {
			rel.Tag = r.Tag
			break
		}
	}
	if err != nil {
		if len(releases) > 0 {
			names := []string{}
			for _, x := range releases[0].Assets {
				names = append(names, x.Name)
			}
			err = fmt.Errorf("%w (files in %s: %s)", err, releases[0].Tag, strings.Join(names, ", "))
		}
		return "", err
	}
	log(fmt.Sprintf("Downloading llama.cpp %s (%s, %d MB)...", rel.Tag, a.Name, a.Size>>20))

	tmp, err := os.CreateTemp("", "phantom-llama-*")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())
	defer tmp.Close()
	dl, _ := http.NewRequestWithContext(ctx, http.MethodGet, a.URL, nil)
	dresp, err := http.DefaultClient.Do(dl)
	if err != nil {
		return "", err
	}
	defer dresp.Body.Close()
	if dresp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download failed: %s", dresp.Status)
	}
	if _, err := io.Copy(tmp, dresp.Body); err != nil {
		return "", err
	}

	staging := dir + ".new"
	_ = os.RemoveAll(staging)
	if strings.HasSuffix(strings.ToLower(a.Name), ".zip") {
		err = unzip(tmp.Name(), staging)
	} else {
		err = untargz(tmp.Name(), staging)
	}
	if err != nil {
		_ = os.RemoveAll(staging)
		return "", err
	}
	// Archives nest the binaries (e.g. build/bin/); keep that folder's contents.
	binDir, err := findBinaryDir(staging)
	if err != nil {
		_ = os.RemoveAll(staging)
		return "", err
	}
	_ = os.RemoveAll(dir)
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		return "", err
	}
	if err := os.Rename(binDir, dir); err != nil {
		return "", err
	}
	_ = os.RemoveAll(staging)
	_ = os.WriteFile(filepath.Join(dir, "VERSION"), []byte(rel.Tag+" "+a.Name+"\n"), 0o644)
	bin := filepath.Join(dir, binaryName())
	log("Installed llama.cpp " + rel.Tag + " to " + dir)
	return bin, nil
}

func binaryName() string {
	if runtime.GOOS == "windows" {
		return "llama-server.exe"
	}
	return "llama-server"
}

func findBinaryDir(root string) (string, error) {
	var found string
	_ = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && d.Name() == binaryName() && found == "" {
			found = filepath.Dir(p)
		}
		return nil
	})
	if found == "" {
		return "", errors.New("llama-server is missing from the downloaded archive")
	}
	return found, nil
}

// safeJoin rejects archive entries that would escape the target folder.
func safeJoin(root, name string) (string, error) {
	p := filepath.Join(root, name)
	if !strings.HasPrefix(p, filepath.Clean(root)+string(os.PathSeparator)) {
		return "", fmt.Errorf("unsafe path in archive: %s", name)
	}
	return p, nil
}

func unzip(src, dst string) error {
	zr, err := zip.OpenReader(src)
	if err != nil {
		return err
	}
	defer zr.Close()
	for _, f := range zr.File {
		p, err := safeJoin(dst, f.Name)
		if err != nil {
			return err
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(p, 0o755); err != nil {
				return err
			}
			continue
		}
		if f.Mode()&os.ModeSymlink != 0 {
			continue
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		err = writeFile(p, rc, f.Mode()|0o644)
		rc.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

func untargz(src, dst string) error {
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return nil
		} else if err != nil {
			return err
		}
		p, err := safeJoin(dst, h.Name)
		if err != nil {
			return err
		}
		switch h.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(p, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				return err
			}
			if err := writeFile(p, tr, os.FileMode(h.Mode)|0o644); err != nil {
				return err
			}
		case tar.TypeSymlink:
			// Shared libraries are often symlinked (libfoo.so -> libfoo.so.1).
			if strings.Contains(h.Linkname, "..") || filepath.IsAbs(h.Linkname) {
				continue
			}
			_ = os.MkdirAll(filepath.Dir(p), 0o755)
			_ = os.Symlink(h.Linkname, p)
		}
	}
}

func writeFile(p string, r io.Reader, mode os.FileMode) error {
	out, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, r); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
