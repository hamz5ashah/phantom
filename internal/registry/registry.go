// Package registry downloads models from Ollama-compatible registries:
// the Ollama library (registry.ollama.ai) and Hugging Face GGUF repos (hf.co).
package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/hamz5ashah/phantom/internal/store"
)

// Progress is reported while pulling; it matches Ollama's /api/pull stream.
type Progress struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

// Client pulls models into a store.
type Client struct {
	HTTP  *http.Client
	Store *store.Store
	// BaseURL overrides the registry address for every host (tests).
	BaseURL string
}

func (c *Client) base(n store.Name) string {
	if c.BaseURL != "" {
		return strings.TrimRight(c.BaseURL, "/")
	}
	return "https://" + n.Host
}

func (c *Client) client() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	return &http.Client{Timeout: 0}
}

// Pull downloads a model and its blobs, resuming partial downloads and
// verifying every blob's sha256 before the model is recorded.
func (c *Client) Pull(ctx context.Context, name string, progress func(Progress)) error {
	if progress == nil {
		progress = func(Progress) {}
	}
	n, err := store.ParseName(name)
	if err != nil {
		return err
	}
	progress(Progress{Status: "pulling manifest"})
	m, err := c.manifest(ctx, n)
	if err != nil {
		return err
	}
	if _, ok := m.Layer(store.MediaModel); !ok {
		return fmt.Errorf("%s has no GGUF weights layer", n.String())
	}
	blobs := []store.Layer{}
	if m.Config.Digest != "" {
		blobs = append(blobs, m.Config)
	}
	for _, l := range m.Layers {
		// Weights, vision projector and config are all Phantom needs; the
		// chat template ships inside the GGUF file itself.
		if l.MediaType == store.MediaModel || l.MediaType == store.MediaProjector {
			blobs = append(blobs, l)
		}
	}
	for _, l := range blobs {
		if err := c.blob(ctx, n, l, progress); err != nil {
			return err
		}
	}
	progress(Progress{Status: "writing manifest"})
	kept := &store.Manifest{SchemaVersion: m.SchemaVersion, MediaType: m.MediaType, Config: m.Config}
	for _, l := range blobs {
		if l != m.Config {
			kept.Layers = append(kept.Layers, l)
		}
	}
	if err := c.Store.SaveManifest(n, kept); err != nil {
		return err
	}
	progress(Progress{Status: "success"})
	return nil
}

func (c *Client) manifest(ctx context.Context, n store.Name) (*store.Manifest, error) {
	url := fmt.Sprintf("%s/v2/%s/%s/manifests/%s", c.base(n), n.Namespace, n.Repo, n.Tag)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := c.client().Do(req)
	if err != nil {
		return nil, fmt.Errorf("could not reach %s: %w", n.Host, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("model %q not found in %s", n.String(), n.Host)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("%s returned %s: %s", n.Host, resp.Status, strings.TrimSpace(string(body)))
	}
	var m store.Manifest
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&m); err != nil {
		return nil, fmt.Errorf("bad manifest from %s: %w", n.Host, err)
	}
	return &m, nil
}

func (c *Client) blob(ctx context.Context, n store.Name, l store.Layer, progress func(Progress)) error {
	dst, err := c.Store.BlobPath(l.Digest)
	if err != nil {
		return err
	}
	if fi, err := os.Stat(dst); err == nil && fi.Size() == l.Size {
		progress(Progress{Status: "pulling " + short(l.Digest), Digest: l.Digest, Total: l.Size, Completed: l.Size})
		return nil
	}
	part := dst + ".partial"
	f, err := os.OpenFile(part, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	have, _ := f.Seek(0, io.SeekEnd)
	if have > l.Size {
		if err := f.Truncate(0); err != nil {
			return err
		}
		have, _ = f.Seek(0, io.SeekStart)
	}

	if have < l.Size {
		url := fmt.Sprintf("%s/v2/%s/%s/blobs/%s", c.base(n), n.Namespace, n.Repo, l.Digest)
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		if have > 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-", have))
		}
		resp, err := c.client().Do(req)
		if err != nil {
			return fmt.Errorf("download of %s failed: %w", short(l.Digest), err)
		}
		defer resp.Body.Close()
		switch resp.StatusCode {
		case http.StatusPartialContent:
		case http.StatusOK:
			// The server ignored the range: start over.
			if err := f.Truncate(0); err != nil {
				return err
			}
			if have, err = f.Seek(0, io.SeekStart); err != nil {
				return err
			}
		default:
			return fmt.Errorf("download of %s failed: %s", short(l.Digest), resp.Status)
		}
		w := &counter{w: f, n: have, report: func(done int64) {
			progress(Progress{Status: "pulling " + short(l.Digest), Digest: l.Digest, Total: l.Size, Completed: done})
		}}
		if _, err := io.Copy(w, resp.Body); err != nil {
			return fmt.Errorf("download of %s interrupted (run pull again to resume): %w", short(l.Digest), err)
		}
		w.flush()
	}
	if err := f.Close(); err != nil {
		return err
	}

	progress(Progress{Status: "verifying sha256 digest", Digest: l.Digest})
	if err := verify(part, l.Digest); err != nil {
		_ = os.Remove(part)
		return err
	}
	return os.Rename(part, dst)
}

func verify(path, digest string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if got := "sha256:" + hex.EncodeToString(h.Sum(nil)); got != digest {
		return errors.New("downloaded file is corrupt (digest mismatch); run pull again")
	}
	return nil
}

func short(digest string) string {
	d := strings.TrimPrefix(digest, "sha256:")
	if len(d) > 12 {
		d = d[:12]
	}
	return d
}

// counter reports download progress at most a few times per second.
type counter struct {
	w      io.Writer
	n      int64
	last   time.Time
	report func(int64)
}

func (c *counter) Write(p []byte) (int, error) {
	k, err := c.w.Write(p)
	c.n += int64(k)
	if time.Since(c.last) > 250*time.Millisecond {
		c.flush()
	}
	return k, err
}

func (c *counter) flush() {
	c.last = time.Now()
	c.report(c.n)
}
