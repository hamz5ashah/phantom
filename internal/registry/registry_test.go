package registry

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/hamz5ashah/phantom/internal/store"
)

func digest(b []byte) string {
	h := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(h[:])
}

func setup(t *testing.T, served []byte, declared []byte) (*Client, *int) {
	ranges := new(int)
	m := store.Manifest{SchemaVersion: 2, Layers: []store.Layer{{MediaType: store.MediaModel, Digest: digest(declared), Size: int64(len(declared))}}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v2/library/m/manifests/latest", func(w http.ResponseWriter, _ *http.Request) { json.NewEncoder(w).Encode(m) })
	mux.HandleFunc("GET /v2/library/m/blobs/{d}", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") != "" {
			*ranges++
		}
		http.ServeContent(w, r, "", time.Time{}, bytes.NewReader(served))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	st := &store.Store{Dir: t.TempDir()}
	os.MkdirAll(filepath.Join(st.Dir, "blobs"), 0o755)
	return &Client{Store: st, BaseURL: srv.URL}, ranges
}

func TestPullResumesPartialDownload(t *testing.T) {
	data := bytes.Repeat([]byte("weights-"), 1000)
	c, ranges := setup(t, data, data)
	p, _ := c.Store.BlobPath(digest(data))
	os.WriteFile(p+".partial", data[:3000], 0o644)

	if err := c.Pull(context.Background(), "m", nil); err != nil {
		t.Fatal(err)
	}
	if *ranges != 1 {
		t.Fatalf("expected a ranged (resumed) request, got %d", *ranges)
	}
	got, _ := os.ReadFile(p)
	if !bytes.Equal(got, data) {
		t.Fatal("blob content differs")
	}
	if _, err := c.Store.Get("m"); err != nil {
		t.Fatal(err)
	}
}

func TestPullRejectsCorruptBlob(t *testing.T) {
	good := []byte("the real weights")
	bad := []byte("tampered weights")
	c, _ := setup(t, bad, good)
	if err := c.Pull(context.Background(), "m", nil); err == nil {
		t.Fatal("expected a digest error")
	}
	if _, err := c.Store.Get("m"); err == nil {
		t.Fatal("a corrupt model must not be recorded")
	}
}
