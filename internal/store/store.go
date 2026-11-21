// Package store keeps downloaded models on disk.
//
// Layout (under PHANTOM_MODELS, default ~/.phantom/models):
//
//	blobs/sha256-<hex>                          content-addressed files (GGUF weights, configs)
//	manifests/<host>/<namespace>/<repo>/<tag>   the registry manifest that names those blobs
package store

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Media types used by Ollama-compatible registries (registry.ollama.ai, hf.co).
const (
	MediaModel     = "application/vnd.ollama.image.model"
	MediaProjector = "application/vnd.ollama.image.projector"
	MediaConfig    = "application/vnd.docker.container.image.v1+json"
)

// Layer is one blob referenced by a manifest.
type Layer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
}

// Manifest is the registry manifest saved for a model.
type Manifest struct {
	SchemaVersion int     `json:"schemaVersion"`
	MediaType     string  `json:"mediaType,omitempty"`
	Config        Layer   `json:"config"`
	Layers        []Layer `json:"layers"`
}

// Layer returns the first layer of the given media type.
func (m *Manifest) Layer(mediaType string) (Layer, bool) {
	for _, l := range m.Layers {
		if l.MediaType == mediaType {
			return l, true
		}
	}
	return Layer{}, false
}

// Size is the total size of all blobs.
func (m *Manifest) Size() int64 {
	n := m.Config.Size
	for _, l := range m.Layers {
		n += l.Size
	}
	return n
}

// Name identifies a model: host/namespace/repo:tag.
type Name struct {
	Host, Namespace, Repo, Tag string
}

// DefaultHost is the Ollama model library.
const DefaultHost = "registry.ollama.ai"

// ParseName accepts the same names as Ollama:
//
//	qwen3                      -> registry.ollama.ai/library/qwen3:latest
//	qwen3:0.6b                 -> registry.ollama.ai/library/qwen3:0.6b
//	user/model:tag             -> registry.ollama.ai/user/model:tag
//	hf.co/owner/repo[:Q4_K_M]  -> Hugging Face GGUF repository (any quantization tag)
func ParseName(s string) (Name, error) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(strings.TrimPrefix(s, "https://"), "http://")
	if s == "" {
		return Name{}, errors.New("model name is empty")
	}
	n := Name{Host: DefaultHost, Namespace: "library", Tag: "latest"}
	if i := strings.LastIndex(s, ":"); i > strings.LastIndex(s, "/") {
		n.Tag, s = s[i+1:], s[:i]
	}
	parts := strings.Split(s, "/")
	switch len(parts) {
	case 1:
		n.Repo = parts[0]
	case 2:
		n.Namespace, n.Repo = parts[0], parts[1]
	case 3:
		n.Host, n.Namespace, n.Repo = parts[0], parts[1], parts[2]
	default:
		return Name{}, fmt.Errorf("invalid model name %q", s)
	}
	if n.Host == "huggingface.co" {
		n.Host = "hf.co"
	}
	for _, p := range []string{n.Host, n.Namespace, n.Repo, n.Tag} {
		if p == "" || p == "." || p == ".." || strings.ContainsAny(p, `\ `) {
			return Name{}, fmt.Errorf("invalid model name %q", s)
		}
	}
	return n, nil
}

// String is the short display form, like Ollama prints it.
func (n Name) String() string {
	base := n.Repo
	if n.Namespace != "library" || n.Host != DefaultHost {
		base = n.Namespace + "/" + n.Repo
	}
	if n.Host != DefaultHost {
		base = n.Host + "/" + base
	}
	return base + ":" + n.Tag
}

// Store is a models directory.
type Store struct{ Dir string }

// Default opens the models directory: PHANTOM_MODELS, else an existing Ollama
// models folder (same layout, so models downloaded with Ollama are reused),
// else ~/.phantom/models.
func Default() (*Store, error) {
	dir := os.Getenv("PHANTOM_MODELS")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, err
		}
		dir = filepath.Join(home, ".phantom", "models")
		for _, ollama := range []string{os.Getenv("OLLAMA_MODELS"), filepath.Join(home, ".ollama", "models")} {
			if ollama == "" {
				continue
			}
			if fi, err := os.Stat(filepath.Join(ollama, "manifests")); err == nil && fi.IsDir() {
				dir = ollama
				break
			}
		}
	}
	return &Store{Dir: dir}, os.MkdirAll(filepath.Join(dir, "blobs"), 0o755)
}

// BlobPath is where a blob with this digest lives.
func (s *Store) BlobPath(digest string) (string, error) {
	hex, ok := strings.CutPrefix(digest, "sha256:")
	if !ok || len(hex) != 64 || strings.Trim(hex, "0123456789abcdef") != "" {
		return "", fmt.Errorf("invalid digest %q", digest)
	}
	return filepath.Join(s.Dir, "blobs", "sha256-"+hex), nil
}

func (s *Store) manifestPath(n Name) string {
	return filepath.Join(s.Dir, "manifests", n.Host, n.Namespace, n.Repo, n.Tag)
}

// SaveManifest records a fully downloaded model.
func (s *Store) SaveManifest(n Name, m *Manifest) error {
	p := s.manifestPath(n)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, p)
}

// Model is an installed model.
type Model struct {
	Name     Name
	Manifest *Manifest
	Modified time.Time
}

// Get loads an installed model by name.
func (s *Store) Get(name string) (*Model, error) {
	n, err := ParseName(name)
	if err != nil {
		return nil, err
	}
	p := s.manifestPath(n)
	b, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("model %q not found, try pulling it first", n.String())
	} else if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("model %q has a damaged manifest: %w", n.String(), err)
	}
	fi, _ := os.Stat(p)
	mod := time.Time{}
	if fi != nil {
		mod = fi.ModTime()
	}
	return &Model{Name: n, Manifest: &m, Modified: mod}, nil
}

// WeightsPath is the GGUF file of an installed model.
func (s *Store) WeightsPath(m *Model) (string, error) {
	l, ok := m.Manifest.Layer(MediaModel)
	if !ok {
		return "", fmt.Errorf("model %q has no weights layer", m.Name.String())
	}
	return s.BlobPath(l.Digest)
}

// ProjectorPath is the vision projector (mmproj) file, if the model has one.
func (s *Store) ProjectorPath(m *Model) (string, bool) {
	l, ok := m.Manifest.Layer(MediaProjector)
	if !ok {
		return "", false
	}
	p, err := s.BlobPath(l.Digest)
	return p, err == nil
}

// List returns all installed models, newest first.
func (s *Store) List() ([]*Model, error) {
	root := filepath.Join(s.Dir, "manifests")
	var out []*Model
	err := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return nil
			}
			return err
		}
		if d.IsDir() || strings.HasSuffix(p, ".tmp") {
			return nil
		}
		rel, _ := filepath.Rel(root, p)
		parts := strings.Split(filepath.ToSlash(rel), "/")
		if len(parts) != 4 {
			return nil
		}
		name := Name{Host: parts[0], Namespace: parts[1], Repo: parts[2], Tag: parts[3]}
		if m, err := s.Get(name.String()); err == nil {
			out = append(out, m)
		}
		return nil
	})
	sort.Slice(out, func(i, j int) bool { return out[i].Modified.After(out[j].Modified) })
	return out, err
}

// Delete removes a model and any blob no other model still uses.
func (s *Store) Delete(name string) error {
	m, err := s.Get(name)
	if err != nil {
		return err
	}
	if err := os.Remove(s.manifestPath(m.Name)); err != nil {
		return err
	}
	inUse := map[string]bool{}
	others, _ := s.List()
	for _, o := range others {
		inUse[o.Manifest.Config.Digest] = true
		for _, l := range o.Manifest.Layers {
			inUse[l.Digest] = true
		}
	}
	for _, l := range append([]Layer{m.Manifest.Config}, m.Manifest.Layers...) {
		if l.Digest == "" || inUse[l.Digest] {
			continue
		}
		if p, err := s.BlobPath(l.Digest); err == nil {
			_ = os.Remove(p)
		}
	}
	return nil
}

// Details summarises a model from its config blob (family, size, quantization).
type Details struct {
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

// Details reads the model's config blob; missing fields stay empty.
func (s *Store) Details(m *Model) Details {
	d := Details{Format: "gguf"}
	p, err := s.BlobPath(m.Manifest.Config.Digest)
	if err != nil {
		return d
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return d
	}
	var cfg struct {
		ModelFamily   string   `json:"model_family"`
		ModelFamilies []string `json:"model_families"`
		ModelType     string   `json:"model_type"`
		FileType      string   `json:"file_type"`
	}
	if json.Unmarshal(b, &cfg) == nil {
		d.Family, d.Families, d.ParameterSize, d.QuantizationLevel =
			cfg.ModelFamily, cfg.ModelFamilies, cfg.ModelType, cfg.FileType
	}
	return d
}
