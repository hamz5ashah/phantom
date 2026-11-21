package store

import "testing"

func TestParseName(t *testing.T) {
	cases := map[string]string{
		"qwen3":         "registry.ollama.ai/library/qwen3:latest",
		"qwen3:0.6b":    "registry.ollama.ai/library/qwen3:0.6b",
		"user/model:v2": "registry.ollama.ai/user/model:v2",
		"hf.co/bartowski/Llama-3.2-1B-GGUF:Q4_K_M": "hf.co/bartowski/Llama-3.2-1B-GGUF:Q4_K_M",
		"https://huggingface.co/owner/repo":        "hf.co/owner/repo:latest",
	}
	for in, want := range cases {
		n, err := ParseName(in)
		if err != nil {
			t.Fatalf("%s: %v", in, err)
		}
		if got := n.Host + "/" + n.Namespace + "/" + n.Repo + ":" + n.Tag; got != want {
			t.Errorf("%s: got %s, want %s", in, got, want)
		}
	}
	for _, bad := range []string{"", "a/b/c/d", "../x", "a b"} {
		if _, err := ParseName(bad); err == nil {
			t.Errorf("%q should be rejected", bad)
		}
	}
	if s := mustParse(t, "qwen3:0.6b").String(); s != "qwen3:0.6b" {
		t.Errorf("short name: %s", s)
	}
	if s := mustParse(t, "hf.co/a/b:Q8_0").String(); s != "hf.co/a/b:Q8_0" {
		t.Errorf("hf short name: %s", s)
	}
}

func mustParse(t *testing.T, s string) Name {
	n, err := ParseName(s)
	if err != nil {
		t.Fatal(err)
	}
	return n
}

func TestBlobPathRejectsBadDigests(t *testing.T) {
	s := &Store{Dir: t.TempDir()}
	for _, d := range []string{"sha256:../../etc", "md5:abc", "sha256:XYZ"} {
		if _, err := s.BlobPath(d); err == nil {
			t.Errorf("%q should be rejected", d)
		}
	}
}
