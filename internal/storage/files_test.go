package storage

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFileStoreWritesAtomicallyAndTrimmed(t *testing.T) {
	root := t.TempDir()
	paths := NewAgentPaths(root, "tenant-a", "agent-b")
	if err := EnsureAgentPaths(paths); err != nil {
		t.Fatalf("ensure agent paths: %v", err)
	}

	store := NewFileStore(paths, 2200, 1375)
	if err := store.WriteMemory("  remembered fact  "); err != nil {
		t.Fatalf("write memory: %v", err)
	}

	raw, err := os.ReadFile(paths.MemoryPath)
	if err != nil {
		t.Fatalf("read memory: %v", err)
	}
	if string(raw) != "remembered fact\n" {
		t.Fatalf("unexpected memory content: %q", string(raw))
	}

	matches, err := filepath.Glob(filepath.Join(paths.RootPath, ".mynah-*"))
	if err != nil {
		t.Fatalf("glob temp files: %v", err)
	}
	if len(matches) != 0 {
		t.Fatalf("expected no temp files left behind, found %v", matches)
	}
}

func TestEnsureAgentPathsCreatesCoreFiles(t *testing.T) {
	root := t.TempDir()
	paths := NewAgentPaths(root, "tenant-a", "agent-b")
	if err := EnsureAgentPaths(paths); err != nil {
		t.Fatalf("ensure agent paths: %v", err)
	}

	for _, path := range []string{paths.RootPath, paths.UsersPath, paths.MemoryPath, paths.ProfilePath} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected path %s to exist: %v", path, err)
		}
	}
}

func TestFileStoreWritesUserProfile(t *testing.T) {
	root := t.TempDir()
	paths := NewAgentPaths(root, "tenant-a", "agent-b")
	if err := EnsureAgentPaths(paths); err != nil {
		t.Fatalf("ensure agent paths: %v", err)
	}

	store := NewFileStore(paths, 2200, 1375)
	if err := store.WriteUserProfile("anna", "  Prefers concise answers.  "); err != nil {
		t.Fatalf("write user profile: %v", err)
	}

	got, err := store.ReadUserProfile("anna")
	if err != nil {
		t.Fatalf("read user profile: %v", err)
	}
	if got != "Prefers concise answers." {
		t.Fatalf("unexpected user profile: %q", got)
	}
}
