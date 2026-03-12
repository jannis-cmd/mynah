package storage

import (
	"os"
	"path/filepath"
	"testing"
	"time"
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

func TestFileStorePersistsRevisionProvenance(t *testing.T) {
	root := t.TempDir()
	paths := NewAgentPaths(root, "tenant-a", "agent-b")
	if err := EnsureAgentPaths(paths); err != nil {
		t.Fatalf("ensure agent paths: %v", err)
	}

	store := NewFileStore(paths, 2200, 1375)
	now := time.Date(2026, 3, 12, 18, 0, 0, 0, time.UTC)
	memoryMeta := RevisionProvenance{
		UserID:    "anna",
		SessionID: "sess_anna_1",
		Timestamp: now,
		Reason:    "stored shared fact",
		Message:   "The barn uses the blue gate.",
	}
	if err := store.WriteMemoryProvenance(memoryMeta); err != nil {
		t.Fatalf("write memory provenance: %v", err)
	}
	userMeta := RevisionProvenance{
		UserID:    "anna",
		SessionID: "sess_anna_1",
		Timestamp: now,
		Reason:    "stored user preference",
		Message:   "I like concise answers.",
	}
	if err := store.WriteUserProfileProvenance("anna", userMeta); err != nil {
		t.Fatalf("write user provenance: %v", err)
	}

	gotMemory, err := store.ReadMemoryProvenance()
	if err != nil {
		t.Fatalf("read memory provenance: %v", err)
	}
	if gotMemory.Target != "memory" || gotMemory.SessionID != "sess_anna_1" || gotMemory.Reason != "stored shared fact" {
		t.Fatalf("unexpected memory provenance: %+v", gotMemory)
	}

	gotUser, err := store.ReadUserProfileProvenance("anna")
	if err != nil {
		t.Fatalf("read user provenance: %v", err)
	}
	if gotUser.Target != "user" || gotUser.UserID != "anna" || gotUser.Reason != "stored user preference" {
		t.Fatalf("unexpected user provenance: %+v", gotUser)
	}
}
