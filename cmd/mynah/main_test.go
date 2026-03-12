package main

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ErniConcepts/mynah/internal/storage"
)

func TestRunShowPrintsProvenance(t *testing.T) {
	dataDir := t.TempDir()
	paths := storage.NewAgentPaths(dataDir, "tenant", "bella")
	if err := storage.EnsureAgentPaths(paths); err != nil {
		t.Fatalf("ensure agent paths: %v", err)
	}

	store := storage.NewFileStore(paths, 2200, 1375)
	if err := store.WriteProfile("## Identity\n- Bella is a horse twin agent."); err != nil {
		t.Fatalf("write profile: %v", err)
	}
	if err := store.WriteMemory("The barn uses the blue gate."); err != nil {
		t.Fatalf("write memory: %v", err)
	}
	if err := store.WriteMemoryProvenance(storage.RevisionProvenance{
		UserID:    "anna",
		SessionID: "sess_anna_1",
		Timestamp: time.Date(2026, 3, 12, 18, 0, 0, 0, time.UTC),
		Reason:    "stored shared fact",
		Message:   "The barn uses the blue gate.",
	}); err != nil {
		t.Fatalf("write memory provenance: %v", err)
	}
	if err := store.WriteUserProfile("anna", "Name: Anna.\nPrefers concise answers."); err != nil {
		t.Fatalf("write user profile: %v", err)
	}
	if err := store.WriteUserProfileProvenance("anna", storage.RevisionProvenance{
		UserID:    "anna",
		SessionID: "sess_anna_1",
		Timestamp: time.Date(2026, 3, 12, 18, 1, 0, 0, time.UTC),
		Reason:    "stored user preference",
		Message:   "My name is Anna and I like concise answers.",
	}); err != nil {
		t.Fatalf("write user provenance: %v", err)
	}

	output := captureStdout(t, func() {
		if err := runShow([]string{"--tenant", "tenant", "--agent", "bella", "--user", "anna", "--data", dataDir}); err != nil {
			t.Fatalf("run show: %v", err)
		}
	})

	if !strings.Contains(output, "=== MEMORY Provenance ===") {
		t.Fatalf("expected memory provenance section, got %q", output)
	}
	if !strings.Contains(output, "session_id: sess_anna_1") {
		t.Fatalf("expected session id in provenance, got %q", output)
	}
	if !strings.Contains(strings.ToLower(output), "blue gate") {
		t.Fatalf("expected stored memory message in output, got %q", output)
	}
	if !strings.Contains(output, "=== USER Provenance (anna) ===") {
		t.Fatalf("expected user provenance section, got %q", output)
	}
}

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stdout
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe stdout: %v", err)
	}
	os.Stdout = writer
	defer func() { os.Stdout = old }()

	done := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, reader)
		done <- buf.String()
	}()

	fn()
	_ = writer.Close()
	return <-done
}
