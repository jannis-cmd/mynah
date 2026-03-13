package main

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/ErniConcepts/mynah/internal/storage"
)

func TestRunShowPrintsCurrentState(t *testing.T) {
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
	if err := store.WriteUserProfile("anna", "Name: Anna.\nPrefers concise answers."); err != nil {
		t.Fatalf("write user profile: %v", err)
	}

	output := captureStdout(t, func() {
		if err := runShow([]string{"--tenant", "tenant", "--agent", "bella", "--user", "anna", "--data", dataDir}); err != nil {
			t.Fatalf("run show: %v", err)
		}
	})

	if strings.Contains(output, "POLICY.json") || strings.Contains(output, "Provenance") || strings.Contains(output, "Rejected Memory") {
		t.Fatalf("expected legacy inspection sections to be absent, got %q", output)
	}
	if !strings.Contains(strings.ToLower(output), "blue gate") {
		t.Fatalf("expected stored memory message in output, got %q", output)
	}
	if !strings.Contains(output, "=== USER.md (anna) ===") || !strings.Contains(output, "Prefers concise answers.") {
		t.Fatalf("expected user section in output, got %q", output)
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
